#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_scene():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    def forward_batched(self, input):
        """
        Handles batched input of shape (B, N, L+3)
        """
        B, N, _ = input.shape
        xyz = input[..., -3:]  # Shape: (B, N, 3)

        if input.shape[-1] > 3 and self.latent_dropout:
            latent_vecs = input[..., :-3]  # Shape: (B, N, L)
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], dim=-1)  # Shape: (B, N, L+3)
        else:
            x = input  # Shape: (B, N, L+3)

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                # Shape: (B, N, 2*L+3) or similar
                x = torch.cat([x, input], dim=-1)
            elif layer != 0 and self.xyz_in_all:
                # Shape: (B, N, current_dim+3)
                x = torch.cat([x, xyz], dim=-1)
            x = lin(x)
            # Apply tanh on the last layer if use_tanh is True
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            # Apply ReLU and optionally BatchNorm on intermediate layers
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    # BatchNorm1d expects (B, C, L) but we have (B, N, C), so we
                    # need to permute
                    x = x.permute(0, 2, 1)  # Shape: (B, C, N)
                    x = bn(x)
                    x = x.permute(0, 2, 1)  # Shape: (B, N, C)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

    def forward_batched_naive(self, input):
        """
        Handles batched input of shape (B, N, L+3) by calling forward_single()
        for each item in the batch.
        """
        B, N, _ = input.shape
        outputs = []

        for b in range(B):
            single_input = input[b]  # Shape: (N, L+3)
            single_output = self.forward_single(single_input)  # Process a single item
            outputs.append(single_output)

        # Concatenate all single outputs along the batch dimension Each
        # single_output is expected to have shape (N, 1), so the result will
        # have shape (B, N, 1)
        outputs = torch.stack(outputs, dim=0)

        return outputs

    # input: N x (L+3)
    def forward_single(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

    def forward(self, input):
        """
        Forward pass for the decoder module that supports both single and batched inputs.

        This method automatically handles inputs of different dimensions: - If
        the input is of shape (N, L+3), it's treated as a single input. - If the
        input is of shape (B, N, L+3), it's treated as batched input.

        For batched inputs, this method calls both the optimized batched forward
        method and a naive batched method that processes inputs one-by-one and
        compares their results.

        Args:
            input (torch.Tensor): Input tensor of shape (N, L+3) for single
                input or (B, N, L+3) for batched input.

        Returns:
            torch.Tensor: The output of the decoder. Shape (N, 1) for single
                input, (B, N, 1) for batched input.
        """
        if input.dim() == 2:
            # (N, 1)
            return self.forward_single(input)
        elif input.dim() == 3:
            # (B, N, 1)
            return self.forward_batched(input)
        else:
            raise ValueError("Unsupported input dimensions")
