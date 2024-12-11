import random
import shutil
from pathlib import Path

import configargparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from lit.network_utils import eval_raykeep_metrics, get_embed_fn, load_raykeep_dataset


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


class RayKeepNet(nn.Module):
    def __init__(
        self,
        net_depth=4,
        net_width=128,
        input_ch=3,
        output_ch=1,
    ):
        """ """
        super(RayKeepNet, self).__init__()
        self.net_depth = net_depth
        self.net_width = net_width
        self.input_ch = input_ch

        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_ch, net_width)]
            + [nn.Linear(net_width, net_width) for i in range(net_depth - 1)]
        )
        self.output_layer = nn.Linear(net_width, output_ch)

        self.linear_layers.apply(RayKeepNet.init_weights)
        self.output_layer.apply(RayKeepNet.init_weights)

    def forward(self, x):
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)


def config_parser():
    parser = configargparse.ArgumentParser()

    # Directory options.
    parser.add_argument(
        "--config",
        is_config_file=True,
        default="raykeep.cfg",
        help="config file path",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="raykeep",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./log",
        help="where to store ckpts and logs",
    )
    parser.add_argument(
        "--no_reload",
        action="store_true",
        help="do not reload weights from saved ckpt",
    )

    # Training options
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluate mode only.",
    )
    parser.add_argument(
        "--net_depth",
        type=int,
        default=8,
        help="Number of ayers in network",
    )
    parser.add_argument(
        "--net_width",
        type=int,
        default=256,
        help="Channels per layer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size (number of random rays per gradient step)",
    )
    parser.add_argument(
        "--lrate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=500,
        help="Exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--total_iters",
        type=int,
        default=500000,
    )

    # Rendering options
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )

    # Iteration options.
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="Frequency of console printout and metric logging",
    )
    parser.add_argument(
        "--i_weights",
        type=int,
        default=10000,
        help="Frequency of weight ckpt saving",
    )

    # Loss type
    parser.add_argument(
        "--loss_type",
        type=str,
        default="img2mse",
        help="Options: img2mse / mseloss / l1loss",
    )

    return parser


def run_network(inputs, model, embed_fn, embed_dir_fn):
    """
    Prepares inputs and applies network.
    """
    ray_dirs, dist, incident = inputs[:, :3], inputs[:, 3], inputs[:, 4]
    embedded_inputs = torch.cat(
        (
            embed_dir_fn(ray_dirs),
            embed_fn(incident.unsqueeze(1)),
            embed_fn(dist.unsqueeze(1)),
        ),
        dim=1,
    )
    outputs = model(embedded_inputs)
    return outputs


def evaluate_in_batches(
    inputs,
    model,
    embed_fn,
    embed_dir_fn,
    device,
    batch_size=1024,
):
    def batch_generator(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in batch_generator(inputs, batch_size):
            batch = torch.tensor(batch).to(device)
            preds = run_network(batch, model, embed_fn, embed_dir_fn)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def main():
    parser = config_parser()
    args = parser.parse_args()

    # Prepare states.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Select loss function
    loss_dict = {
        "img2mse": lambda x, y: torch.mean((x - y) ** 2),
        "mseloss": nn.MSELoss(),
        "bceloss": nn.BCELoss(),
        "l1loss": nn.L1Loss(reduction="mean"),
    }
    print(f"Selected loss type: {args.loss_type}")
    rgb_loss = loss_dict[args.loss_type]

    # Prepare loggings.
    base_dir = args.base_dir
    exp_name = args.exp_name
    log_dir = Path(base_dir) / exp_name
    log_args_path = log_dir / "args.txt"
    log_config_path = log_dir / "config.txt"

    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_args_path, "w", encoding="utf-8") as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            f.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        shutil.copy(args.config, log_config_path)

    # Embedder
    # Input: [dir_x, dir_y, dir_z, dist, incident_angle]
    # Embedder for dist, incident_angle
    embed_fn, input_ch = get_embed_fn(
        args.multires,
        input_dims=1,
    )
    # Embedder for (dir_x, dir_y, dir_z)
    embed_dir_fn, input_ch_views = get_embed_fn(
        args.multires,  # args.multires_views
        input_dims=3,
    )
    total_input_ch = input_ch * 2 + input_ch_views

    # Model
    model = RayKeepNet(
        net_depth=args.net_depth,
        net_width=args.net_width,
        input_ch=total_input_ch,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        params=list(model.parameters()),
        lr=args.lrate,
        betas=(0.9, 0.999),
    )

    # Find all .tar files in log_dir.
    is_ckpt_loaded = False
    ckpt_paths = sorted(log_dir.glob("*.tar"))
    if len(ckpt_paths) > 0:
        print(f"Found checkpoints: {ckpt_paths}")
        if args.no_reload:
            print("Ignoring checkpoints, not reloading.")
            global_iter = 0
        else:
            ckpt_path = ckpt_paths[-1]
            print(f"Reloading from {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            global_iter = ckpt["global_iter"]
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            model.load_state_dict(ckpt["network_fn_state_dict"])
            is_ckpt_loaded = True
    else:
        global_iter = 0

    # Load data.
    train_inputs, train_outputs, val_inputs, val_outputs = load_raykeep_dataset(
        lit_paths.nuscenes.to_nuscenes_raykeep / "raykeep_data.npz",
        balance=True,
    )

    debug_with_few_samples = False
    if debug_with_few_samples:
        num_samples = 500000
        train_inputs = train_inputs[:num_samples]
        train_outputs = train_outputs[:num_samples]

    if args.eval_only:
        if not is_ckpt_loaded:
            raise ValueError("No checkpoint loaded, cannot evaluate.")
        model.eval()

        print("[Validation set eval]")
        with torch.no_grad():
            pred_outputs = run_network(
                torch.tensor(val_inputs).to(device),
                model,
                embed_fn,
                embed_dir_fn,
            )
        pd_raykeeps = pred_outputs.cpu().numpy().round().astype(np.int_).ravel()
        gt_raykeeps = val_outputs.astype(np.int_)
        eval_raykeep_metrics(
            pd_raykeeps=pd_raykeeps,
            gt_raykeeps=gt_raykeeps,
        )

        print("[Training set eval]")
        pd_raykeeps = evaluate_in_batches(
            train_inputs, model, embed_fn, embed_dir_fn, device
        )
        pd_raykeeps = pd_raykeeps.round().astype(np.int_).ravel()
        gt_raykeeps = train_outputs.astype(np.int_)
        eval_raykeep_metrics(
            pd_raykeeps=pd_raykeeps,
            gt_raykeeps=gt_raykeeps,
        )

        exit(0)

    # (N, 5) cat (N,) -> (N, 6)
    train_data = np.concatenate([train_inputs, train_outputs[:, None]], axis=1)

    # Prepare ray batch tensor if batching random rays
    batch_size = args.batch_size
    train_data = torch.tensor(train_data).to(device)
    total_iters = args.total_iters + 1

    print("Begin")
    loss_log = []
    i_batch = 0
    for i in range(global_iter + 1, total_iters):
        # Load batch data.
        batch = train_data[i_batch : i_batch + batch_size]
        inputs, gt_keeps = batch[:, :5], batch[:, 5]
        i_batch += batch_size

        # Reset and shuffle data.
        if i_batch >= len(train_data):
            rand_idx = torch.randperm(len(train_data))
            train_data = train_data[rand_idx]
            i_batch = 0

        # Core optimization loop.
        pd_keeps = run_network(inputs, model, embed_fn, embed_dir_fn)
        optimizer.zero_grad()
        loss = rgb_loss(pd_keeps, gt_keeps.unsqueeze(1))
        loss.backward()
        optimizer.step()

        # Update learning rate.
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_iter / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

        # Save checkpoint.
        if i % args.i_weights == 0:
            ckpt_path = log_dir / f"{i:06d}.tar"
            ckpt = {
                "global_iter": global_iter,
                "network_fn_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoints at {ckpt_path}")

        # Log training.
        loss_log.append(loss.item())
        if i % args.i_print == 0:
            loss_save = np.array(loss_log)
            plt.plot(loss_save)
            plt.savefig(log_dir / "loss_curve.png")
            plt.close()
            print(f"[TRAIN] Iter: {i}, Loss: {loss.item():.4f}")

        global_iter += 1

    # Save loss log.
    loss_log = np.array(loss_log)
    np.save(log_dir / "loss_log.npy", loss_log)


if __name__ == "__main__":
    main()
