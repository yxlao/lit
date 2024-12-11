import numpy as np
import sklearn
import torch


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embed_fn(multires, input_dims=3):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder = Embedder(**embed_kwargs)
    embed_fn = lambda x, eo=embedder: eo.embed(x)
    out_dim = embedder.out_dim

    return embed_fn, out_dim


def load_raykeep_dataset(file_path, split_ratio=0.8, balance=False):
    """
    Load and split the ray keep dataset with an option to balance the dataset.

    Args:
        file_path (str): Path to the dataset file.
        split_ratio (float): Ratio of the dataset to use for training.
        balance (bool): Whether to balance the dataset by undersampling the majority class.
    """
    # Save random state.
    np_random_state = np.random.get_state()
    np.random.seed(0)

    # Load dataset
    data = np.load(file_path)
    network_inputs = data["network_inputs"]
    network_outputs = data["network_outputs"]

    # Shuffle dataset
    shuffle_indices = np.random.permutation(len(network_inputs))
    network_inputs = network_inputs[shuffle_indices]
    network_outputs = network_outputs[shuffle_indices]

    # Assert: check that the network_outputs is either 0 or 1.
    assert np.isin(network_outputs, [0, 1]).all()

    # Split train/val
    split_index = int(len(network_inputs) * split_ratio)
    train_inputs = network_inputs[:split_index]
    train_outputs = network_outputs[:split_index]
    val_inputs = network_inputs[split_index:]
    val_outputs = network_outputs[split_index:]

    # Balance training dataset if required
    if balance:
        keep_indices = np.where(train_outputs == 1)[0]
        drop_indices = np.where(train_outputs == 0)[0]

        if len(keep_indices) < len(drop_indices):
            drop_indices = np.random.choice(
                drop_indices, len(keep_indices), replace=False
            )
        else:
            keep_indices = np.random.choice(
                keep_indices, len(drop_indices), replace=False
            )

        balanced_indices = np.concatenate([keep_indices, drop_indices])
        np.random.shuffle(balanced_indices)

        train_inputs = train_inputs[balanced_indices]
        train_outputs = train_outputs[balanced_indices]

    # Restore random state
    np.random.set_state(np_random_state)

    return train_inputs, train_outputs, val_inputs, val_outputs


def eval_raykeep_metrics(pd_raykeeps: np.ndarray, gt_raykeeps: np.ndarray):
    """
    Evaluate the performance of a ray keep prediction model.

    Args:
        pd_raykeeps (np.ndarray): Predicted ray keep values (probabilities
            rounded to 0 or 1).
        gt_raykeeps (np.ndarray): Ground truth ray keep values (0 or 1).

    Raises:
        ValueError: If input arrays are not 1D, have mismatched shapes, or
            contain incorrect data.
    """
    # Check if inputs are numpy arrays and are 1D
    if not isinstance(pd_raykeeps, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
    if not isinstance(gt_raykeeps, np.ndarray):
        raise ValueError("Inputs must be numpy arrays")
    if pd_raykeeps.ndim != 1 or gt_raykeeps.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")

    # Check if shapes of the input arrays match
    if pd_raykeeps.shape != gt_raykeeps.shape:
        raise ValueError("Shape of predicted and ground truth arrays must match")

    # Check if inputs are binary (0 or 1)
    if not (np.isin(pd_raykeeps, [0, 1]).all() and np.isin(gt_raykeeps, [0, 1]).all()):
        raise ValueError("Values in both arrays must be either 0 or 1")

    # Ensure inputs are integer type
    pd_raykeeps = pd_raykeeps.astype(np.int_)
    gt_raykeeps = gt_raykeeps.astype(np.int_)

    # Confusion Matrix and other metrics
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(gt_raykeeps, pd_raykeeps).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = sklearn.metrics.accuracy_score(gt_raykeeps, pd_raykeeps)
    gt_val_drop_ratio = 1 - np.mean(gt_raykeeps)
    pd_val_drop_ratio = 1 - np.mean(pd_raykeeps)

    # Print the evaluation metrics
    print(f"Precision      : {precision:.03f}")
    print(f"Recall         : {recall:.03f}")
    print(f"F1 score       : {f1_score:.03f}")
    print(f"Accuracy       : {accuracy:.03f}")
    print(f"Gt drop ratio  : {gt_val_drop_ratio:.03f}")
    print(f"Pred drop ratio: {pd_val_drop_ratio:.03f}")


def main():
    pass


if __name__ == "__main__":
    main()
