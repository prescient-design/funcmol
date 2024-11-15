
import os
import torch

from funcmol.dataset.field_maker import FieldMaker
from funcmol.utils.utils_nf import load_network, infer_codes, normalize_code


def add_noise_to_code(codes: torch.Tensor, smooth_sigma: float = 0.1) -> torch.Tensor:
    """
    Adds Gaussian noise to the input codes.

    Args:
        codes (torch.Tensor): Input codes to which noise will be added.

    Returns:
        torch.Tensor: Codes with added noise.
        torch.Tensor: Noise added to the codes.
    """
    if smooth_sigma == 0.0:
        return codes
    noise = torch.empty(codes.shape, device=codes.device, dtype=codes.dtype).normal_(0, smooth_sigma)
    return codes + noise


def load_checkpoint_fm(
    model: torch.nn.Module,
    pretrained_path: str,
    optimizer: torch.optim.Optimizer = None,
    fabric = None,
):
    """
    Loads a checkpoint file and restores the model and optimizer states.

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        pretrained_path (str): The path to the directory containing the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the checkpoint into. Defaults to None.
        best_model (bool, optional): Whether to load the best model checkpoint or the regular checkpoint.
            Defaults to True.

    Returns:
        tuple: A tuple containing the loaded model, optimizer (if provided), and the number of epochs trained.
    """
    checkpoint = fabric.load(os.path.join(pretrained_path, "checkpoint.pth.tar"))
    if optimizer is not None:
        load_network(checkpoint, model, fabric, is_compile=True, sd="state_dict_ema", net_name="denoiser")
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer, checkpoint["code_stats"]
    else:
        load_network(checkpoint, model, fabric, is_compile=True, sd="state_dict_ema", net_name="denoiser")
        fabric.print(f">> loaded model trained for {checkpoint['epoch']} epochs")
        return model, checkpoint["code_stats"]


def log_metrics(
    exp_name: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_res: float,
    time: float,
    fabric: object
):
    """
    Logs the metrics of a training experiment.

    Args:
        exp_name (str): The name of the experiment.
        epoch (int): The current epoch number.
        train_loss (float): The training loss value.
        val_loss (float): The validation loss value.
        best_res (float): The best result achieved so far.
        time (float): The time taken for the epoch.
        fabric (object): An object with a print method to output the log.

    Returns:
    None
    """
    str_ = f">> {exp_name} epoch: {epoch} ({time:.2f}s)\n"
    str_ += f"[train_loss] {train_loss:.2f} |"
    if val_loss is not None and best_res is not None:
        str_ += f" | [val_loss] {val_loss:.2f} (best: {best_res:.2f})"
    fabric.print(str_)

def compute_codes(
    loader_field: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    config_nf: dict,
    split: str,
    fabric: object,
    normalize_codes: bool,
    field_maker: FieldMaker=None,
    code_stats: dict=None,
) -> tuple:
    """
    Computes the codes using the provided encoder and data loader.

    Args:
        loader_field (torch.utils.data.DataLoader): DataLoader for the field data.
        enc (torch.nn.Module): Encoder model to generate codes.
        config_nf (dict): Configuration dictionary for the neural field.
        split (str): Data split identifier (e.g., 'train', 'val', 'test').
        fabric (object): Fabric object for distributed training.
        normalize_codes (bool): Whether to normalize the codes.
        field_maker (FieldMaker, optional): Optional FieldMaker object. Defaults to None.
        code_stats (dict, optional): Optional dictionary to store code statistics. Defaults to None.
    Returns:
        tuple: A tuple containing the generated codes and the code statistics.
    """
    codes = infer_codes(
        loader_field,
        enc,
        config_nf,
        fabric=fabric,
        to_cpu=True,
        field_maker=field_maker,
        code_stats=code_stats,
        n_samples=100_000,
    )
    if code_stats is None:
        code_stats = process_codes(codes, fabric, split, normalize_codes)
    else:
        get_stats(codes, fabric=fabric, message=f"====normalized codes {split}====")
    return codes, code_stats


def compute_code_stats_offline(
    loader: torch.utils.data.DataLoader,
    split: str,
    fabric: object,
    normalize_codes: bool
) -> dict:
    """
    Computes statistics for codes offline.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        split (str): The data split (e.g., 'train', 'val', 'test').
        fabric (object): An object representing the fabric used for processing.
        normalize_codes (bool): Whether to normalize the codes.

    Returns:
        dict: A dictionary containing the computed code statistics.
    """
    codes = loader.dataset.curr_codes[:]
    code_stats = process_codes(codes, fabric, split, normalize_codes)
    return code_stats


def process_codes(
    codes: torch.Tensor,
    fabric: object,
    split: str,
    normalize_codes: bool,
) -> dict:
    """
    Process the codes from the checkpoint.

    Args:
        checkpoint (dict): The checkpoint containing the codes.
        logger (object): The logger object for logging messages.
        device (torch.device): The device to use for processing the codes.
        is_filter (bool, optional): Whether to filter the codes. Defaults to False.

    Returns:
        tuple: A tuple containing the processed codes, statistics, and normalized codes.
    """
    max, min, mean, std = get_stats(
        codes,
        fabric=fabric,
        message=f"====codes {split}====",
    )
    code_stats = {
        "mean": mean,
        "std": std,
    }
    if normalize_codes:
        codes = normalize_code(codes, code_stats)
        max_normalized, min_normalized, _, _ = get_stats(
            codes,
            fabric=fabric,
            message=f"====normalized codes {split}====",
        )
    else:
        max_normalized, min_normalized = max, min
    code_stats.update({
        "max_normalized": max_normalized.item(),
        "min_normalized": min_normalized.item(),
    })
    return code_stats


def get_stats(
    codes: torch.Tensor,
    fabric: object = None,
    message: str = None,
):
    """
    Calculate statistics of the input codes.

    Args:
        codes_init (torch.Tensor): The input codes.
        fabric (object, optional): The logger object for logging messages. Defaults to None.
        message (str, optional): Additional message to log. Defaults to None.

    Returns:
        tuple: A tuple containing the calculated statistics:
            - max (torch.Tensor): The maximum values.
            - min (torch.Tensor): The minimum values.
            - mean (torch.Tensor): The mean values.
            - std (torch.Tensor): The standard deviation values.
            - (optional) codes (torch.Tensor): The filtered codes if `is_filter` is True.
    """
    if message is not None:
        fabric.print(message)
    max = codes.max()
    min = codes.min()
    mean = codes.mean()
    std = codes.std()
    fabric.print(f"min: {min.item()}")
    fabric.print(f"max: {max.item()}")
    fabric.print(f"mean: {mean.item()}")
    fabric.print(f"std: {std.item()}")
    fabric.print(f"codes size: {codes.shape}")
    return max, min, mean, std
