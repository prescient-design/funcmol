import os
import torch
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from torch import nn
import shutil

from funcmol.models.decoder import Decoder, _normalize_coords, get_atom_coords
from funcmol.models.encoder import Encoder
from funcmol.utils.utils_base import convert_xyzs_to_sdf, save_xyz
from funcmol.utils.utils_vis import visualize_voxel_grid


def create_neural_field(config: dict, fabric: object) -> tuple:
    """
    Creates and compiles the Encoder and Decoder neural network models based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the models.
            Expected keys:
                - "decoder": A dictionary with the following keys:
                    - "code_dim" (int): The dimension of the bottleneck code.
                    - "hidden_dim" (int): The hidden dimension of the decoder.
                    - "coord_dim" (int): The coordinate dimension for the decoder.
                    - "n_layers" (int): The number of layers in the decoder.
                    - "input_scale" (float): The input scale for the decoder.
                - "dset": A dictionary with the following keys:
                    - "n_channels" (int): The number of input channels.
                    - "grid_dim" (int): The grid dimension of the dataset.
                - "encoder": A dictionary with the following keys:
                    - "level_channels" (list of int): The number of channels at each level of the encoder.
                    - "smaller" (bool, optional): A flag indicating whether to use a smaller encoder. Defaults to False.
        fabric (object): An object that provides utility functions such as printing and model compilation.

    Returns:
        tuple: A tuple containing the compiled Encoder and Decoder models.
    """
    enc = Encoder(
        bottleneck_channel=config["decoder"]["code_dim"],
        in_channels=config["dset"]["n_channels"],
        level_channels=config["encoder"]["level_channels"],
        smaller=config["encoder"]["smaller"] if "smaller" in config["encoder"] else False,
    )
    n_params_enc = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    fabric.print(f">> enc has {(n_params_enc/1e6):.02f}M parameters")

    dec = Decoder(
        n_channels=config["dset"]["n_channels"],
        grid_dim=config["dset"]["grid_dim"],
        hidden_dim=config["decoder"]["hidden_dim"],
        code_dim=config["decoder"]["code_dim"],
        coord_dim=config["decoder"]["coord_dim"],
        n_layers=config["decoder"]["n_layers"],
        input_scale=config["decoder"]["input_scale"],
        fabric=fabric
    )
    n_params_dec = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    fabric.print(f">> dec has {(n_params_dec/1e6):.02f}M parameters")

    fabric.print(">> compiling models...")
    dec = torch.compile(dec)
    enc = torch.compile(enc)
    fabric.print(">> models compiled")

    return enc, dec


def train_nf(
    config: dict,
    loader: torch.utils.data.DataLoader,
    dec: nn.Module,
    optim_dec: torch.optim.Optimizer,
    enc: nn.Module,
    optim_enc: torch.optim.Optimizer,
    criterion: nn.Module,
    fabric: object,
    metrics=None,
    field_maker=None,
) -> tuple:
    """
    Trains the neural field autoencoder model.

    Args:
        config (dict): Configuration dictionary containing training parameters.
        loader (torch.utils.data.DataLoader): DataLoader for the training data.
        dec (nn.Module): Decoder neural network module.
        optim_dec (torch.optim.Optimizer): Optimizer for the decoder.
        enc (nn.Module): Encoder neural network module.
        optim_enc (torch.optim.Optimizer): Optimizer for the encoder.
        criterion (nn.Module): Loss function.
        fabric (object): Fabric object for distributed training.
        metrics (optional): Metrics object for evaluating the model.
        field_maker (optional): field_maker object for processing input data.

    Returns:
        tuple: A tuple containing the average loss and computed metrics.
    """
    enc.train()
    dec.train()
    if metrics is not None:
        for key in metrics.keys():
            metrics[key].reset()

    for i, batch in enumerate(loader):
        # Forward pass through 3D CNN encoder
        codes, occs = infer_codes_occs_batch(batch, enc, config, to_cpu=False, field_maker=field_maker)

        # Forward pass through INR decoder
        pred = dec(batch["xs"], codes)

        # optimize best loss
        loss = criterion(pred, occs)
        metrics["miou"].update((pred > 0.5).to(torch.uint8), (occs > 0.5).to(torch.uint8))
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        fabric.backward(loss)
        optim_enc.step()
        optim_dec.step()

        metrics["loss"].update(loss)

    return metrics["loss"].compute().item(), metrics["miou"].compute().item()


@torch.no_grad()
def eval_nf(
    loader: torch.utils.data.DataLoader,
    dec: nn.Module,
    enc: nn.Module,
    criterion: nn.Module,
    config: dict,
    metrics=None,
    save_plot_png=False,
    fabric=None,
    field_maker=None,
    sample_full_grid=False,
) -> tuple:
    """
    Evaluate a neural field model using the provided data loader, encoder, decoder, and criterion.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        dec (nn.Module): Decoder model (INR decoder).
        enc (nn.Module): Encoder model (3D CNN encoder).
        criterion (nn.Module): Loss function to compute the evaluation loss.
        config (dict): Configuration dictionary containing various settings.
        metrics (dict, optional): Dictionary of metrics to update during evaluation. Defaults to None.
        save_plot_png (bool, optional): Whether to save evaluation plots as PNG files. Defaults to False.
        fabric (optional): Fabric object for saving plots. Defaults to None.
        field_maker (optional): Field maker object for generating fields. Defaults to None.
        sample_full_grid (bool, optional): Whether to sample the full grid. Defaults to False.

    Returns:
        tuple: A tuple containing the computed loss and mean intersection over union (mIoU) values.
    """
    dec = dec.module
    enc = enc.module
    dec.eval()
    enc.eval()
    for key in metrics.keys():
        metrics[key].reset()
    mols_gt = []
    mols_pred = []
    mols_pred_dict = defaultdict(list)
    codes_dict = defaultdict(list)
    for i, batch in enumerate(loader):
        # Forward pass through 3D CNN encoder
        codes, occs_gt = infer_codes_occs_batch(batch, enc, config, to_cpu=False, field_maker=field_maker)

        # Forward pass through INR decoder
        xs = batch["xs"]
        if sample_full_grid:
            xs = xs[0].unsqueeze(0)
            occs_pred = dec(xs, codes)
        else:
            occs_pred = dec(xs, codes)

        # Compute loss and update metrics
        loss = criterion(occs_pred, occs_gt)
        metrics["miou"].update((occs_pred > 0.5).to(torch.uint8), (occs_gt > 0.5).to(torch.uint8))
        metrics["loss"].update(loss)

        if save_plot_png:
            save_voxel_eval_nf(config, fabric, occs_gt=occs_gt, mols_gt=mols_gt, occs_pred=occs_pred, refine=True, i=i, mols_pred=mols_pred, mols_pred_dict=mols_pred_dict, codes=codes, codes_dict=codes_dict)
    if save_plot_png:
        save_sdf_eval_nf(config, fabric, mols_pred=mols_pred, mols_gt=mols_gt, dec=dec, mols_pred_dict=mols_pred_dict, codes_dict=codes_dict, refine=True)

    return metrics["loss"].compute().item(), metrics["miou"].compute().item()


def save_voxel_eval_nf(config, fabric, occs_pred, occs_gt=None, refine=True, i=0, mols_pred=[], mols_gt=[], mols_pred_dict=defaultdict(list), codes=None, codes_dict=defaultdict(list)):
    dirname_voxels = os.path.join(config["dirname"], "res")
    if os.path.exists(dirname_voxels):
        shutil.rmtree(dirname_voxels)
    os.makedirs(dirname_voxels, exist_ok=False)
    fabric.print(f">> saving images in {dirname_voxels}")

    occs_pred = occs_pred.permute(0, 2, 1).reshape(-1, occs_gt.size(2), config["dset"]["grid_dim"], config["dset"]["grid_dim"], config["dset"]["grid_dim"])
    if occs_gt is not None:
        occs_gt = occs_gt.permute(0, 2, 1).reshape(-1, occs_gt.size(2), config["dset"]["grid_dim"], config["dset"]["grid_dim"], config["dset"]["grid_dim"])

    for b in range(occs_pred.size(0)):
        # Predictions
        visualize_voxel_grid(occs_pred[b], os.path.join(dirname_voxels, f"./iter{i}_batch{b}_pred.png"), threshold=0.2, sparse=False)
        mol_init_pred = get_atom_coords(occs_pred[b].cpu(), rad=config["dset"]["atomic_radius"])
        if not refine:
            mol_init_pred["coords"] *= config["dset"]["resolution"]
        if mol_init_pred is not None:
            fabric.print("pred", mol_init_pred["coords"].size())
            if refine:
                mol_init_pred = _normalize_coords(mol_init_pred, config["dset"]["grid_dim"])
                num_coords = int(mol_init_pred["coords"].size(1))
                mols_pred_dict[num_coords].append(mol_init_pred)
                codes_dict[num_coords].append(codes[b])
            else:
                mols_pred.append(mol_init_pred)

        # Ground truth
        if occs_gt is not None:
            visualize_voxel_grid(occs_gt[b], os.path.join(dirname_voxels, f"./iter{i}_batch{b}_gt.png"), threshold=0.2, sparse=False)
            mol_init_gt = get_atom_coords(occs_gt[b].cpu(), rad=config["dset"]["atomic_radius"])
            mol_init_gt["coords"] *= config["dset"]["resolution"]
            if mol_init_gt is not None:
                fabric.print("gt", mol_init_gt["coords"].size())
                mols_gt.append(mol_init_gt)


def save_sdf_eval_nf(config, fabric, mols_pred, mols_gt=None, dec=None, mols_pred_dict=None, codes_dict=None, refine=True):
    dirname_voxels = os.path.join(config["dirname"], "res")

    # prediction
    if refine:
        mols_pred = dec._refine_coords(
            grouped_mol_inits=mols_pred_dict,
            grouped_codes=codes_dict,
            maxiter=200,
            grid_dim=config["dset"]["grid_dim"],
            resolution=config["dset"]["resolution"],
            fabric=fabric,
        )
    save_xyz(mols_pred, dirname_voxels, fabric, atom_elements=config["dset"]["elements"])
    convert_xyzs_to_sdf(dirname_voxels, fabric=fabric, delete=True, fname=f"molecules_obabel_pred_refine_{refine}.sdf")

    # ground truth
    if mols_gt is not None:
        save_xyz(mols_gt, dirname_voxels, fabric, atom_elements=config["dset"]["elements"])
        convert_xyzs_to_sdf(dirname_voxels, fabric=fabric, delete=True, fname="molecules_obabel_gt_refine_False.sdf")


def infer_codes(
    loader: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    config: dict,
    fabric = None,
    to_cpu: bool = False,
    field_maker = None,
    code_stats=None,
    n_samples=None,
) -> torch.Tensor:
    """
    Infers codes from a given data loader using a specified encoder model.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader providing the batches of data.
        enc (torch.nn.Module): Encoder model used to infer codes.
        config (dict): Configuration dictionary containing model and dataset parameters.
        fabric: Optional fabric object for distributed training (default: None).
        to_cpu (bool): Flag indicating whether to move the inferred codes to CPU (default: False).
        field_maker: Optional field maker object for additional processing (default: None).
        code_stats: Optional object to collect code statistics (default: None).
        n_samples (int, optional): Number of samples to infer codes for. If None, infer codes for all samples (default: None).

    Returns:
        torch.Tensor: Tensor containing the inferred codes.
    """
    enc.eval()
    codes_all = []
    if fabric is not None:
        fabric.print(">> Inferring codes - batches of", config["dset"]["batch_size"])
    len_codes = 0
    for i, batch in tqdm(enumerate(loader)):
        with torch.no_grad():
            codes, _ = infer_codes_occs_batch(batch, enc, config, to_cpu, field_maker=field_maker, code_stats=code_stats)
        codes = fabric.all_gather(codes).view(-1, config["decoder"]["code_dim"])
        len_codes += codes.size(0)
        codes_all.append(codes)
        if n_samples is not None and len_codes >= n_samples:
            break
    return torch.cat(codes_all, dim=0)


def infer_codes_occs_batch(batch, enc, config, to_cpu=False, field_maker=None, code_stats=None):
    """
    Infer codes and occurrences for a batch of data.

    Args:
        batch (Tensor): The input batch of data.
        enc (Callable): The encoder function to generate codes from voxels.
        config (dict): Configuration dictionary.
        to_cpu (bool, optional): If True, move the codes to CPU. Defaults to False.
        field_maker (Callable, optional): A function to generate occurrences and voxels from the batch. Defaults to None.
        code_stats (dict, optional): Statistics for normalizing the codes. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the inferred codes and occurrences.
    """
    occs, voxels = field_maker.forward(batch)
    codes = enc(voxels)
    if code_stats is not None:
        codes = normalize_code(codes, code_stats)
    if to_cpu:
        codes = codes.cpu()
    return codes, occs


def set_requires_grad(module: nn.Module, tf: bool = False) -> None:
    """
    Set the requires_grad attribute of a module and its parameters.

    Args:
        module (nn.Module): The module for which requires_grad attribute needs to be set.
        tf (bool, optional): The value to set for requires_grad attribute. Defaults to False.
    """
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf


def load_network(
    checkpoint: dict,
    net: nn.Module,
    fabric: object,
    net_name: str = "dec",
    is_compile: bool = True,
    sd: str = None,
) -> nn.Module:
    """
    Load a neural network's state dictionary from a checkpoint and update the network's parameters.

    Args:
        checkpoint (dict): A dictionary containing the checkpoint data.
        net (nn.Module): The neural network model to load the state dictionary into.
        fabric (object): An object with a print method for logging.
        net_name (str, optional): The key name for the network's state dictionary in the checkpoint. Defaults to "dec".
        is_compile (bool, optional): A flag indicating whether the network is compiled. Defaults to True.
        sd (str, optional): A specific key for the state dictionary in the checkpoint. If None, defaults to using net_name.

    Returns:
        nn.Module: The neural network model with the loaded state dictionary.
    """
    net_dict = net.state_dict()
    weight_first_layer_before = next(iter(net_dict.values())).sum()
    new_state_dict = OrderedDict()
    key = f"{net_name}_state_dict" if sd is None else sd
    for k, v in checkpoint[key].items():
        if sd is not None:
            k = k[17:] if k[:17] == "_orig_mod.module." else k
        else:
            k = k[10:] if k[:10] == "_orig_mod." and not is_compile else k  # remove compile prefix.
        new_state_dict[k] = v

    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    weight_first_layer_after = next(iter(net_dict.values())).sum()
    assert weight_first_layer_before != weight_first_layer_after, "loading did not work"
    fabric.print(f">> loaded {net_name}")

    return net


def load_optim_fabric(
    optim: torch.optim.Optimizer,
    checkpoint: dict,
    config: dict,
    fabric: object,
    net_name: str = "dec"
) -> torch.optim.Optimizer:
    """
    Loads the optimizer state from a checkpoint and updates the learning rate.
    Args:
        optim (torch.optim.Optimizer): The optimizer to be loaded.
        checkpoint (dict): The checkpoint containing the optimizer state.
        config (dict): Configuration dictionary containing learning rate information.
        fabric (object): An object with a print method for logging.
        net_name (str, optional): The name of the network. Defaults to "dec".
    Returns:
        torch.optim.Optimizer: The optimizer with the loaded state and updated learning rate.
    """
    fabric.print(f"optim_{net_name} ckpt", checkpoint[f"optim_{net_name}"]["param_groups"])
    optim.load_state_dict(checkpoint[f"optim_{net_name}"])
    for g in optim.param_groups:
        g["lr"] = config["dset"][f"lr_{net_name}"]
    fabric.print(f"Loaded optim_{net_name}")
    return optim


def save_checkpoint(
    epoch: int,
    config: dict,
    loss_tot: float,
    loss_min_tot: float,
    enc: nn.Module,
    dec: nn.Module,
    optim_enc: torch.optim.Optimizer,
    optim_dec: torch.optim.Optimizer,
    fabric: object,
)-> float:
    """
    Saves a model checkpoint if the current total loss is less than the minimum total loss.

    Args:
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training parameters.
        loss_tot (float): The current total loss.
        loss_min_tot (float): The minimum total loss encountered so far.
        enc (nn.Module): The encoder model.
        dec (nn.Module): The decoder model.
        optim_enc (torch.optim.Optimizer): The optimizer for the encoder.
        optim_dec (torch.optim.Optimizer): The optimizer for the decoder.
        fabric (object): An object responsible for saving the model state.

    Returns:
        float: The updated minimum total loss.
    """

    if loss_min_tot is None or loss_tot < loss_min_tot:
        if loss_min_tot is not None:
            loss_min_tot = loss_tot
        try:
            state = {
                "epoch": epoch,
                "dec_state_dict": dec.state_dict(),
                "enc_state_dict": enc.state_dict(),
                "optim_dec": optim_dec.state_dict(),
                "optim_enc": optim_enc.state_dict(),
                "config": config,
            }
            fabric.save(os.path.join(config["dirname"], "model.pt"), state)
            fabric.print(">> saved checkpoint")
        except Exception as e:
            fabric.print(f"Error saving checkpoint: {e}")
    return loss_min_tot


def load_neural_field(nf_checkpoint: dict, fabric: object, config: dict = None) -> tuple:
    """
    Load and initialize the neural field encoder and decoder from a checkpoint.

    Args:
        nf_checkpoint (dict): The checkpoint containing the saved state of the neural field model.
        fabric (object): The fabric object used for setting up the modules.
        config (dict, optional): Configuration dictionary for initializing the encoder and decoder.
                                 If None, the configuration from the checkpoint will be used.

    Returns:
        tuple: A tuple containing the initialized encoder and decoder modules.
    """
    if config is None:
        config = nf_checkpoint["config"]

    dec = Decoder(
        n_channels=config["dset"]["n_channels"],
        grid_dim=config["dset"]["grid_dim"],
        hidden_dim=config["decoder"]["hidden_dim"],
        code_dim=config["decoder"]["code_dim"],
        coord_dim=config["decoder"]["coord_dim"],
        n_layers=config["decoder"]["n_layers"],
        input_scale=config["decoder"]["input_scale"],
        fabric=fabric
    )
    dec = load_network(nf_checkpoint, dec, fabric, net_name="dec")
    dec = torch.compile(dec)
    dec.eval()

    enc = Encoder(
        bottleneck_channel=config["decoder"]["code_dim"],
        in_channels=config["dset"]["n_channels"],
        level_channels=config["encoder"]["level_channels"],
        smaller=config["encoder"]["smaller"] if "smaller" in config["encoder"] else False,
    )
    enc = load_network(nf_checkpoint, enc, fabric, net_name="enc")
    enc = torch.compile(enc)
    enc.eval()

    dec = fabric.setup_module(dec)
    enc = fabric.setup_module(enc)

    return enc, dec


def normalize_code(codes: torch.Tensor, code_stats: dict) -> torch.Tensor:
    """
    Normalizes the input tensor of codes using the provided statistics.

    Args:
        codes (torch.Tensor): A tensor containing the codes to be normalized.
        code_stats (dict): A dictionary containing the mean and standard deviation
                           for normalization. It should have the keys "mean" and "std".

    Returns:
        torch.Tensor: The normalized tensor.
    """
    mean = code_stats["mean"]
    std = code_stats["std"]
    return (codes - mean) / std