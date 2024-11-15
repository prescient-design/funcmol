import os
import time
from funcmol.dataset.field_maker import FieldMaker
import hydra
from omegaconf import OmegaConf
import torch
import torchmetrics

from funcmol.models.funcmol import create_funcmol
from funcmol.utils.utils_fm import (
    add_noise_to_code, load_checkpoint_fm, log_metrics,
    compute_code_stats_offline, compute_codes
)
from funcmol.models.adamw import AdamW
from funcmol.models.ema import ModelEma
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import infer_codes_occs_batch, load_neural_field, normalize_code
from funcmol.dataset.dataset_code import create_code_loaders
from funcmol.dataset.dataset_field import create_field_loaders


@hydra.main(config_path="configs", config_name="train_fm_qm9", version_base=None)
def main(config):
    fabric = setup_fabric(config)

    # field_maker
    field_maker = FieldMaker(config, sample_points=False)
    field_maker = field_maker.to(fabric.device)

    exp_name, dirname = config["exp_name"], config["dirname"]
    if not config["on_the_fly"]:
        codes_dir = config["codes_dir"]
    config = OmegaConf.to_container(config)
    config["exp_name"], config["dirname"] = exp_name, dirname  # TODO: make this better
    if not config["on_the_fly"]:
        config["codes_dir"] = codes_dir
    fabric.print(">> saving experiments in:", config["dirname"])

    ##############################
    # load pretrained neural field
    nf_checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
    enc, dec = load_neural_field(nf_checkpoint, fabric)
    dec_module = dec.module if hasattr(dec, "module") else dec

    ##############################
    # code loaders
    config_nf = nf_checkpoint["config"]
    config_nf["debug"] = config["debug"]
    config_nf["dset"]["batch_size"] = config["dset"]["batch_size"]

    if config["on_the_fly"]:
        loader_train = create_field_loaders(config, split="train", fabric=fabric)
        loader_val = create_field_loaders(config, split="val", fabric=fabric) if fabric.global_rank == 0 else None
        _, code_stats = compute_codes(
            loader_train, enc, config_nf, "train", fabric, config["normalize_codes"],
            field_maker=field_maker, code_stats=None
        )
    else:
        loader_train = create_code_loaders(config, split="train", fabric=fabric)
        loader_val = create_code_loaders(config, split="val", fabric=fabric) if fabric.global_rank == 0 else None
        code_stats = compute_code_stats_offline(loader_train, "train", fabric, config["normalize_codes"])
    dec_module.set_code_stats(code_stats)

    config["num_iterations"] = config["num_epochs"] * len(loader_train)

    ##############################
    # create Funcmol, optimizer, criterion, EMA, metrics
    funcmol = create_funcmol(config, fabric)
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = AdamW(funcmol.parameters(), lr=config["lr"], weight_decay=config["wd"])
    optimizer.zero_grad()

    with torch.no_grad():
        funcmol_ema = ModelEma(funcmol, decay=config["ema_decay"])
        funcmol_ema = torch.compile(funcmol_ema)

    if config["reload_model_path"] is not None:
        fabric.print(f">> loading checkpoint from {config['reload_model_path']}")
        funcmol, optimizer, code_stats = load_checkpoint_fm(funcmol, config["reload_model_path"], optimizer, fabric=fabric)
        dec_module.set_code_stats(code_stats)
        with torch.no_grad():
            if config["reload_model_path"] is not None:
                funcmol_ema, _ = load_checkpoint_fm(funcmol_ema, config["reload_model_path"], fabric=fabric)
    funcmol, optimizer = fabric.setup(funcmol, optimizer)

    ##############################
    # metrics
    metrics = torchmetrics.MeanMetric().to(fabric.device)
    metrics_val = torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device)

    ##############################
    # start training
    fabric.print(">> start training the denoiser", config["exp_name"])
    best_res = 1e10
    acc_iter = 0

    for epoch in range(0, config["num_epochs"]):
        t0 = time.time()

        # train
        train_loss, acc_iter = train_denoiser(
            loader_train,
            enc,
            dec_module,
            funcmol,
            criterion,
            optimizer,
            metrics,
            config,
            funcmol_ema,
            acc_iter,
            fabric,
            field_maker=field_maker
        )

        # eval
        val_loss = None
        if (epoch + 1) % 5 == 0:
            with fabric.rank_zero_first():
                if fabric.global_rank == 0:
                    val_loss = val_denoiser(
                        loader_val,
                        enc,
                        dec_module,
                        funcmol_ema,
                        criterion,
                        metrics_val,
                        config,
                        field_maker=field_maker
                    )
                    if val_loss < best_res:
                        best_res = val_loss

        # sample and save
        if ((epoch + 1) % config["sample_every"] == 0 or epoch == config["num_epochs"] - 1) and epoch != 0:
            with fabric.rank_zero_first():
                if fabric.global_rank == 0:
                    try:
                        sample(funcmol_ema, dec_module, config, fabric)
                    except Exception as e:
                        fabric.print(f"Error during sampling: {e}")
            # save
            state = {
                "epoch": epoch + 1,
                "config": config,
                "state_dict_ema": funcmol_ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "code_stats": dec_module.code_stats
            }
            fabric.save(os.path.join(config["dirname"], "checkpoint.pth.tar"), state)


        # log metrics
        log_metrics(
            config["exp_name"],
            epoch,
            train_loss,
            val_loss,
            best_res,
            time.time() - t0,
            fabric,
        )

        if config["wandb"]:
            fabric.log_dict({
                "trainer/global_step": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })


def train_denoiser(
    loader: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    dec_module: torch.nn.Module,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: torchmetrics.MeanMetric,
    config: dict,
    model_ema: ModelEma=None,
    acc_iter: int = 0,
    fabric: object = None,
    field_maker: FieldMaker=None
) -> tuple:
    """
    Train a denoising model using the provided data loader, model, and training configuration.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the training data.
        enc (torch.nn.Module): Encoder module.
        dec_module (torch.nn.Module): Decoder module.
        model (torch.nn.Module): The denoising model to be trained.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        metrics (torchmetrics.MeanMetric): Metric to track the training performance.
        config (dict): Configuration dictionary containing training parameters.
        model_ema (ModelEma, optional): Exponential moving average model. Defaults to None.
        acc_iter (int, optional): Accumulated iteration count. Defaults to 0.
        fabric (object, optional): Fabric object for distributed training. Defaults to None.
        field_maker (FieldMaker, optional): FieldMaker object for generating fields. Defaults to None.

    Returns:
        tuple: A tuple containing the computed metric value and the updated accumulated iteration count.
    """
    metrics.reset()
    model.train()

    if not config["on_the_fly"]:  # and fabric.global_rank == 0:
        index = torch.randint(0, loader.dataset.num_augmentations, [1])[0].item()  # random.randint(0, self.num_augmentations)
        loader.dataset.load_codes(index)

    for batch in loader:
        adjust_learning_rate(optimizer, acc_iter, config)
        acc_iter += 1

        if config["on_the_fly"]:
            with torch.no_grad():
                codes, _ = infer_codes_occs_batch(
                    batch, enc, config, to_cpu=False, field_maker=field_maker,
                    code_stats=dec_module.code_stats if config["normalize_codes"] else None
                )
        else:
            codes = normalize_code(batch, dec_module.code_stats)

        smooth_codes = add_noise_to_code(codes, smooth_sigma=config["smooth_sigma"])
        loss = compute_loss(codes, smooth_codes, model, criterion)

        optimizer.zero_grad()
        fabric.backward(loss)
        optimizer.step()

        model_ema.update(model)
        metrics.update(loss)

    return metrics.compute().item(), acc_iter


@torch.no_grad()
def val_denoiser(
    loader: torch.utils.data.DataLoader,
    enc: torch.nn.Module,
    dec_module: torch.nn.Module,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    metrics: torchmetrics.MeanMetric,
    config: dict,
    field_maker: FieldMaker=None
) -> float:
    """
    Validate the denoising model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        enc (torch.nn.Module): Encoder module.
        dec_module (torch.nn.Module): Decoder module.
        model (torch.nn.Module): Denoising model.
        criterion (torch.nn.Module): Loss function.
        metrics (torchmetrics.MeanMetric): Metric to compute the mean loss.
        config (dict): Configuration dictionary containing various settings.
        field_maker (FieldMaker, optional): Optional FieldMaker instance for on-the-fly code generation.

    Returns:
        float: Computed mean loss over the validation dataset.
    """
    enc = enc.module
    model = model.module
    model.eval()
    metrics.reset()
    for batch in loader:
        if config["on_the_fly"]:
            codes, _ = infer_codes_occs_batch(
                batch, enc, config, to_cpu=False, field_maker=field_maker,
                code_stats=dec_module.code_stats if config["normalize_codes"] else None
            )
        else:
            codes = normalize_code(batch, dec_module.code_stats)
        smooth_codes = add_noise_to_code(codes, smooth_sigma=config["smooth_sigma"])
        loss = compute_loss(codes, smooth_codes, model, criterion)
        metrics.update(loss)
    return metrics.compute().item()


def sample(
    model: torch.nn.Module,
    dec_module: torch.nn.Module,
    config: dict,
    fabric: object,
) -> None:
    """
    Samples from the given model and saves the generated samples to the specified directory.

    Args:
        model (torch.nn.Module): The model to sample from.
        dec_module (torch.nn.Module): The decoder module used during sampling.
        config (dict): Configuration dictionary containing parameters such as 'ema_decay' and 'dirname'.
        fabric (object): An object providing utility functions such as 'print'.

    Returns:
        None
    """
    if config["ema_decay"] > 0:
        model = model.module
    model.eval()

    dirname_out = os.path.join(config["dirname"], "samples")
    os.makedirs(dirname_out, exist_ok=True)

    t0 = time.time()
    with torch.no_grad():
        _ = model.sample(dec=dec_module, save_dir=dirname_out, config=config, fabric=fabric)
    tf = time.time()
    fabric.print(f">> done sampling (time ellapsed: {(tf - t0):.2f}s")


def compute_loss(
    codes: torch.Tensor,
    smooth_codes: torch.Tensor,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    """
    Computes the loss between the predicted outputs of the model and the target codes.
    Args:
        codes (torch.Tensor): The target tensor containing the true codes.
        smooth_codes (torch.Tensor): The input tensor to be fed into the model.
        model (torch.nn.Module): The neural network model used for prediction.
        criterion (torch.nn.Module): The loss function used to compute the loss.
    Returns:
        torch.Tensor: The computed loss value.
    """
    return criterion(model(smooth_codes), codes)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, iteration: int, config: dict) -> float:
    """
    Adjusts the learning rate based on the current iteration and configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer object.
        iteration (int): The current iteration.
        config (dict): The configuration dictionary containing the learning rate and other parameters.

    Returns:
        float: The adjusted learning rate.
    """
    if config["use_lr_schedule"] <= 0:
        return config["lr"]
    else:
        if iteration < config["num_warmup_iter"]:
            lr_ratio = config["lr"] * float((iteration + 1) / config["num_warmup_iter"])
        else:
            # decay proportionally with the square root of the iteration
            lr_ratio = 1 - (iteration - config["num_warmup_iter"] + 1) / (config["num_iterations"] - config["num_warmup_iter"] + 1)
            lr_ratio = lr_ratio ** .5
        lr = config["lr"] * lr_ratio
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr


if __name__ == "__main__":
    main()
