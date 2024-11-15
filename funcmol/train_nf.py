import time
import hydra
import os
import torch
import torchmetrics
from torch import nn
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import (
    create_neural_field, train_nf, eval_nf, save_checkpoint, load_network, load_optim_fabric
)
from funcmol.dataset.dataset_field import create_field_loaders
from funcmol.dataset.field_maker import FieldMaker


@hydra.main(config_path="configs", config_name="train_nf_drugs", version_base=None)
def main(config):
    fabric = setup_fabric(config)

    field_maker = FieldMaker(config)
    field_maker = field_maker.to(fabric.device)

    ##############################
    # data loaders
    loader_train = create_field_loaders(config, split="train", fabric=fabric)
    loader_val = create_field_loaders(config, split="val", fabric=fabric) if fabric.global_rank == 0 else None

    # model
    enc, dec = create_neural_field(config, fabric)
    criterion = nn.MSELoss()

    # optimizers
    optim_enc = torch.optim.Adam([{"params": enc.parameters(), "lr": config["dset"]["lr_enc"]}])
    optim_dec = torch.optim.Adam([{"params": dec.parameters(), "lr": config["dset"]["lr_dec"]}])

    # fabric setup
    dec, optim_dec = fabric.setup(dec, optim_dec)
    enc, optim_enc = fabric.setup(enc, optim_enc)

    # reload
    if config["reload_model_path"] is not None:
        try:
            checkpoint = fabric.load(os.path.join(config["reload_model_path"], "model.pt"))
            fabric.print(f">> loaded checkpoint: {config['reload_model_path']}")

            dec = load_network(checkpoint, dec, fabric, net_name="dec")
            optim_dec = load_optim_fabric(optim_dec, checkpoint, config, fabric, net_name="dec")

            enc = load_network(checkpoint, enc, fabric, net_name="enc")
            optim_enc = load_optim_fabric(optim_enc, checkpoint, config, fabric, net_name="enc")
        except Exception as e:
            fabric.print(f"Error loading checkpoint: {e}")

    # Metrics
    metrics = {
        "loss": torchmetrics.MeanMetric().to(fabric.device),
        "miou": torchmetrics.classification.BinaryJaccardIndex().to(fabric.device),
    }
    metrics_val = {
        "loss": torchmetrics.MeanMetric(sync_on_compute=False).to(fabric.device),
        "miou": torchmetrics.classification.BinaryJaccardIndex(sync_on_compute=False).to(fabric.device),
    }

    ##############################
    # start training the neural field
    fabric.print(">> start training the neural field", config["exp_name"])
    best_loss = None  # save checkpoint each time save_checkpoint is called

    for epoch in range(config["n_epochs"]):
        start_time = time.time()

        adjust_learning_rate(optim_enc, optim_dec, epoch, config)

        # train
        loss_train, miou_train = train_nf(
            config,
            loader_train,
            dec,
            optim_dec,
            enc,
            optim_enc,
            criterion,
            fabric,
            metrics=metrics,
            field_maker=field_maker
        )

        # val
        loss_val, miou_val = None, None
        if (epoch % config["eval_every"] == 0 or epoch == config["n_epochs"] - 1) and epoch > 0:
            # Master rank performs evaluation and checkpointing
            if fabric.global_rank == 0:
                loss_val, miou_val = eval_nf(
                    loader_val,
                    dec,
                    enc,
                    criterion,
                    config,
                    metrics=metrics_val,
                    fabric=fabric,
                    field_maker=field_maker
                )
                save_checkpoint(
                    epoch, config, loss_val, best_loss, enc, dec, optim_enc, optim_dec, fabric)
            else:
                fabric.barrier()
        # log
        elapsed_time = time.time() - start_time
        log_epoch(config, epoch, loss_train, miou_train, loss_val, miou_val, elapsed_time, fabric)


def adjust_learning_rate(optim_enc, optim_dec, epoch, config):
    """
    Adjusts the learning rate for the encoder and decoder optimizers based on the current epoch and configuration.

    Parameters:
    optim_enc (torch.optim.Optimizer): The optimizer for the encoder.
    optim_dec (torch.optim.Optimizer): The optimizer for the decoder.
    epoch (int): The current epoch number.
    config (dict): Configuration dictionary containing learning rate settings and decay milestones.

    Returns:
    None
    """
    # Handcoded by now. Can be improved.
    if "lr_decay" not in config or config["lr_decay"] is None or not config["lr_decay"]:
        return
    lr_enc = config["dset"]["lr_enc"]
    lr_dec = config["dset"]["lr_dec"]
    for milestone in [80]:
        lr_enc *= 0.1 if epoch >= milestone else 1.0
        lr_dec *= 0.1 if epoch >= milestone else 1.0

    for param_group in optim_enc.param_groups:
        param_group["lr"] = lr_enc
    for param_group in optim_dec.param_groups:
        param_group["lr"] = lr_dec
    print("epoch", epoch, "lr_enc", lr_enc, "lr_dec", lr_dec)


def log_epoch(
    config: dict,
    epoch: int,
    loss_train: float,
    miou_train: float,
    loss_val: float,
    miou_val: float,
    elapsed_time: float,
    fabric: object
    ) -> None:
    """
    Logs the training and validation metrics for a given epoch.

    Args:
        config (dict): Configuration dictionary containing dataset and experiment details.
        epoch (int): The current epoch number.
        loss_train (float): Training loss for the current epoch.
        miou_train (float): Training mean Intersection over Union (mIoU) for the current epoch.
        loss_val (float): Validation loss for the current epoch.
        miou_val (float): Validation mean Intersection over Union (mIoU) for the current epoch.
        elapsed_time (float): Time elapsed since the start of training in seconds.
        fabric (object): An object for logging metrics, such as a Weights and Biases (wandb) logger.

    Returns:
        None
    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    log = f"| {config['exp_name']} [{epoch}/{config['n_epochs']}]" +\
        f" | train_loss: {loss_train:0.3e} | train_miou: {miou_train:0.3e}"
    if loss_val is not None:
        log += f" | val_loss: {loss_val:0.3e} | val_miou: {miou_val:0.3e}"
    log += f" | {int(hours):0>2}h:{int(minutes):0>2}m:{seconds:05.2f}s"

    if config["wandb"]:
        fabric.log_dict({
            "trainer/global_step": epoch,
            "train_loss": loss_train,
            "train_miou": miou_train,
            "val_loss": loss_val,
            "val_miou": miou_val
        })
    fabric.print(log)


if __name__ == "__main__":
    main()
