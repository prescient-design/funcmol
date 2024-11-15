import shutil
import torchmetrics
import hydra
import os
import time
import omegaconf
import torch
from torch import nn

from funcmol.utils.utils_nf import eval_nf, infer_codes, load_neural_field
from funcmol.utils.utils_base import convert_xyzs_to_sdf, save_xyz, setup_fabric
from funcmol.dataset.dataset_field import create_field_loaders
from funcmol.dataset.field_maker import FieldMaker


@hydra.main(config_path="configs", config_name="eval_nf", version_base=None)
def main(config):
    # initial setup
    assert config["eval_metric"] in ["miou", "sampling"], "eval_metric must be either 'miou' or 'sampling'"
    fabric = setup_fabric(config)

    # load checkpoint and update config
    checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
    config_model = checkpoint["config"]
    for key in config.keys():
        if key in config_model and isinstance(config_model[key], omegaconf.dictconfig.DictConfig):
            config_model[key].update(config[key])
        else:
            config_model[key] = config[key]
    config = config_model  # update config with checkpoint config
    enc, dec = load_neural_field(checkpoint, fabric, config)
    dec_module = dec.module
    criterion = nn.MSELoss()

    # n samples
    n_samples = config["n_samples"]

    # field_maker
    field_maker = FieldMaker(config)
    field_maker = field_maker.to(fabric.device)

    # Print config
    fabric.print(f">> config: {config}")
    fabric.print(f">> seed: {config['seed']}")

    # metrics
    metrics = {
        "loss": torchmetrics.MeanMetric().to(fabric.device),
        "miou": torchmetrics.classification.BinaryJaccardIndex().to(fabric.device),
    }

    # create output directory
    fabric.print(">> saving in directory", config["dirname"])
    if os.path.exists(config["dirname"]):
        shutil.rmtree(config["dirname"])
    os.makedirs(config["dirname"], exist_ok=False)

    # start eval
    fabric.print(">> start eval")
    job_start_time = time.time()

    # miou
    if config["eval_metric"] == "miou":
        if config["dset"]["dset_name"] == "qm9":
            config["dset"]["batch_size"] = min(76, n_samples)
        elif config["dset"]["dset_name"] == "drugs":
            config["dset"]["batch_size"] = min(8, n_samples)
        else:
            config["dset"]["batch_size"] = min(2, n_samples)
        loader = create_field_loaders(config, split=config["split"], n_samples=n_samples, fabric=fabric, sample_full_grid=True)
        loss_val, miou_val = eval_nf(
            loader,
            dec,
            enc,
            criterion,
            config,
            metrics=metrics,
            save_plot_png=config["save_grids_png"],
            fabric=fabric,
            field_maker=field_maker,
            sample_full_grid=True
        )
        fabric.print(f">> split {config['split']}, Loss: {loss_val}, MIoU: {miou_val}")

    # Midi metrics
    elif config["eval_metric"] == "sampling":
        field_maker.set_sample_points(False) # do not sample points
        config["dset"]["batch_size"] = min(config["dset"]["batch_size"], n_samples)

        loader = create_field_loaders(config, split=config["split"], n_samples=n_samples, fabric=fabric)
        with torch.no_grad():
            codes_all = infer_codes(loader, enc, config, fabric, field_maker=field_maker)
            mols = dec_module.codes_to_molecules(codes_all, unnormalize=False, config=config, fabric=fabric)
        save_xyz(mols, config["dirname"], fabric, atom_elements=config["dset"]["elements"])
        convert_xyzs_to_sdf(config["dirname"], fabric=fabric)

    job_end_time = time.time()
    elapsed_time = job_end_time - job_start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    fabric.print(
        "====> {} runid {}, split {}, {:0>2}h:{:0>2}m:{:05.2f}s <====".format(
            config["dset"]["dset_name"],
            config["exp_name"],
            config["split"],
            int(hours),
            int(minutes),
            seconds,
        )
    )

if __name__ == "__main__":
    main()