from tqdm import tqdm
import time
import hydra
import os
from funcmol.utils.utils_nf import load_neural_field
from funcmol.utils.utils_base import setup_fabric
from funcmol.dataset.field_maker import FieldMaker
from funcmol.dataset.dataset_field import create_field_loaders
import omegaconf
import torch


@hydra.main(config_path="configs", config_name="infer_codes", version_base=None)
def main(config):
    # initial setup
    fabric = setup_fabric(config)

    # load checkpoint and update config
    checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
    config_model = checkpoint["config"]
    for key in config.keys():
        if key in config_model and \
            isinstance(config_model[key], omegaconf.dictconfig.DictConfig):
            config_model[key].update(config[key])
        else:
            config_model[key] = config[key]
    config = config_model  # update config with checkpoint config
    enc, _ = load_neural_field(checkpoint, fabric, config)

    # field_maker
    field_maker = FieldMaker(config, sample_points=False)
    field_maker = field_maker.to(fabric.device)

    # data loader
    loader = create_field_loaders(config, split=config["split"], fabric=fabric)

    # Print config
    fabric.print(f">> config: {config}")
    fabric.print(f">> seed: {config['seed']}")

    # create output directory
    fabric.print(">> saving codes in", config["dirname"])
    os.makedirs(config["dirname"], exist_ok=True)

    # count number of files on directory
    n_code_files = len([
        f for f in os.listdir(config["dirname"])
        if os.path.isfile(os.path.join(config["dirname"], f)) and \
        f.startswith("codes") and f.endswith(".pt")
    ])
    fabric.print(f">> found {n_code_files} code files")

    # start eval
    fabric.print(f">> start code inference in {config['split']} split")
    n_dset_iter = config["n_dataset_iterations"] if config["split"] == "train" else 1
    iter_counter = n_code_files
    enc.eval()

    with torch.no_grad():
        for dataset_iter in range(n_dset_iter):
            codes = []
            fabric.print(f">> dataset iteration {dataset_iter}")
            t0 = time.time()
            for batch in tqdm(loader):
                _, grids = field_maker.forward(batch)
                codes_batch = enc(grids)
                codes.append(codes_batch.detach().cpu())
            fabric.print(f">> saving chunk {iter_counter}")
            codes = torch.cat(codes, dim=0)
            torch.save(codes, os.path.join(config["dirname"], f"codes_{iter_counter:04d}.pt"))
            iter_counter += 1
            del codes

            elapsed_time = time.time() - t0
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            fabric.print(f">> time for iter {dataset_iter}: {int(hours):0>2}h:{int(minutes):0>2}m:{seconds:05.2f}s")


if __name__ == "__main__":
    main()