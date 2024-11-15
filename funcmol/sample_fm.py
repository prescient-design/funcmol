import os
from funcmol.models.funcmol import create_funcmol
import hydra
import torch
import omegaconf
from funcmol.train_fm import sample
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import load_neural_field
from funcmol.utils.utils_fm import load_checkpoint_fm


@hydra.main(config_path="configs", config_name="sample_fm", version_base=None)
def main(config):
    # Initial setup
    fabric = setup_fabric(config)

    checkpoint_fm = fabric.load(os.path.join(config["fm_pretrained_path"], "checkpoint.pth.tar"))
    config_ckpt = checkpoint_fm["config"]
    for key in config.keys():
        if key in config_ckpt and isinstance(config_ckpt[key], omegaconf.dictconfig.DictConfig):
            config_ckpt[key].update(config[key])
        else:
            config_ckpt[key] = config[key]
    config = config_ckpt
    fabric.print(f"updated config: {config}")

    # load checkpoint
    with torch.no_grad():
        funcmol = create_funcmol(config, fabric)
        funcmol, code_stats = load_checkpoint_fm(funcmol, config["fm_pretrained_path"], fabric=fabric)
        funcmol = fabric.setup_module(funcmol)

        nf_checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
        config_nf = nf_checkpoint["config"]
        _, dec = load_neural_field(nf_checkpoint, fabric, config=config_nf)
        dec_module = dec.module if hasattr(dec, "module") else dec
        dec_module.set_code_stats(code_stats)

    # ----------------------
    # sample
    fabric.print(">> saving samples in", config["dirname"])
    sample(funcmol, dec_module, config, fabric)

if __name__ == "__main__":
    main()
