# see:
# https://github.com/Genentech/voxmol/blob/main/voxmol/dataset/create_data_geomdrugs.py

import argparse
import gc
import os
import pickle
import torch
import urllib.request

from pyuul import utils
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

from funcmol.utils.utils_base import atomlistToRadius
from funcmol.utils.constants import ELEMENTS_HASH, radiusSingleAtom

RDLogger.DisableLog("rdApp.*")

RAW_URL_TRAIN = "https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle"
RAW_URL_VAL = "https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle"
RAW_URL_TEST = "https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle"


def download_data(raw_data_dir: str):
    """
    Download the raw data files from the specified URLs and save them in the given directory.

    Args:
        raw_data_dir (str): The directory where the raw data files will be saved.

    Returns:
        None
    """
    urllib.request.urlretrieve(RAW_URL_TRAIN, os.path.join(raw_data_dir, "train_data.pickle"))
    urllib.request.urlretrieve(RAW_URL_VAL, os.path.join(raw_data_dir, "val_data.pickle"))
    urllib.request.urlretrieve(RAW_URL_TEST, os.path.join(raw_data_dir, "test_data.pickle"))


def preprocess_geom_drugs_dataset(raw_data_dir: str, data_dir: str, split: str = "train"):
    """
    Preprocesses the geometry drugs dataset.

    Args:
        raw_data_dir (str): The directory path where the raw data is stored.
        data_dir (str): The directory path where the preprocessed data will be saved.
        split (str, optional): The dataset split to preprocess. Defaults to "train".

    Returns:
        tuple: A tuple containing two lists: the preprocessed data for the specified
            split and a smaller subset of the data.
    """
    print("  >> load data raw from ", os.path.join(raw_data_dir, f"{split}_data.pickle"))
    with open(os.path.join(raw_data_dir, f"{split}_data.pickle"), 'rb') as f:
        all_data = pickle.load(f)

    # get all conformations of all molecules
    mols_confs = []
    for i, data in enumerate(all_data):
        _, all_conformers = data
        for j, conformer in enumerate(all_conformers):
            if j >= 5:
                break
            mols_confs.append(conformer)

    # write sdf / load with PyUUL
    print("  >> write .sdf of all conformations and extract coords/types with PyUUL")
    sdf_path = os.path.join(data_dir, f"{split}.sdf")
    with Chem.SDWriter(sdf_path) as w:
        for m in mols_confs:
            w.write(m)
    coords, atname = utils.parseSDF(sdf_path)
    atoms_channel = utils.atomlistToChannels(atname, hashing=ELEMENTS_HASH)
    radius = atomlistToRadius(atname, hashing=radiusSingleAtom)

    # create the dataset
    print("  >> create the dataset for this split")
    data, data_small = [], []
    num_errors = 0
    for i, mol in enumerate(tqdm(mols_confs)):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if smiles is None:
            num_errors += 1
        datum = {
            "mol": mol,
            "smiles": smiles,
            "coords": coords[i].clone(),
            "atoms_channel": atoms_channel[i].clone(),
            "radius": radius[i].clone(),
        }

        data.append(datum)
        if i < 5000:
            data_small.append(datum)
    print(f"  >> split size: {len(data)} ({num_errors} errors)")

    return data, data_small


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="data/drugs/raw/")
    parser.add_argument("--data_dir", type=str, default="data/drugs/")
    args = parser.parse_args()

    if not os.path.isdir(args.raw_data_dir):
        os.makedirs(args.raw_data_dir, exist_ok=True)
        download_data(args.raw_data_dir)

    os.makedirs(args.data_dir, exist_ok=True)

    data, data_small = {}, {}
    for split in ["train", "val", "test"]:
        print(f">> preprocessing {split}...")

        dset, dset_small = preprocess_geom_drugs_dataset(args.raw_data_dir, args.data_dir, split)
        torch.save(dset, os.path.join(args.data_dir, f"{split}_data.pth"),)
        torch.save(dset_small, os.path.join(args.data_dir, f"{split}_data_small.pth"),)

        del dset, dset_small
        gc.collect()
