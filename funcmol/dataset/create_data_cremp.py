import argparse
import gc
import os
import torch
import pickle

from pyuul import utils
import numpy as np
import pandas as pd
from p_tqdm import p_map

from rdkit import Chem
from rdkit import RDLogger

from funcmol.utils.utils_base import atomlistToRadius
from funcmol.utils.constants import ELEMENTS_HASH, radiusSingleAtom, PADDING_INDEX


RDLogger.DisableLog("rdApp.*")

CREMP_MAXNATOMS = 150


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def split_data(data_dir):
    """
    Splits the dataset into training, validation, and test sets and saves them as CSV files.

    Parameters:
    data_dir (str): The directory where the input 'summary.csv' file is located and where the output CSV files will be saved.
    """
    dataset = pd.read_csv(os.path.join(data_dir, "summary.csv"))

    n_samples = len(dataset)
    n_train = 30000
    n_test = int(0.1 * n_samples)
    n_val = n_samples - (n_train + n_test)

    # shuffle dataset with df.sample, then split
    # sample split as previous work
    train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

    train.to_csv(os.path.join(data_dir, "train.csv"))
    val.to_csv(os.path.join(data_dir, "val.csv"))
    test.to_csv(os.path.join(data_dir, "test.csv"))


def pyuul_from_mol(mol: Chem.Mol, ii: int):
    """
    Generate a padded tensor of atom coordinates and formatted atom names from a molecule.

    Args:
        mol (Chem.Mol): A molecule object from RDKit.
        ii (int): The index of the conformer to use.

    Returns:
        Tuple[torch.Tensor, List[str]]: A tuple containing:
            - A padded tensor of atom coordinates.
            - A list of formatted atom names.
    """
    # just give one conformation
    conf = mol.GetConformer(ii)  # 0 means the first conformer

    # Collect atom names (elemental types), e.g., H, C, O, N, etc.
    atom_names = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if len(list(set(atom_names))) > 5:
        print(list(set(atom_names)))

    # Collect the atom coordinates as a Torch tensor
    atom_coords_float = [list(conf.GetAtomPosition(i)) for i in range(conf.GetNumAtoms())]
    atom_coords = torch.tensor([atom_coords_float])

    # Format atom names similar to parseSDF method
    formatted_atom_names = ["MOL_0_" + atom_name + "_A" for atom_name in atom_names]

    return torch.nn.functional.pad(
        atom_coords, (0, 0, 0, CREMP_MAXNATOMS - atom_coords.shape[1]), "constant", PADDING_INDEX
    ), formatted_atom_names


def worker(iirow, n_conformations, data_dir):
    ii, row = iirow
    with open(os.path.join(data_dir, f"pickle/{row.sequence}.pickle"), "rb") as f:
        it = pickle.load(f)

    mol = it["rd_mol"]

    data = []
    for jj in range(mol.GetNumConformers()):
        if jj == n_conformations:
            break
        if it["conformers"][jj]["boltzmannweight"] < 0.01:
            continue

        coords, atname = pyuul_from_mol(it["rd_mol"], jj)
        atoms_channel = utils.atomlistToChannels([atname], hashing=ELEMENTS_HASH)
        atoms_channel = torch.nn.functional.pad(
            atoms_channel, (0, CREMP_MAXNATOMS - atoms_channel.shape[1]), "constant", PADDING_INDEX
        )

        radius = atomlistToRadius([atname], hashing=radiusSingleAtom)
        radius = torch.nn.functional.pad(
            radius, (0, CREMP_MAXNATOMS - radius.shape[1]), "constant", PADDING_INDEX
        )

        smiles = it["smiles"]

        data.append(
            {
                # This contains WAY too much data.
                # "mol": it['rd_mol'],
                "mol": Chem.rdmolfiles.MolFromSmiles(smiles),
                "smiles": smiles,
                "coords": coords[0].clone(),
                "atoms_channel": atoms_channel[0].clone(),
                "radius": radius[0].clone(),
                "target": row.to_dict(),
            }
        )

    return data, ii < 5000


def worker2(iirow, data_dir):
    ii, row = iirow
    # print(row)
    with open(os.path.join(data_dir, f"pickle/{ii}.pickle"), "rb") as f:
        it = pickle.load(f)

    return it["rd_mol"].GetConformer(0).GetNumAtoms()


def preprocess_cremp_dataset(data_dir, n_conformations=1, split="train"):
    """
    Preprocess the CREMP dataset by reading the target CSV file, applying parallel processing
    to generate conformations, and filtering the results.

    Args:
        data_dir (str): The directory where the dataset is located.
        n_conformations (int, optional): The number of conformations to generate for each molecule. Default is 1.
        split (str, optional): The dataset split to process (e.g., "train", "test"). Default is "train".

    Returns:
        tuple: A tuple containing two lists:
            - data (list): A flattened list of all generated conformations.
            - data_small (list): A flattened list of generated conformations that passed the filtering criteria.
    """
    target_df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"), index_col=0)
    # Use p_map for parallelization
    results = p_map(
        worker, target_df.iterrows(), [n_conformations] * len(target_df), [data_dir] * len(target_df)
    )

    # Filter results
    data = [r[0] for r in results]
    data_small = [r[0] for r in results if r[1]]

    # flatten data
    # print(len(data))
    # print(len(data[0]))
    data = [it for sb in data for it in sb]
    data_small = [it for sb in data_small for it in sb]
    # quit()

    print(f"  >> split size: {len(data)} ({0} errors)")

    return data, data_small


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cremp/")
    parser.add_argument("--n_conformations", type=int, default=50)
    args = parser.parse_args()

    split_data(args.data_dir)

    data, data_small = {}, {}
    for split in ["train", "test", "val"]:
        print(f">> preprocessing {split}...")

        dset, dset_small = preprocess_cremp_dataset(args.data_dir, args.n_conformations, split=split)
        torch.save(
            dset,
            os.path.join(args.data_dir, f"{split}_{args.n_conformations}_data.pth"),
        )
        torch.save(
            dset_small, os.path.join(args.data_dir, f"{split}_{args.n_conformations}_data_small.pth")
        )

        del dset, dset_small
        gc.collect()
