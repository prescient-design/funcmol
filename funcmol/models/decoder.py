from functools import partial
import torch
from torch import nn
import numpy as np
from scipy import ndimage as ndi
from collections import defaultdict
from tqdm import tqdm
from itertools import chain
from funcmol.models.mfn import GaborNet


class Decoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        hidden_dim: int = 2048,
        code_dim: int = 2048,
        coord_dim: int = 3,
        n_layers: int = 6,
        input_scale: int = 64,
        grid_dim: int = 32,
        fabric=None,
    ):
        """
        Initializes the Decoder class.

        Args:
            n_channels (int): Number of channels for the output.
            hidden_dim (int, optional): Dimension of the hidden layers. Defaults to 2048.
            code_dim (int, optional): Dimension of the code. Defaults to 2048.
            coord_dim (int, optional): Dimension of the coordinates. Defaults to 3.
            n_layers (int, optional): Number of layers in the network. Defaults to 6.
            input_scale (int, optional): Scale of the input. Defaults to 64.
            grid_dim (int, optional): Dimension of the grid. Defaults to 32.
            fabric (optional): Fabric object for device management and printing. Defaults to None.
        """
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.code_dim = code_dim
        self.grid_dim = grid_dim
        self.fabric = fabric
        self.coords = get_grid(self.grid_dim)[1].unsqueeze(0).to(self.fabric.device)
        fabric.print(">> coords shape:", self.coords.shape)
        self.net = GaborNet(
            self.coord_dim,
            self.hidden_dim,
            self.code_dim,
            n_channels,
            n_layers,
            input_scale=input_scale,
        )

    def forward(self, x: torch.Tensor, codes: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the INRDecoder module.

        Args:
            x: Input tensor.
            codes: Optional tensor of codes.

        Returns:
            Tensor: Output tensor.
        """
        return self.net(x, codes)

    def render_code(
        self,
        codes: torch.Tensor,
        batch_size_render: int,
        fabric: object = None,
    ) -> torch.Tensor:
        """
        Renders molecules from given codes.

        Args:
            codes (torch.Tensor): Tensor containing the codes to be rendered.
                                  The codes need to be unnormalized before rendering.
            batch_size_render (int): The size of the batches for rendering.
            fabric (object, optional): An object that provides a print method for logging.
                                       Defaults to None.

        Returns:
            torch.Tensor: A tensor representing the rendered molecules in a grid format.
        """
        # PS: code need to be unnormealized before rendering
        with torch.no_grad():
            fabric.print(f">> Rendering molecules - batches of {batch_size_render}")
            if codes.device != self.fabric.device:
                codes = codes.to(self.fabric.device)
            xs = self.coords.reshape(1, -1, 3)
            pred = self.forward_batched(xs, codes, batch_size_render=batch_size_render, threshold=0.2)
            grid = pred.permute(0, 2, 1).reshape(-1, self.n_channels, self.grid_dim, self.grid_dim, self.grid_dim)
        return grid

    def forward_batched(self, xs, codes, batch_size_render=100_000, threshold=None, to_cpu=True):
        """
        When memory is limited, render the grid in batches.
        """
        pred_list = []
        batched_xs = torch.split(xs, batch_size_render, dim=1)
        for x in tqdm(batched_xs):
            pred_batched = self(x, codes)
            if to_cpu:
                pred_batched = pred_batched.cpu()
            if threshold is not None:
                pred_batched[pred_batched < threshold] = 0
            pred_list.append(pred_batched)
        pred = torch.cat(pred_list, dim=1)
        return pred

    def codes_to_molecules(
        self,
        codes: torch.Tensor,
        unnormalize: bool = True,
        config: dict = None,
        fabric: object = None,
    ) -> list:
        """
        Convert codes to molecular structures.

        Args:
            codes (torch.Tensor): The input codes representing molecular structures.
            unnormalize (bool): Flag indicating whether to unnormalize the codes.
            config (dict): Configuration dictionary containing parameters for the conversion.
            fabric (optional): An optional fabric object for logging and printing.

        Returns:
            list: A list of refined molecular structures.
        """
        mols_dict = defaultdict(list)
        codes_dict = defaultdict(list)

        codes = codes.detach()
        if unnormalize:
            codes = self.unnormalize_code(codes).detach()

        mols_dict, codes_dict = self.codes_to_grid(
            codes=codes,
            mols_dict=mols_dict,
            codes_dict=codes_dict,
            config=config,
            fabric=fabric,
        )
        fabric.print("(n_atoms, n_samples): ", end=" ")
        for key, value in sorted(mols_dict.items()):
            fabric.print(f"({key}, {len(value)})", end=" ")
        fabric.print()

        # Refine coordinates
        mols = self._refine_coords(
            grouped_mol_inits=mols_dict,
            grouped_codes=codes_dict,
            maxiter=200,
            grid_dim=self.grid_dim,
            resolution=config["dset"]["resolution"],
            fabric=fabric,
        )

        return mols

    def codes_to_grid(
        self,
        codes: torch.Tensor,
        mols_dict: dict = defaultdict(list),
        codes_dict: dict = defaultdict(list),
        config=None,
        fabric=None,
    ) -> tuple:
        """
        Converts a batch of codes to grids and extracts atom coordinates.

        Args:
            batched_codes (torch.Tensor): The batch of codes to convert to grids.
            mols_dict (dict, optional): A dictionary to store molecule objects. Defaults to defaultdict(list).
            codes_dict (dict, optional): A dictionary to store the corresponding codes. Defaults to defaultdict(list).

        Returns:
            tuple: A tuple containing the grid shape, molecule objects, dictionaries of molecule objects, codes, and indices.
        """
        # 1. render grid
        grids = self.render_code(codes, config["wjs"]["batch_size_render"], fabric)

        # 2. find peaks (atom coordinates)
        fabric.print(">> Finding peaks")
        for idx, grid in tqdm(enumerate(grids)):
            mol_init = get_atom_coords(grid, rad=config["dset"]["atomic_radius"])
            if mol_init is not None:
                mol_init = _normalize_coords(mol_init, self.grid_dim)
                num_coords = int(mol_init["coords"].size(1))
                if num_coords <= 500:
                    mols_dict[num_coords].append(mol_init)
                    codes_dict[num_coords].append(codes[idx].cpu())
                else:
                    fabric.print(f"Molecule {idx} has more than 500 atoms")
            else:
                fabric.print(f"No atoms found in grid {idx}")
        return mols_dict, codes_dict

    def _refine_coords(
        self,
        grouped_mol_inits: dict,
        grouped_codes: dict,
        maxiter: int = 10,
        grid_dim: int = 32,
        resolution: int = 0.25,
        fabric=None,
    ) -> list:
        """
        Refines the coordinates of molecules in batches and handles errors during the process.

        Args:
            grouped_mol_inits (dict): A dictionary where keys are group identifiers and values are lists of initial molecule data.
            grouped_codes (dict): A dictionary where keys are group identifiers and values are lists of codes corresponding to the molecules.
            maxiter (int, optional): Maximum number of iterations for the refinement process. Default is 10.
            grid_dim (int, optional): Dimension of the grid used for normalization. Default is 32.
            resolution (int, optional): Resolution used for normalization. Default is 0.25.
            fabric (optional): An object with a print method for logging messages.

        Returns:
            list: A list of refined molecule data.
        """
        fabric.print(">> Refining molecules")
        for key, mols in tqdm(grouped_mol_inits.items()):
            try:
                coords = self._refine_coords_batch(
                    grouped_mol_inits[key],
                    grouped_codes[key],
                    maxiter=maxiter,
                    fabric=fabric,
                )
                for i in range(coords.size(0)):
                    mols[i]["coords"] = coords[i].unsqueeze(0)
                    mols[i] = _unnormalize_coords(mols[i], grid_dim, resolution)
            except Exception as e:
                fabric.print(f"Error refinement: {e}")
                for i in range(len(grouped_mol_inits[key])):
                    try:
                        mols[i] = _unnormalize_coords(mols[i], grid_dim, resolution)
                    except Exception:
                        fabric.print(
                            f"Error unnormalization: {e} for {i}/{len(grouped_mol_inits[key])}"
                        )
        return list(chain.from_iterable(grouped_mol_inits.values()))

    def _refine_coords_batch(
        self,
        mols_init: list,
        codes: list,
        maxiter: int = 10,
        batch_size_refinement: int = 100,
        fabric=None,
    ) -> torch.Tensor:
        """
        Refines the coordinates of molecules in batches.

        Args:
            mols_init (list): List of initial molecule data, where each element is a dictionary containing
                              'coords' and 'atoms_channel' tensors.
            codes (list): List of codes corresponding to each molecule.
            maxiter (int, optional): Maximum number of iterations for the optimizer. Default is 10.
            batch_size_refinement (int, optional): Number of molecules to process in each batch. Default is 100.
            fabric (optional): Fabric object that provides device and optimizer setup.

        Returns:
            torch.Tensor: Refined coordinates of all molecules concatenated along the first dimension.
        """
        num_batches = len(mols_init) // batch_size_refinement
        if len(mols_init) % batch_size_refinement != 0:
            nb_iter = num_batches + 1
        else:
            nb_iter = num_batches
        refined_coords = []

        for i in range(nb_iter):
            min_bound = i * batch_size_refinement
            max_bound = (len(mols_init) if i == num_batches else (i + 1) * batch_size_refinement)
            coords = torch.stack(
                [mols_init[j]["coords"].squeeze(0) for j in range(min_bound, max_bound)], dim=0,
            ).to(fabric.device)
            coords_init = coords.clone()
            coords.requires_grad = True

            with torch.no_grad():
                atoms_channel = torch.stack(
                    [mols_init[j]["atoms_channel"] for j in range(min_bound, max_bound)], dim=0,
                ).to(fabric.device)
                occupancy = (
                    torch.nn.functional.one_hot(atoms_channel.long(), self.n_channels).float().squeeze()
                ).to(fabric.device)
                code = torch.stack(
                    [codes[j] for j in range(min_bound, max_bound)], dim=0
                ).to(fabric.device)

            def closure():
                optimizer.zero_grad()
                pred = self.net(coords, code)
                loss = -(pred * occupancy).max(dim=2).values.mean()
                loss.backward()
                return loss

            optim_factory = partial(
                torch.optim.LBFGS,
                history_size=10,
                max_iter=4,
                line_search_fn="strong_wolfe",
                lr=1.0,
            )
            optimizer = fabric.setup_optimizers(optim_factory([coords]))
            tol, loss = 1e-4, 1e10
            for _ in range(maxiter):
                prev_loss = loss
                loss = optimizer.step(closure)
                if abs(loss - prev_loss).item() < tol:
                    break
                if (coords - coords_init).abs().max() > 1:
                    print("Refine coords diverges, so use initial coordinates...")
                    coords = coords_init
                    break
            refined_coords.append(coords.detach().cpu())
        return torch.cat(refined_coords, dim=0)

    def set_code_stats(self, code_stats: dict) -> None:
        """
        Set the code statistics.

        Args:
            code_stats: Code statistics.
        """
        self.code_stats = code_stats

    def unnormalize_code(self, codes: torch.Tensor):
        """
        Unnormalizes the given codes based on the provided normalization parameters.

        Args:
            codes (torch.Tensor): The codes to be unnormalized.
            code_stats (dict): The statistics for the codes.

        Returns:
            torch.Tensor: The unnormalized codes.
        """
        mean, std = (
            self.code_stats["mean"].to(codes.device),
            self.code_stats["std"].to(codes.device),
        )
        return codes * std + mean


########################################################################################
## auxiliary functions
def local_maxima(data, order=1):
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data.numpy(), footprint=footprint)
    filtered = torch.from_numpy(filtered)
    data[data <= filtered] = 0
    return data


def find_peaks(voxel):
    voxel = voxel.squeeze()
    voxel[voxel < 0.25] = 0
    return torch.cat(
        [
            local_maxima(voxel[channel_idx], 1).unsqueeze(0)
            for channel_idx in range(voxel.shape[0])
        ],
        dim=0,
    )


def get_atom_coords(grid, rad=0.5):
    peaks = find_peaks(grid)
    # current version only works for fixed radius (ie, all atoms with same radius rad)
    coords = []
    atoms_channel = []
    radius = []

    for channel_idx in range(peaks.shape[0]):
        px, py, pz = torch.where(peaks[channel_idx] > 0)
        if px.numel() > 0:
            px, py, pz = px.float(), py.float(), pz.float()
            coords.append(torch.stack([px, py, pz], dim=1))
            atoms_channel.append(
                torch.full((px.shape[0],), channel_idx, dtype=torch.float32)
            )
            radius.append(torch.full((px.shape[0],), rad, dtype=torch.float32))

    if not coords:
        return None

    structure = {
        "coords": torch.cat(coords, dim=0).unsqueeze(0),
        "atoms_channel": torch.cat(atoms_channel, dim=0).unsqueeze(0),
        "radius": torch.cat(radius, dim=0).unsqueeze(0),
    }

    return structure


def _normalize_coords(mol, grid_dim):
    mol["coords"] -= (grid_dim - 1) / 2
    mol["coords"] /= grid_dim / 2
    return mol


def _unnormalize_coords(mol, grid_dim, resolution=0.25):
    mol["coords"] *= grid_dim / 2
    mol["coords"] *= resolution
    return mol


def get_grid(grid_dim):
    discrete_grid = (np.arange(grid_dim) - (grid_dim // 2)) / (grid_dim // 2)
    full_grid = torch.Tensor(
        [[a, b, c] for a in discrete_grid for b in discrete_grid for c in discrete_grid]
    )
    return discrete_grid, full_grid