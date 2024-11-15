import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
import plotly.io as pio
pio.kaleido.scope.mathjax = None


COLORS_SPARSE = [
    "Greys_r",
    "Greys",
    "Reds_r",
    "Blues_r",
    "Greens_r",
    "Mint_r",
    "Oranges_r",
    "Magenta_r",
    "Purples_r",
    "Peach_r",
    [[0, "rgb(0,0,0)"], [1, "rgb(255,191,0)"]],  # Yellow
    [[0, "rgb(0,0,0)"], [1, "rgb(175, 96, 26)"]],  # Brown
]


COLORS = [
    "Greys",
    "Greys_r",
    "Reds",
    "Blues",
    "Greens",
    "Mint",
    "Oranges",
    "Magenta",
    "Purples",
    "Peach",
    [[0, "rgb(0,0,0)"], [1, "rgb(255,191,0)"]],  # Yellow
    [[0, "rgb(0,0,0)"], [1, "rgb(175, 96, 26)"]],  # Brown
]



def visualize_voxel_grid(
    voxel: torch.Tensor,
    fname: str = "figures/temp.png",
    threshold: float = 0.1,
    to_png: bool = True,
    to_html: bool = False,
    sparse: bool = True,
):
    """
    Visualizes a voxel grid using volume rendering.

    Args:
        voxel (torch.Tensor): The voxel grid tensor of shape CxLxLxL,
            where C is the number of channels and L is the grid size.
        fname (str, optional): The file path to save the visualization image. Defaults to "figures/temp.png".
        threshold (float, optional): The threshold value to remove voxels below this value. Defaults to 0.1.
        to_png (bool, optional): Whether to save the visualization as a PNG image. Defaults to True.
        to_html (bool, optional): Whether to save the visualization as an HTML file. Defaults to False.
        sparse (bool, optional): Whether to visualize the sparse voxel grid (for faster visualization). Defaults to True.
    """
    sns.set_theme()
    voxel = voxel.detach().cpu().numpy()
    assert len(voxel.shape) == 4, "voxel grid need to be of the form CxLxLxL"
    voxel[voxel < threshold] = 0
    if not sparse:
        X, Y, Z = np.mgrid[: voxel.shape[-3], : voxel.shape[-2], : voxel.shape[-1]]

    fig = go.Figure()
    for channel in range(voxel.shape[0]):
        if not sparse:
            voxel_channel = voxel[channel : channel + 1]
            if voxel_channel.sum().item() == 0:
                continue
            fig.add_volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=voxel_channel.flatten(),
                isomin=0.19,
                isomax=0.2,
                opacity=0.1, # needs to be small to see through all surfaces
                surface_count=17,  # needs to be a large number for good volume rendering
                colorscale=COLORS[channel],
                showscale=False,
            )
        else:
            voxel_channel = voxel[channel]
            non_zero_indices = np.nonzero(voxel_channel)
            if len(non_zero_indices[0]) == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=non_zero_indices[0],
                y=non_zero_indices[1],
                z=non_zero_indices[2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=voxel_channel[non_zero_indices],
                    colorscale=COLORS_SPARSE[channel],
                    opacity=1.0
                ),
                showlegend=False
            ))
    if sparse:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[0, voxel.shape[-3]]),
                yaxis=dict(range=[0, voxel.shape[-2]]),
                zaxis=dict(range=[0, voxel.shape[-1]]),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )
    if to_html:
        fig.write_html(fname.replace("png", "html"))
    if to_png:
        fig.write_image(fname, format='png', validate=False)
