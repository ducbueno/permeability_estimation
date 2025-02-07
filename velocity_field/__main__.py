"""Computes permeability maps for porous media samples using velocity field data.

This module processes 3D voxel data of porous media samples to calculate permeability
maps along different axes. It combines geometric properties of the pore space with
velocity field data from LBPM simulations to generate weighted permeability maps.

Typical usage example:
    python -m velocity_field sample.raw
"""

import sys
from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops

from lbpm_manager.lbpm_manager import LBPMManager
from preprocessing import cubify_voxels
from subarray_sampler import SubarraySampler


def _get_slice_permeability_map(slc):
    """Calculates the permeability map for a 2D slice of porous media.

    Uses pore geometry and connectivity to compute local permeability values
    based on shape factors and hydraulic diameters.

    Args:
        slc: numpy.ndarray
            A 2D binary array representing a slice of porous media.

    Returns:
        numpy.ndarray
            A 2D array of float values representing the permeability map.
    """
    slice_porosity = np.count_nonzero(slc) / slc.size
    connectivity_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled_pores, _ = label(slc, structure=connectivity_kernel)
    properties = regionprops(labeled_pores)

    permeability_map = np.zeros_like(labeled_pores, dtype=float)
    for prop in properties:
        area = prop.area
        perimeter = prop.perimeter
        shape_factor = 4 * np.pi * area / perimeter**2
        hydraulic_diameter = 4 * area / perimeter
        permeability = (slice_porosity * shape_factor * hydraulic_diameter**2) / 32
        permeability_map[labeled_pores == prop.label] = permeability

    return permeability_map


def get_permeability_map(sample, axis):
    """Generates a 3D permeability map along a specified axis.

    Computes permeability maps for each 2D slice perpendicular to the given axis
    and combines them into a 3D array.

    Args:
        sample: numpy.ndarray
            3D binary array representing the porous media sample.
        axis: int
            Axis along which to compute the permeability (0: z, 1: x, 2: y).

    Returns:
        numpy.ndarray
            3D array containing permeability values.
    """
    permeability_map = np.zeros_like(sample, dtype=float)
    for i in range(sample.shape[axis]):
        slc = np.take(sample, i, axis=axis)
        idx = [slice(None)] * sample.ndim
        idx[axis] = i
        permeability_map[tuple(idx)] = _get_slice_permeability_map(slc)

    return permeability_map


def main():
    """Main execution function.

    Processes a binary voxel file to compute directional permeability maps weighted
    by velocity field data. Saves the resulting permeability maps as raw binary files.

    Command line arguments:
        sys.argv[1]: Path to the input voxel file.

    Outputs:
        Creates binary files containing permeability maps for x, y, and z directions
        in the 'data/velocity_field/outputs' directory.
    """
    sample_path = Path(sys.argv[1])
    sample = cubify_voxels(sample_path, dtype=np.uint8)
    sampler = SubarraySampler(sample)
    mgr = LBPMManager(vl=100.0)

    output_dir = Path("data/velocity_field/outputs")
    output_dir.mkdir(exist_ok=True)

    for i, (subarray, _) in enumerate(
        sampler.sample((sample.shape[0] // 20, sample.shape[1], sample.shape[2]))
    ):
        print(f"Computing permeability maps (XYZ) for subarray {i}")
        perm_x = get_permeability_map(subarray, axis=1)
        perm_y = get_permeability_map(subarray, axis=2)
        perm_z = get_permeability_map(subarray, axis=0)

        print(f"Running simulation for subarray {i}")
        data = mgr.run_simulation(subarray)

        vx = data["Velocity_x"][: subarray.shape[0], :, :]
        vy = data["Velocity_y"][: subarray.shape[0], :, :]
        vz = data["Velocity_z"][: subarray.shape[0], :, :]
        v_abs = np.sqrt(vx**2 + vy**2 + vz**2)
        vx_norm = vx / v_abs
        vy_norm = vy / v_abs
        vz_norm = vz / v_abs

        print(f"Computing podered permeability XYZ maps for subarray {i}")
        pondered_perm_x = perm_x * vx_norm
        pondered_perm_y = perm_y * vy_norm
        pondered_perm_z = perm_z * vz_norm

        pondered_perm_x.tofile(
            output_dir / f"{sample_path.stem}_permeability_x_{i}.raw"
        )
        pondered_perm_y.tofile(
            output_dir / f"{sample_path.stem}_permeability_y_{i}.raw"
        )
        pondered_perm_z.tofile(
            output_dir / f"{sample_path.stem}_permeability_z_{i}.raw"
        )


if __name__ == "__main__":
    main()
