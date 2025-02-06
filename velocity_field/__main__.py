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
    permeability_map = np.zeros_like(sample, dtype=float)
    for i in range(sample.shape[axis]):
        slc = np.take(sample, i, axis=axis)
        idx = [slice(None)] * sample.ndim
        idx[axis] = i
        permeability_map[tuple(idx)] = _get_slice_permeability_map(slc)

    return permeability_map


def main():
    sample_path = Path(sys.argv[1])
    sample = cubify_voxels(sample_path, dtype=np.uint8)
    sampler = SubarraySampler(sample)
    mgr = LBPMManager(vl=100.0)

    pondered_perm_x_list = []
    pondered_perm_y_list = []
    pondered_perm_z_list = []

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

        pondered_perm_x_list.append(pondered_perm_x)
        pondered_perm_y_list.append(pondered_perm_y)
        pondered_perm_z_list.append(pondered_perm_z)

    final_perm_x = np.concatenate(pondered_perm_x_list, axis=0)
    final_perm_y = np.concatenate(pondered_perm_y_list, axis=0)
    final_perm_z = np.concatenate(pondered_perm_z_list, axis=0)

    output_dir = Path("data/velocity_field/outputs")
    output_dir.mkdir(exist_ok=True)

    final_perm_x.tofile(output_dir / f"{sample_path.stem}_permeability_x.raw")
    final_perm_y.tofile(output_dir / f"{sample_path.stem}_permeability_y.raw")
    final_perm_z.tofile(output_dir / f"{sample_path.stem}_permeability_z.raw")


if __name__ == "__main__":
    main()
