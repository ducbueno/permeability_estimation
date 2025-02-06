import re
import sys
from pathlib import Path

import numpy as np

TYPE_MAP = {"uint8": np.uint8, "float32": np.float32}


def ct2porosity(sample_path, ctn_brine, ctn_n2=-872.5):
    sample = np.fromfile(sample_path, dtype=np.float32)
    sample[sample < -1e-4] = 0.0
    sample = sample / (ctn_brine - ctn_n2)
    sample[sample < 0.0] = 0.0
    sample.tofile(sample_path)


def cubify_voxels(sample_path, dtype):
    sample = np.fromfile(sample_path, dtype=dtype)

    if not (dims_match := re.search(r"(\d+)x(\d+)x(\d+)", sample_path.name)):
        raise ValueError("Filename must contain dimensions in format: n1xn2xn3")

    shape = tuple(reversed(tuple(int(dim) for dim in dims_match.groups())))
    print(f"Original shape: {shape}")
    sample = sample.reshape(shape)
    expanded_shape = (shape[0] * 7, *shape[1:])
    print(f"Expanded shape: {expanded_shape}")
    expanded_sample = np.zeros(expanded_shape, dtype=dtype)

    for i in range(shape[0]):
        start_idx = i * 7
        end_idx = start_idx + 7
        expanded_sample[start_idx:end_idx, ...] = sample[i, ...]

    expanded_shape_str = f"{expanded_shape[0]}x{expanded_shape[1]}x{expanded_shape[2]}"
    new_filename = (
        re.sub(r"\d+x\d+x\d+", f"{expanded_shape_str}", sample_path.stem) + ".raw"
    )
    expanded_sample.tofile(sample_path.parent / new_filename)


def main():
    sample_path = Path(sys.argv[1])
    ct2porosity(sample_path, ctn_brine=231.29)
    cubify_voxels(sample_path, TYPE_MAP[sys.argv[2]])


if __name__ == "__main__":
    main()
