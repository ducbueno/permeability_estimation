import math

import numpy as np
from scipy.stats import qmc


class SubarraySampler:
    def __init__(self, array, invert=False):
        self.invert = invert
        self.array = array

    def sample_random(self, num_samples: int, factor: int = 10):
        """Sample subarrays using Latin Hypercube Sampling.

        Args:
            num_samples: Number of samples to generate
            factor: Division factor for subarray size

        Yields:
            np.ndarray: Subarray samples
        """
        full_dims = np.array(self.array.shape)
        sub_dims = full_dims // factor
        half_sub_dims = sub_dims // 2

        samples_per_dim = math.ceil(num_samples ** (1 / 3))
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=samples_per_dim**3)

        centers = (samples * (full_dims - sub_dims) + half_sub_dims).astype(int)

        for center in centers:
            slices = tuple(slice(c - h, c + h) for c, h in zip(center, half_sub_dims))
            yield self.array[slices]

    def sample(self, dims):
        """Sample subarrays in a deterministic way.

        Args:
            dim: Either a single integer for cubic samples (length, length, length),
                or a tuple of 3 integers for custom dimensions (x, y, z)

        Yields:
            Tuple containing:
                - np.ndarray: Subarray sample of shape determined by length parameter
                - np.ndarray: Starting position of the sample in the original array
        """
        if isinstance(dims, int):
            dims = (dims, dims, dims)

        full_dims = np.array(self.array.shape)
        samples_per_dim = full_dims // dims

        start_points = np.array(
            np.meshgrid(
                np.arange(0, samples_per_dim[0] * dims[0], dims[0]),
                np.arange(0, samples_per_dim[1] * dims[1], dims[1]),
                np.arange(0, samples_per_dim[2] * dims[2], dims[2]),
                indexing="ij",
            )
        )

        start_points = start_points.reshape(3, -1).T

        for start in start_points:
            slices = tuple(slice(s, s + dim) for s, dim in zip(start, dims))
            yield self.array[slices], start
