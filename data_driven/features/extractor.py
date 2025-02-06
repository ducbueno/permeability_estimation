import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from subarray_sampler import SubarraySampler

sys.path.append("../lbpm_manager/")
from quantimpy import minkowski as mk  # pyright: ignore

from lbpm_manager import LBPMManager  # pyright: ignore


class FeatureExtractor:
    def __init__(self, input_file, vl=8.0, invert=True):
        self.lbpm_manager = LBPMManager(vl=vl)
        self.sampler = SubarraySampler(input_file, invert=invert)
        self.directions = {0: "z", 1: "y", 2: "x"}

    def _compute_statistics(self, data):
        mean = np.mean(data)
        median = np.median(data)
        mode = (
            data[np.argmax(np.bincount(data.astype(int)))]
            if data.dtype.kind in {"i", "u"}
            else np.nan
        )
        stats = {
            "mean": mean,
            "median": median,
            "mode": mode,
            "variance": np.var(data, ddof=1),
            "std_dev": np.std(data, ddof=1),
            "range": np.ptp(data),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25),
            "mad": np.median(np.abs(data - median)),
            "skewness": skew(data),
            "kurtosis": kurtosis(data),
        }

        for p in [1, 25, 50, 75, 99]:
            stats[f"percentile_{p}"] = np.percentile(data, p)

        return stats

    def _compute_2d_features(self, subarray, axis, direction_name):
        shape = subarray.shape[axis]
        eulerc_2d = np.zeros(shape)
        specific_lengths = np.zeros(shape)
        porosity_2d = np.zeros(shape)

        slices = [slice(None)] * 3
        for i in range(shape):
            slices[axis] = i  # pyright: ignore
            slc = subarray[tuple(slices)]

            area, length, eulerc = mk.functionals(slc.astype(bool))
            eulerc_2d[i] = eulerc
            specific_lengths[i] = length / (area ** (1 / 2)) if area > 0 else np.nan
            porosity_2d[i] = np.mean(slc)

        stats_dict = {}
        for name, data in [
            ("eulerc_2d", eulerc_2d),
            ("specific_lengths", specific_lengths),
            ("porosity_2d", porosity_2d),
        ]:
            stats = self._compute_statistics(data)
            stats_dict |= {f"{name}_{direction_name}_{k}": v for k, v in stats.items()}

        return stats_dict

    def _extract_features_train(self, n_samples, factor=10):
        features_list = []

        try:
            for i, subarray in enumerate(
                self.sampler.sample_random(n_samples, factor=factor)
            ):
                print(f"Processing subarray {i + 1}\n")

                features = {}
                for axis, direction in self.directions.items():
                    features |= self._compute_2d_features(subarray, axis, direction)

                volume, surface_area, _, eulerc_3d = mk.functionals(
                    subarray.astype(bool)
                )
                features.update(
                    {
                        "specific_surface_area": (
                            surface_area / (volume ** (2 / 3)) if volume > 0 else np.nan
                        ),
                        "eulerc_3d": eulerc_3d,
                        "porosity_3d": np.mean(subarray),
                    }
                )

                features |= self.lbpm_manager.run_simulation(subarray)
                features_list.append(features)

        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving partial results...")

        return features_list

    def extract_features_train(self, n_samples, factor=10):
        if not Path("./data/features.csv").exists():
            feature_list = self._extract_features_train(n_samples, factor)
            df = pd.DataFrame(feature_list)
            df.to_csv("./data/features.csv", index=False)
        else:
            df = pd.read_csv("./data/features.csv")

        return df

    def _extract_features_inference(self, length=128):
        features_list = []

        for subarray, origin in self.sampler.sample(dims=length):
            features = {"subdomain_origin": origin}

            for axis, direction in self.directions.items():
                features |= self._compute_2d_features(subarray, axis, direction)

                volume, surface_area, _, eulerc_3d = mk.functionals(
                    subarray.astype(bool)
                )
                features.update(
                    {
                        "specific_surface_area": (
                            surface_area / (volume ** (2 / 3)) if volume > 0 else np.nan
                        ),
                        "eulerc_3d": eulerc_3d,
                        "porosity_3d": np.mean(subarray),
                    }
                )

            features_list.append(features)

        return features_list

    def extract_features_inference(self, length=128):
        if not Path("./data/inference_features.csv").exists():
            feature_list = self._extract_features_inference(length)
            df = pd.DataFrame(feature_list)
            df.to_csv("./data/inference_features.csv", index=False)
        else:
            df = pd.read_csv("./data/inference_features.csv")

        return df
