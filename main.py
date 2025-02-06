from pathlib import Path

import numpy as np

from features.extractor import FeatureExtractor
from models.permeability_predictor import PermeabilityPredictor
from models.rock_type_classifier import RockTypeClassifier
from subarray_sampler import SubarraySampler

BASE_SAMPLE = "./data/samples/P01_vs8um_segmented_1488x1340x1200.raw"
INPUT_SAMPLE_SEGMENTED = (
    "./data/samples/IL-71_Segmented_8-bit-uint_vs100um_2961x512x512.raw"
)
INPUT_SAMPLE_POROSITY = (
    "./data/samples/IL-71_Sat-Dry_32-bit-float_vs100um_2961x512x512.raw"
)
OUTPUT_SAMPLE_PERMEABILITY = (
    "./data/outputs/IL-71_Permeability_32-bit-float_vs100um_2961x512x512.raw"
)


def clean_nans(data):
    data = data.dropna(axis=1, how="all")
    data.interpolate(method="polynomial", order=3, inplace=True)

    return data


def predict_permeability(inspect_model=False):
    train_extractor = FeatureExtractor(BASE_SAMPLE)
    df = train_extractor.extract_features_train(2000)
    df = clean_nans(df)
    X_train = df.drop(["permeability", "lbpm_porosity"], axis=1)

    inference_extractor = FeatureExtractor(
        INPUT_SAMPLE_SEGMENTED,
        invert=False,
    )
    df_inference = inference_extractor.extract_features_inference(128)
    df_inference = clean_nans(df_inference)
    X_inference = df_inference.drop(["subdomain_origin"], axis=1)

    common_columns = list(set(X_train.columns).intersection(set(X_inference.columns)))
    X_train = X_train[common_columns]
    X_inference = X_inference[common_columns]

    rt_classifier = RockTypeClassifier()
    optimal_k = rt_classifier.find_optimal_k(X_train)
    cluster_labels = rt_classifier.fit_predict(X_train, optimal_k)

    predictor = PermeabilityPredictor(
        batch_size=64, learning_rate=0.01, epochs=200, model_type="KozenyCarmanNN"
    )
    predictor.fit(df, cluster_labels)
    if inspect_model:
        predictor.plot_predictions(df, cluster_labels)

    X_inference = X_inference.fillna(X_inference.mean())
    inferred_labels = rt_classifier.predict(X_inference)
    X_inference["rock_type"] = inferred_labels
    X_inference["subdomain_origin"] = df_inference["subdomain_origin"]

    present_rock_types = X_inference["rock_type"].unique()  # pyright: ignore
    present_rock_types.sort()  # pyright: ignore

    porosities = SubarraySampler(
        INPUT_SAMPLE_POROSITY,
        invert=False,
        dtype=np.float32,  # pyright: ignore
    )
    permeabilities = np.zeros_like(porosities.array)

    for rt in present_rock_types:
        X_inference_rock = X_inference[X_inference["rock_type"] == rt]

        for _, row in X_inference_rock.iterrows():  # pyright: ignore
            subdomain_origin = [
                int(o) for o in str(row["subdomain_origin"]).strip("[]").split()
            ]
            porosity = porosities.array[
                subdomain_origin[0] : subdomain_origin[0] + 128,
                subdomain_origin[1] : subdomain_origin[1] + 128,
                subdomain_origin[2] : subdomain_origin[2] + 128,
            ]
            permeability = predictor.predict(porosity, rock_type=rt).reshape(
                128, 128, 128
            )
            permeabilities[
                subdomain_origin[0] : subdomain_origin[0] + 128,
                subdomain_origin[1] : subdomain_origin[1] + 128,
                subdomain_origin[2] : subdomain_origin[2] + 128,
            ] = permeability

    return permeabilities


def compute_equivalent_permeability(permeabilities):
    slice_means = np.mean(permeabilities, axis=(1, 2))
    slice_means = slice_means[slice_means > 0]

    epsilon = 1e-10
    return len(slice_means) / np.sum(1.0 / (slice_means + epsilon))


def main():
    if not Path(OUTPUT_SAMPLE_PERMEABILITY).exists():
        permeabilities = predict_permeability(inspect_model=True)
        permeabilities.tofile(OUTPUT_SAMPLE_PERMEABILITY)
    else:
        permeabilities = np.fromfile(OUTPUT_SAMPLE_PERMEABILITY, dtype=np.float32)
        permeabilities = permeabilities.reshape(2961, 512, 512)

    print(f"Equiv. permeability: {compute_equivalent_permeability(permeabilities)}")


if __name__ == "__main__":
    main()
