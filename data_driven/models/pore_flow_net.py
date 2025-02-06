import torch
import torch.nn as nn


class PoreFlowNet(nn.Module):
    def __init__(self, input_dim, constraint_type="kozeny_carman"):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.constraint_type = constraint_type

    def forward(
        self, x, porosity, specific_surface_area=None, return_uncertainty=False
    ):
        # Direct prediction
        k_pred = self.feature_net(x)

        # Physics-based prediction based on selected constraint
        k_physics = self._get_physics_prediction(porosity, specific_surface_area)

        # Base prediction
        prediction = k_pred * torch.sigmoid(k_physics)

        if not return_uncertainty:
            return prediction

        # Base model doesn't provide uncertainty, return zero uncertainty
        return {
            "mean": prediction,
            "variance": torch.zeros_like(prediction),
            "samples": prediction.unsqueeze(0),
        }

    def _get_physics_prediction(self, porosity, specific_surface_area=None):
        if self.constraint_type == "kozeny_carman":
            return self._kozeny_carman(porosity)
        elif self.constraint_type == "katz_thompson":
            return self._katz_thompson(porosity, specific_surface_area)
        elif self.constraint_type == "carman_maloney":
            return self._carman_maloney(porosity, specific_surface_area)
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")

    def _kozeny_carman(self, porosity):
        epsilon = 1e-6  # Prevent division by zero
        return (porosity**3) / ((1 - porosity + epsilon) ** 2)

    def _katz_thompson(self, porosity, specific_surface_area):
        if specific_surface_area is None:
            return self._kozeny_carman(porosity)
        epsilon = 1e-6
        lc = 1 / (specific_surface_area + epsilon)
        return (lc**2) * porosity

    def _carman_maloney(self, porosity, specific_surface_area):
        if specific_surface_area is None:
            return self._kozeny_carman(porosity)
        epsilon = 1e-6
        tortuosity = 1 / (porosity**0.5 + epsilon)
        return porosity**3 / (tortuosity * (specific_surface_area + epsilon) ** 2)
