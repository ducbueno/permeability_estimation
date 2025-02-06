import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            # nn.Softplus(),
        )

    def forward(self, x):
        return self.model(x)


class KozenyCarmanNN(nn.Module):
    def __init__(self):
        super(KozenyCarmanNN, self).__init__()
        self.parameter_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softplus(),
        )

    def forward(self, x):
        params = self.parameter_net(x)
        A, B = params[:, 0:1], params[:, 1:2]

        x = x.clamp(min=1e-6, max=1 - 1e-6)
        ratio = x / (1 - x)

        return A * torch.pow(ratio, B)

    def get_parameters(self, x):
        with torch.no_grad():
            params = self.parameter_net(x)
            return params[:, 0], params[:, 1]


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PermeabilityPredictor:
    def __init__(
        self, batch_size=32, learning_rate=0.001, epochs=100, model_type="SimpleNN"
    ):
        """
        Initialize the PermeabilityPredictor.

        Args:
            batch_size (int): Size of batches for training
            learning_rate (float): Learning rate for optimization
            epochs (int): Number of training epochs
            model_type (str): Type of model to use ('SimpleNN' or 'KozenyCarmanNN')
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_type = model_type
        if model_type not in ["SimpleNN", "KozenyCarmanNN"]:
            raise ValueError("model_type must be either 'SimpleNN' or 'KozenyCarmanNN'")
        self.models = {}
        self.scaler_X = {}
        self.scaler_y = {}

    def _train_single_model(self, rock_type_df):
        """Train a model for a single rock type."""
        # Prepare data
        X = rock_type_df["porosity_3d"].values.reshape(-1, 1)
        y = rock_type_df["permeability"].values.reshape(-1, 1)

        # Scale the data
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # Split the data
        X_train, _, y_train, _ = train_test_split(X_scaled, y_scaled, test_size=0.2)

        # Create datasets and dataloaders
        train_dataset = CustomDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Initialize model based on model_type
        model = SimpleNN() if self.model_type == "SimpleNN" else KozenyCarmanNN()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Training loop
        model.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        return model, scaler_X, scaler_y

    def fit(self, df, cluster_labels):
        """
        Fit models for each rock type.

        Args:
            df (pd.DataFrame): DataFrame containing 'permeability' and 'porosity_3d'
            cluster_labels (array-like): Rock type labels for each sample
        """
        df_models = df[["permeability", "porosity_3d"]].copy()
        df_models.loc[:, "rock_type"] = cluster_labels

        # Create separate DataFrames for each rock type
        unique_rock_types = np.unique(cluster_labels)
        rock_type_dfs = {
            i: df_models[df_models["rock_type"] == i] for i in unique_rock_types
        }

        # Train a model for each rock type
        for rock_type, rock_df in rock_type_dfs.items():
            model, scaler_X, scaler_y = self._train_single_model(rock_df)
            self.models[rock_type] = model
            self.scaler_X[rock_type] = scaler_X
            self.scaler_y[rock_type] = scaler_y

    def predict(self, porosity, rock_type):
        """
        Predict permeability for given porosity and rock type.

        Args:
            porosity (array-like): Porosity values
            rock_type (int): Rock type label

        Returns:
            array: Predicted permeability values
        """
        if rock_type not in self.models:
            raise ValueError(f"No model trained for rock type {rock_type}")

        model = self.models[rock_type]
        scaler_X = self.scaler_X[rock_type]
        scaler_y = self.scaler_y[rock_type]

        X = np.array(porosity).reshape(-1, 1)
        X_scaled = scaler_X.transform(X)

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_scaled))
            y_pred = scaler_y.inverse_transform(y_pred.numpy())

        return y_pred

    def plot_predictions(self, df, cluster_labels):
        """
        Plot original data and predictions for each rock type.

        Args:
            df (pd.DataFrame): DataFrame containing 'permeability' and 'porosity_3d'
            cluster_labels (array-like): Rock type labels for each sample
        """
        df_models = df[["permeability", "porosity_3d"]].copy()
        df_models.loc[:, "rock_type"] = cluster_labels

        for rock_type in np.unique(cluster_labels):
            rock_df = df_models[df_models["rock_type"] == rock_type]
            X = np.array(rock_df["porosity_3d"]).reshape(-1, 1)
            y_pred = self.predict(X, rock_type)

            error = (
                np.mean(np.abs(y_pred.ravel() - rock_df["permeability"]))
                / rock_df["permeability"].max()
            )

            plt.figure(figsize=(10, 6))
            plt.scatter(X, rock_df["permeability"], label="Original")
            plt.scatter(X, y_pred, label="Predicted")
            plt.yscale("log")
            plt.xlabel("Porosity")
            plt.ylabel("Permeability")
            plt.legend()
            plt.title(f"Rock Type {rock_type}\nError = {error:.3f}")
            plt.show()
