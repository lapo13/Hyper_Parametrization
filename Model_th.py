from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

import numpy as np

class Model:
    """Wrapper addestramento/predizione con supporto GPU."""
    def __init__(self, params: Dict):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model: _TorchRegressor | None = None
        self.trained = False

    def build(self, input_dim: int, output_dim: int = 1):
        self.model = _TorchRegressor(input_dim, output_dim, self.params).to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)  # <-- qui tolto .view(-1,1)
        #print(f"Debug - X_t shape: {X_t.shape}, y_t shape: {y_t.shape}")

        # Ensure shapes match
        assert X_t.shape[0] == y_t.shape[0], f"Batch dimension mismatch: {X_t.shape[0]} vs {y_t.shape[0]}"
        
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    
    def _dataset_size(self, X: np.ndarray) -> int:
        return X.shape[0] if X.ndim > 1 else len(X)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        assert self.model is not None, "Call build() first"
        self.model.train()

        batch_size = int(self.params.get("batch_size", 32))
        lr = float(self.params.get("learning_rate", 1e-4))
        opt_name = str(self.params.get("optimizer", "adam")).lower()
        weight_decay = float(self.params.get("weight_decay", 0.0))
        clip_norm = float(self.params.get("clip_norm", 1.0))  # per gradient clipping

        loader = self._make_loader(X_train, y_train, batch_size)
        criterion = nn.L1Loss()

        # Scelta ottimizzatore
        if opt_name == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.params.get("momentum", 0.0),
                weight_decay=weight_decay
            )
        elif opt_name == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )

        best_loss = float("inf")
        early_stop_patience = int(self.params.get("early_stop_patience", 10))
        patience = 0

        for epoch in range(self.params.get("epochs", 0)):
            epoch_loss = 0.0

            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)

                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            # Media su tutto il dataset
            epoch_loss /= self._dataset_size(X_train)
            scheduler.step(epoch_loss)

            print(f"[Epoch {epoch+1}/{self.params.get('epochs')}] Loss: {epoch_loss:.6f}")

            # Early stopping with adaptive tolerance
            if best_loss > 0.5:
                tolerance = 0.01   
            elif best_loss > 0.1:
                tolerance = 0.002   
            else:
                tolerance = 0.001  
            
            if epoch_loss + tolerance < best_loss:
                print(f"  New best loss: {best_loss:.6f} (improved by {(best_loss - epoch_loss):.5f})")
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                print(f"  No improvement (need > {tolerance:.4f} reduction, patience: {patience}/{early_stop_patience})")
            
            if patience >= early_stop_patience:
                print(f"Early stopping triggered: no improvement > {tolerance:.4f} for {patience} epochs")
                print(f"Final best loss: {best_loss:.6f}")
                break

        self.trained = True


    @torch.no_grad()
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.trained, "Model not trained"
        self.model.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        preds = self.model(X_t).detach().cpu().numpy()
        return preds


class _TorchRegressor(nn.Module):
    """MLP per regressione costruito dinamicamente da params."""
    def __init__(self, input_dim: int, output_dim: int, params: Dict):
        super().__init__()
        layers = []
        in_features = input_dim
        hidden_units = params.get("architecture", [])
        num_layers = len(hidden_units)
        activation = params.get("activation", "relu").lower()
        dropout = params.get("dropout", 0.0)

        act = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "gelu": nn.GELU(),
            "None": None
        }.get(activation, nn.ReLU())


        for layer in range(num_layers):
            if hidden_units == []:
                raise ValueError("La lista dei neuroni è vuota! Controlla la configurazione.")
            layers.append(nn.Linear(in_features, hidden_units[layer]))
            if act is not None:
                layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_units[layer]
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class THModelInstantiator:
    @staticmethod
    def create_model(params: Dict, input_dim: int = 42, output_dim: int = 24) -> Model:
        m = Model(params)
        m.build(input_dim, output_dim)
        return m