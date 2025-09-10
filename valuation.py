import numpy as np
import torch
import time

from typing import List, Dict, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.backend import clear_session # type: ignore
from Model_tf import TFModelInstantiator

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), 1e-2)))) * 100.0)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape), "R2": float(r2)}


class CrossValuation:
    def __init__(self, model_instantiator, X: np.ndarray, y: np.ndarray, iter: int = 3, k: int = 3, seed: int = 42):
        assert X.shape[1] == 42 and y.shape[1] == 24, "X shape (N, D), y shape (N,)"
        self.inst = model_instantiator
        self.X, self.y = X.astype(np.float32), y.astype(np.float32)
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        self.iter = iter

    def evaluate(self, params_list: List[Dict]) -> List[Tuple[Dict, Dict, Dict]]:
        results: List[Tuple[Dict, Dict, Dict]] = []
        n = 0 # counter for traking the number of model

        # X, y = dataset
        std_scores = np.std(self.y, axis=1)  # deviazione standard per ogni serie

        std_scores_normalized = (std_scores - np.min(std_scores)) / (np.max(std_scores) - np.min(std_scores))

        # Creo bin (ad esempio 5 gruppi)
        bins = np.quantile(std_scores_normalized, [0.2, 0.4, 0.6, 0.8])
        std_bins = np.digitize(std_scores, bins)

        for params in params_list:
            n += 1
            print(f"------------------------------------training and valuating model n: {n} \n")
            fold_metrics: List[Dict[str, float]] = []

            model = self.inst.create_model(params, input_dim=self.input_dim, output_dim=self.output_dim)
            initial_weights = model.get_weights()

            for _ in range(self.iter):
                
                # Stratified K-Fold
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                for tr_idx, te_idx in kf.split(self.X, std_bins):
                    Xtr, Xte = self.X[tr_idx], self.X[te_idx]
                    ytr, yte = self.y[tr_idx], self.y[te_idx]

                    #print(f"Training fold shapes - X: {Xtr.shape}, y: {ytr.shape}")

                    start_time = time.perf_counter()
                    model.train(Xtr, ytr, Xte, yte)
                    end_time = time.perf_counter()

                    preds = model.predict(Xte)

                    train_time = end_time - start_time if (end_time and start_time) else 0.0
                    
                    metrics = regression_metrics(yte, preds)
                    metrics["Train_Time"] = train_time
                    fold_metrics.append(metrics)

                    if isinstance(self.inst, TFModelInstantiator):
                        model.set_weights(initial_weights)

                    '''if isinstance(self.inst, THModelInstantiator):
                        # Mitiga OOM su GPU tra i fold
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache '''

            keys = fold_metrics[0].keys()
            avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
            std = {f"{k}_STD": float(np.std([m[k] for m in fold_metrics])) for k in keys}
            avg.update(std)
            if isinstance(self.inst, TFModelInstantiator):
                    clear_session()
            results.append((params, avg, std))

        return results