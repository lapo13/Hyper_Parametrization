import numpy as np
import time

from typing import List, Dict, Tuple

#Le eccezioni sono state gestite per evitare crash in casi in cui il modello produce NaN o altre anomalie,
#la gestione assegna valori pessimi alle metriche in modo che tali modelli vengano penalizzati nella selezione evolutiva.

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.backend import clear_session # type: ignore
from Model_tf import TFModelInstantiator

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average"))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    r2 = r2_score(y_true, y_pred, multioutput="variance_weighted")
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape), "R2": float(r2)}


class CrossValuation:
    def __init__(self, model_instantiator, X: np.ndarray, y: np.ndarray, iter: int = 3, k: int = 3, seed: int = 42):
        assert y.shape[1] == 24, "X shape (N, D), y shape (N,)"
        self.inst = model_instantiator
        self.X, self.y = X.astype(np.float32), y.astype(np.float32)
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        self.iter = iter

    def evaluate(self, params_list: List[Dict], gen: int, higher_bound: int = 40, lower_bound: int = 8) -> List[Tuple[Dict, Dict, Dict]]:
        n_models = len(params_list)
        
        # Strategia di valutazione basata sulla dimensione della popolazione
        if n_models > higher_bound:
            # Popolazione grande: usa train/test split semplice
            evaluation_method = self._evaluate_train_test_split
            print(f"Gen {gen}: {n_models} modelli - usando Train/Test Split")
        elif n_models > lower_bound:
            # Popolazione media: usa K-Fold classico
            evaluation_method = self._evaluate_kfold
            print(f"Gen {gen}: {n_models} modelli - usando K-Fold classico")
        else:
            # Popolazione piccola: usa K-Fold iterativo per maggiore accuratezza
            evaluation_method = self._evaluate_iterative_kfold
            print(f"Gen {gen}: {n_models} modelli - usando K-Fold iterativo")
        
        return evaluation_method(params_list, gen)

    def _evaluate_train_test_split(self, params_list: List[Dict], gen: int) -> List[Tuple[Dict, Dict, Dict]]:
        """Valutazione con semplice train/test split per popolazioni grandi"""
        results = []
        
        # Prepara i dati una volta sola
        X, y = self.X, self.y
        
        # Crea un singolo split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        for i, params in enumerate(params_list):
            print(f"Modello {i+1}/{len(params_list)} - Train/Test Split")
            
            model = self.inst.create_model(params, input_dim=self.input_dim, output_dim=self.output_dim)
            try:
                start_time = time.perf_counter()
                model.train(X_train, y_train, X_test, y_test)
                end_time = time.perf_counter()
                
                preds = model.predict(X_test)
                train_time = end_time - start_time
                
                metrics = regression_metrics(y_test, preds)
                metrics["Train_Time"] = train_time
            except Exception as e:
                print(f"Errore durante l'addestramento o la previsione del modello {i+1}: {e}")
                metrics = {"MAE": float('inf'), "RMSE": float('inf'), "MAPE": float('inf'), "R2": float('-inf'), "Train_Time": float('inf')}
            # Per train/test split, STD = 0 per tutte le metriche
            std_metrics = {f"{k}_STD": 0.0 for k in metrics.keys()}
            
            results.append((params, metrics, std_metrics))
            
            if isinstance(self.inst, TFModelInstantiator):
                clear_session()
        
        return results

    def _evaluate_kfold(self, params_list: List[Dict], gen: int) -> List[Tuple[Dict, Dict, Dict]]:
        """Valutazione con K-Fold classico per popolazioni medie"""
        results = []
        
        # Prepara stratified k-fold
        std_scores = np.std(self.y, axis=1)
        std_scores_normalized = (std_scores - np.min(std_scores)) / (np.max(std_scores) - np.min(std_scores))
        bins = np.quantile(std_scores_normalized, [0.2, 0.4, 0.6, 0.8])
        std_bins = np.digitize(std_scores, bins)
        
        kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        
        for i, params in enumerate(params_list):
            print(f"Modello {i+1}/{len(params_list)} - K-Fold classico")
            fold_metrics = []
            
            model = self.inst.create_model(params, input_dim=self.input_dim, output_dim=self.output_dim)
            initial_weights = model.get_weights()
            
            for (tr_idx, te_idx) in kf.split(self.X, std_bins):
                Xtr, Xte = self.X[tr_idx], self.X[te_idx]
                ytr, yte = self.y[tr_idx], self.y[te_idx]
                
                try:
                    start_time = time.perf_counter()
                    model.train(Xtr, ytr, Xte, yte)
                    end_time = time.perf_counter()
                    
                    preds = model.predict(Xte)
                    train_time = end_time - start_time
                    
                    metrics = regression_metrics(yte, preds)
                except Exception as e:
                    print(f"Errore durante l'addestramento o la previsione del modello {i+1} nel fold: {e}")
                    metrics = {"MAE": float('inf'), "RMSE": float('inf'), "MAPE": float('inf'), "R2": float('-inf')}
                    train_time = float('inf')
                metrics["Train_Time"] = train_time
                fold_metrics.append(metrics)
                
                # Reset weights per il prossimo fold
                if isinstance(self.inst, TFModelInstantiator):
                    model.set_weights(initial_weights)
            
            # Calcola medie e deviazioni standard
            keys = fold_metrics[0].keys()
            avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
            std = {f"{k}_STD": float(np.std([m[k] for m in fold_metrics])) for k in keys}
            avg.update(std)
            
            results.append((params, avg, std))
            
            if isinstance(self.inst, TFModelInstantiator):
                clear_session()
        
        return results

    def _evaluate_iterative_kfold(self, params_list: List[Dict], gen: int) -> List[Tuple[Dict, Dict, Dict]]:
        """Valutazione con K-Fold iterativo per popolazioni piccole (massima accuratezza)"""
        results = []
        
        # Prepara dati per stratified k-fold
        std_scores = np.std(self.y, axis=1)
        std_scores_normalized = (std_scores - np.min(std_scores)) / (np.max(std_scores) - np.min(std_scores))
        bins = np.quantile(std_scores_normalized, [0.2, 0.4, 0.6, 0.8])
        std_bins = np.digitize(std_scores, bins)
        
        for i, params in enumerate(params_list):
            print(f"Modello {i+1}/{len(params_list)} - K-Fold iterativo")
            fold_metrics = []
            
            model = self.inst.create_model(params, input_dim=self.input_dim, output_dim=self.output_dim)
            initial_weights = model.get_weights()
            
            # K-Fold iterativo (come nella versione originale)
            for iteration in range(self.iter):
                kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42 + iteration)
                
                for tr_idx, te_idx in kf.split(self.X, std_bins):
                    Xtr, Xte = self.X[tr_idx], self.X[te_idx]
                    ytr, yte = self.y[tr_idx], self.y[te_idx]
                    
                    try:
                        start_time = time.perf_counter()
                        model.train(Xtr, ytr, Xte, yte)
                        end_time = time.perf_counter()
                        
                        preds = model.predict(Xte)
                        train_time = end_time - start_time
                        
                        metrics = regression_metrics(yte, preds)
                        metrics["Train_Time"] = train_time
                    except Exception as e:
                        print(f"Errore durante l'addestramento o la previsione del modello {i+1} nel fold iterativo: {e}")
                        metrics = {"MAE": float('inf'), "RMSE": float('inf'), "MAPE": float('inf'), "R2": float('-inf'), "Train_Time": float('inf')}
                    fold_metrics.append(metrics)
                    
                    if isinstance(self.inst, TFModelInstantiator):
                        model.set_weights(initial_weights)
            
            # Calcola medie e deviazioni standard
            keys = fold_metrics[0].keys()
            avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
            std = {f"{k}_STD": float(np.std([m[k] for m in fold_metrics])) for k in keys}
            avg.update(std)
            
            results.append((params, avg, std))
            
            if isinstance(self.inst, TFModelInstantiator):
                clear_session()
        
        return results