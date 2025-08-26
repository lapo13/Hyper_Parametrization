import random
import math
import numpy as np

from typing import List, Dict, Tuple, Optional
from Hyper_Space import HyperparameterSpace


class Evolution:
    def __init__(self, hyper_space: HyperparameterSpace, elitism: int = 2, tournament_size: int = 3, crossover_type: str = "uniform", base_mutation_rate: float = 0.2):
        self.hyper_space = hyper_space
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.crossover_type = crossover_type
        self.base_mutation_rate = base_mutation_rate
        self.generation = 0
        self.scores: List[Tuple[Dict, float]] = []
        self.history: List[Tuple[Dict, float]] = []  # best per gen

    # ---- normalizzazione metriche
    def _normalize_metrics(
                self,
                metrics_mean: Dict[str, float],
                metrics_std: Dict[str, float], # Deviazione standard delle metriche sui fold
                weights: Optional[Dict[str, float]] = None, 
                variance_penalty_weight: float = 0.5, # Peso per la penalità sulla stabilità
                max_time: Optional[float] = None,
                log_time: bool = True
            ) -> float:
        """
        Calcola uno score combinato che bilancia performance e stabilità.
        - R2: massimizzato.
        - RMSE, MAPE: minimizzati (trasformazione 1/(1+x)).
        - Train_Time: minimizzato.
        - Aggiunge una penalità basata sulla deviazione standard del RMSE tra i fold.
        """
        if weights is None:
            weights = {
                "R2": 0.4,
                "RMSE": 0.3,
                "MAPE": 0.2,
                "Train_Time": 0.1
            }

        def inv_cost(x: float) -> float:
            return 1.0 / (1.0 + x)

        performance_score = 0.0
        for key, w in weights.items():
            if key not in metrics_mean:
                raise KeyError(f"Metrica media mancante: '{key}'")

            v = float(metrics_mean[key])

            if key == "R2":
                contrib = w * max(0.0, v)
            
            elif key == "Train_Time":
                rel_t = (v / max_time) if (max_time is not None and max_time > 0) else v
                x = np.log1p(rel_t) if log_time else rel_t
                contrib = w * inv_cost(x)
            
            else: # Errori (RMSE, MAPE)
                contrib = w * inv_cost(v)

            performance_score += contrib

        stability_metric = "MAPE_STD" 
        if stability_metric not in metrics_std:
            raise KeyError(f"Metrica di deviazione standard mancante: '{stability_metric}'")
        
        mean_val = metrics_mean.get(stability_metric, 1.0)
        std_val = metrics_std.get(stability_metric, 0.0)
        
        # Aggiungiamo un epsilon per evitare divisione per zero se l'errore medio è 0
        coefficient_of_variation = std_val / (mean_val + 1e-6)
        
        # La penalità è proporzionale alla variabilità
        variance_penalty = variance_penalty_weight * coefficient_of_variation
        
        # Lo score finale è la performance MENO la penalità per l'instabilità
        final_score = performance_score - variance_penalty

        return float(final_score)
    
    # ---- selezione ----
    def _tournament_selection(self) -> dict:
        """
        Seleziona un individuo usando un torneo multi-metrica.
        La selezione considera tutte le metriche: R2 massimizzato, MAE/RMSE/MAPE minimizzati.
        """
        
        bucket = random.sample(self.scores, k=min(self.tournament_size, len(self.scores)))
        best_ind = max(bucket, key=lambda x: x[1])[0]

        return best_ind


    # ---- crossover ----
    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        if self.crossover_type == "uniform":
            return {k: random.choice([p1[k], p2[k]]) for k in p1.keys()}
        elif self.crossover_type == "single_point":
            keys = list(p1.keys())
            pt = random.randint(1, len(keys)-1)
            return {**{k: p1[k] for k in keys[:pt]}, **{k: p2[k] for k in keys[pt:]}}
        elif self.crossover_type == "multi_point":
            keys = list(p1.keys())
            points = sorted(random.sample(range(1, len(keys)), k=min(2, len(keys)-1)))
            child, toggle, prev = {}, True, 0
            for cut in points + [len(keys)]:
                src = p1 if toggle else p2
                for k in keys[prev:cut]:
                    child[k] = src[k]
                toggle, prev = (not toggle), cut
            return child
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")

    # ---- mutazione adattiva ----
    def _adaptive_mut_rate(self, max_generations: int) -> float:
        """
        Mutation rate adattivo:
        - Decay esponenziale più lento per poche generazioni
        - Restart ogni ~20 generazioni per massima esplorazione e boost
        - Floor alto per popolazione piccola
        """
        
        min_floor = 0.18
        restart_cycles = 2  # Solo 2 restart per 64 generazioni = cicli da 32
        decay_strength = 0.8  # Decay molto graduale
        restart_boost = 2.0  # Boost significativo per pop. piccola
        
        cycle_length = max_generations / restart_cycles
        cycle_position = (self.generation % cycle_length) / cycle_length
        
        # Decay esponenziale molto graduale
        exponential_decay = math.exp(-decay_strength * cycle_position)
        current_rate = self.base_mutation_rate * exponential_decay
        
        # Boost nelle prime 20% di ogni ciclo (primi ~6 generazioni del ciclo)
        if cycle_position < 0.2:
            current_rate *= restart_boost
        
        return max(min_floor, current_rate)
        

    # ---- evolve ----
    def evolve(self, results: List[Tuple[Dict, Dict, Dict]], pop_size: int, max_generations: int = 50) -> List[Dict]:
        self.generation += 1
        mut_rate = self._adaptive_mut_rate(max_generations)

        self.scores = [(r[0], self._normalize_metrics(r[1], r[2])) for r in results]

        # Elitismo usando lo score multi-metrica
        sorted_res = sorted(self.scores, key=lambda x: x[1], reverse=True)
        elite_params = [r[0] for r in sorted_res[:self.elitism]]

        # Nuova popolazione
        new_pop: List[Dict] = list(elite_params)

        while len(new_pop) < pop_size:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            child = self._crossover(p1, p2)
            child = self.hyper_space.mutate(child, mutation_rate=mut_rate)
            new_pop.append(child)

        self.history.append(sorted_res[0])
        return new_pop
    
    def early_stop_criteria(
        self, 
        patience: int = 10, 
        tolerance: float = 1e-3
    ) -> bool:
        """
        Determina se fermare l'evoluzione basandosi sulla stagnazione di una metrica specifica.

        Args:
            metric_name (str): La chiave della metrica da monitorare nel dizionario degli score
                               (es. 'accuracy', 'f1_score', 'val_loss').
            patience (int): Il numero di generazioni recenti da analizzare.
            tolerance (float): La soglia di deviazione standard sotto la quale si considera
                               la performance stagnante.

        Returns:
            bool: True se le condizioni di arresto sono soddisfatte, altrimenti False.
        """
        # 1. Controlla se abbiamo abbastanza dati storici per decidere
        if len(self.history) < patience:
            return False

        # 2. Estrai i punteggi della metrica di interesse dalle ultime 'patience' generazioni
        recent_tuples = self.scores[-patience:]
        recent_performance = [r[1] for r in recent_tuples]

        # 3. Calcola la deviazione standard di questa finestra di performance
        std_dev = np.std(recent_performance)

        # 4. Applica il criterio di arresto
        print(f"Generazione {len(self.history)} | Dev. Std. (ultime {patience} gen): {std_dev:.6f}")
        
        if std_dev < tolerance:
            print(f"--> Criterio di arresto raggiunto: Dev. Std. ({std_dev:.6f}) < Tolleranza ({tolerance})")
            return True
        
        return False
        