import random
import numpy as np

from typing import Dict, Tuple
from decimal import Decimal, ROUND_DOWN

class HyperparameterSpace:
    def __init__(self):
        self.seen: set[Tuple[Tuple[str, object], ...]] = set()
        self.space = {
            "learning_rate": ("continuous", 1e-4, 1e-1),  
            "architecture": ("categorical", [
                # Architetture leggere (1-2 layers)
                [32],
                [64],
                [128],
                [256],
                
                [32, 32],
                [64, 64],
                [128, 128],
                [256, 256],

                #aarchitetture medie (2 layers)
                [64, 32],
                [128, 64],
                [256, 128],

                # Architetture moderate (3 layers)
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128],
                
                # Architetture a "bottleneck"
                [256, 64, 256],
                [128, 32, 128],
                
                # Architetture crescenti
                [32, 64, 128],
                [64, 128, 256],

                #architetture a 4 layer "bottlenek"
                [256, 64, 64, 256],
                [128, 32, 32, 128]
            ]), 
            "activation": ("categorical", ["relu"]),
            "batch_size": ("discrete", [16,32,64,128]), 
            "dropout": ("discrete", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), 
            "optimizer": ("categorical", ["sgd"]),
            "weight_decay": ("continuous", 1e-5, 1e-2),
            "early_stop_patience": ("discrete", [10, 15, 20]), 
            "epochs": ("discrete", [300, 400, 500, 600])
            }

    def _hash_params(self, params: Dict) -> Tuple[Tuple[str, object], ...]:
        """Crea un hash immutabile dei parametri, convertendo eventuali liste in tuple."""
        hashable_params = {}
        for k, v in params.items():
            if isinstance(v, list):
                hashable_params[k] = tuple(v)
            else:
                hashable_params[k] = v
        return tuple(sorted(hashable_params.items()))

    def _sample_continuous_param(self, name, spec, params):
        """
        Campiona parametri continui con distribuzione appropriata.
        
        Args:
            name: Nome del parametro
            spec: Specifica del parametro (tipo, low, high)
            params: Dizionario dei parametri da aggiornare
        """
        ptype, lo, hi = spec
        
        if ptype == "continuous":
            # Parametri che beneficiano di distribuzione log-uniform
            log_params = {"learning_rate", "weight_decay"}
            
            if name in log_params:
                # Gestione speciale per weight_decay che può essere 0
                if name == "weight_decay" and lo == 0:
                    # 20% chance di essere esattamente 0 (no regularization)
                    if random.random() < 0.2:
                        params[name] = 0.0
                        return
                    # Altrimenti campiona da range log-uniform escludendo 0
                    lo = max(1e-8, 1e-6)  # Minimo non-zero
                
                # Sampling log-uniform
                log_lo = np.log10(max(lo, 1e-12))
                log_hi = np.log10(max(hi, 1e-12))
                val = 10 ** random.uniform(log_lo, log_hi)
                
            else:
                # Sampling uniforme per altri parametri continui
                val = random.uniform(lo, hi)
            
            # Precision control
            val = Decimal(val)
            params[name] = float(val.quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
        
        else:
            raise ValueError("parameter non continous")

    def sample(self):
        """Campiona un set completo di iperparametri"""
        params = {}
        
        for key in self.space:
            spec = self.space[key]
            ptype = spec[0]
            
            if ptype == "continuous":
                self._sample_continuous_param(key, spec, params)
                
            elif ptype == "discrete":
                params[key] = random.choice(spec[1])
                
            elif ptype == "categorical":
                params[key] = random.choice(spec[1])
        
        return params

    def mutate(self, params: Dict, mutation_rate: float = 0.2) -> Dict:
        newp = dict(params)
        for name, spec in self.space.items():
            if random.random() < mutation_rate:
                ptype = spec[0]

                if ptype == "continuous":
                    lo, hi = spec[1], spec[2]
                    factor = random.uniform(0.5, 1.5)
                    val = np.clip(newp[name] * factor, lo, hi)
                    val = Decimal(val)
                    newp[name] = float(val.quantize(Decimal('0.0001'), rounding=ROUND_DOWN))

                elif ptype == "discrete":
                        newp[name] = random.choice(spec[1])

                elif ptype == "categorical":
                    newp[name] = random.choice(spec[1])

        return newp
