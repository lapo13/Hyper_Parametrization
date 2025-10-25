import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers, optimizers # type: ignore
from typing import Dict, List, Optional
from tensorflow.keras.callbacks import EarlyStopping # type: ignore


def progressive_penalty_loss_tf(y_true, y_pred, alpha=2.5, beta=1.4, penalty_weight=0.5):
    """
    Loss con penalità progressiva
    """
    # Errori per ogni ora
    hourly_errors = tf.abs(y_pred - y_true)  # [batch, 24]
    
    # Componenti base
    mean_errors = tf.reduce_mean(hourly_errors, axis=1)  # [batch]
    std_errors = tf.math.reduce_std(hourly_errors, axis=1)  # [batch]
    
    # Penalità progressiva
    max_errors = tf.reduce_max(hourly_errors, axis=1)  # [batch]
    penalty = tf.square(tf.nn.relu(max_errors - beta*mean_errors))
    
    # Score finale per ogni sample
    sample_losses = mean_errors + alpha * std_errors + penalty_weight * penalty
    
    # IMPORTANTE: ritorna la media finale per avere uno scalare
    return tf.reduce_mean(sample_losses)


class TFModel:
    """Wrapper addestramento/predizione con TensorFlow/Keras + early stopping dinamico"""
    def __init__(self, params: Dict):
        self.params = params
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        self.model: keras.Model | None = None
        self.trained = False
        self.history = []

    def build(self, input_dim: int, output_dim: int = 1):
        """Costruzione dinamica della rete"""
        hidden_units: List[int] = self.params.get("architecture", [])
        activation = self.params.get("activation", "relu").lower()
        if activation == "none":
            activation = None
        dropout: List[float] = (self.params.get("dropout", [0.0]))

        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        if hidden_units == []:
            raise ValueError("La lista dei neuroni è vuota! Controlla la configurazione.")

        for i, units in enumerate(hidden_units):
            x = layers.Dense(units, activation=activation)(x)
            if i < len(dropout):
                x = layers.Dropout(dropout[i])(x)

        outputs = layers.Dense(output_dim, activation='linear')(x)
        self.model = keras.Model(inputs, outputs)

    def get_weights(self):
        if self.model:
            return self.model.get_weights()
        
    def set_weights(self, weights):
        if self.model:
            return self.model.set_weights(weights) 
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, split_val: float = 0.2):
        assert self.model is not None, "Call build() first"

        # Parametri
        batch_size = int(self.params.get("batch_size", 32))
        lr = float(self.params.get("learning_rate", 1e-4))
        opt_name = str(self.params.get("optimizer", "adam")).lower()
        weight_decay = float(self.params.get("weight_decay", 0.0))
        epochs = int(self.params.get("epochs", 50))

        # Ottimizzatori
        if opt_name == "sgd":
            optimizer = optimizers.SGD(learning_rate=lr, momentum=weight_decay)
        elif opt_name == "adamw":
            optimizer = optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        else:
            optimizer = optimizers.Adam(learning_rate=lr)


        self.model.compile(optimizer=optimizer, loss= 'mae', metrics=["mse"])

        with tf.device(self.device):
            # Callback early stopping
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=self.params.get("early_stop_patience", 0),      # numero di epoche senza miglioramenti prima di fermarsi
                restore_best_weights=True,  # ripristina i pesi migliori
                verbose=1
            )

            if X_val is None or y_val is None:
                hist = self.model.fit(
                X_train, y_train,
                validation_split=split_val,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                shuffle=True,
                callbacks=[early_stop])
            # Training
            else:
                hist = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    shuffle=True,
                    callbacks=[early_stop])
            

            # Accesso alle loss
            losses = hist.history["loss"]

        self.trained = True
        return self.history

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.trained, "Model not trained"
        with tf.device(self.device):
            preds = self.model.predict(X_test, verbose=0)
        return preds


class TFModelInstantiator:
    @staticmethod
    def create_model(params: Dict, input_dim: int = 42, output_dim: int = 24) -> TFModel:
        m = TFModel(params)
        m.build(input_dim, output_dim)
        return m

