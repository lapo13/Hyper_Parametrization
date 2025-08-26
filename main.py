import pandas as pd
import os
import re
import ast
import numpy as np
import time
import matplotlib.pyplot as plt

from Hyper_Space import HyperparameterSpace
#from Model_th import THModelInstantiator
from Model_tf import TFModelInstantiator
from valuation import CrossValuation, regression_metrics
from Evolution import Evolution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def encode_timestamp_minute_cyclic(ts, prefix):
        time_in_minutes = ts.dt.hour * 60 + ts.dt.minute + ts.dt.second / 60
        angle = 2 * np.pi * time_in_minutes / period_minutes
        return {
            f"{prefix}_sin": np.sin(angle),
            f"{prefix}_cos": np.cos(angle)
        }

def series_to_2d_array(series, output_dim=24):
    arr = np.stack(series.apply(lambda x: np.array(x, dtype=np.float32)))
    if arr.shape[1] != output_dim:
        raise ValueError(f"Ogni elemento deve avere {output_dim} valori, trovato {arr.shape[1]}")
    return arr

if __name__ == "__main__":
    # Dati METRO
    path = "./data"
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]

    selected_file = []
    var = "averageSpeed"
    for file in files:
        if var in file:
            selected_file.append(file)
    #print(f"Selected file: {selected_file}")

    # Caricamento dati
    selezionati = [p for p in selected_file if re.search(r'METRO.+desc\.csv', p)]
    print (selezionati)
    dati = [pd.read_csv(path+'/'+p) for p in selezionati]
    dtset_completo = pd.concat(dati, ignore_index=True)
    #print(dtset_completo.shape)

    # Preprocessing
    dtset_completo['TTT'] = dtset_completo['TTT'].apply(ast.literal_eval)
        # Filtra le righe dove la colonna TTT ha lunghezza 24
    dtset_filtrato = dtset_completo[dtset_completo["TTT"].apply(lambda x: len(x) == 24)].sample(frac=1).reset_index(drop=True)

    cols_with_nan = dtset_filtrato.columns[dtset_filtrato.isna().any()]
    dtset_filtrato = dtset_filtrato.drop(columns=cols_with_nan)

    categorical = [
    'serviceType', 'typeLabel',
    'subnature', 'nature', 'is_weekend',
    'is_festive', 'day', 'month', 'type_of_TTT', 'metric', 'geometry_type', 'highway'
    ]
    #consiglio, per serie cicliche usare funzioni periodiche tipo seno coseno 
    df_encoded = pd.get_dummies(dtset_filtrato
                                , columns=categorical)

    to_drop = [
        'address',
        'brokerName', 'organization',
        'linkDBpedia', 'serviceUri', 'photoOrigs',
        'photoThumbs', 'photos', 'model', 'producer',
        'name', 'format', 'comments', 'protocol'
    ]

    df_clean = df_encoded.drop(columns=to_drop)

    df_clean.head()

    # Calcolo della differenza in minuti dall’inizio del giorno
    period_minutes = 1440

    df_clean["interval_start"] = pd.to_datetime(df_clean["interval_start"])
    df_clean["interval_end"] = pd.to_datetime(df_clean["interval_end"])

    # Aggiungo codifica per `interval_start`
    df_final = df_clean.assign(**encode_timestamp_minute_cyclic(df_clean["interval_start"], "start_time"))

    # Aggiungo codifica per `interval_end`
    df_final = df_clean.assign(**encode_timestamp_minute_cyclic(df_clean["interval_end"], "end_time"))

    df_final["duration_minutes"] = (df_final["interval_end"] - df_final["interval_start"]).dt.total_seconds() / 60

    df_final = df_final.drop(columns=["interval_start", "interval_end"])

    df_final.head()

    # Separazione delle feature e del target
    X = df_final.drop(columns=['TTT'])
    y = df_final['TTT']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print(f"Train size: {X_train.shape}, {y_train.shape}")
    print(f"Test size: {X_test.shape}, {y_test.shape}")

    # Convert X data
    X_train_np = X_train.to_numpy().astype(np.float32)
    X_test_np = X_test.to_numpy().astype(np.float32)

    # Convert y data
    y_train_np = series_to_2d_array(y_train, output_dim=24)
    y_test_np  = series_to_2d_array(y_test, output_dim=24)

    # Debug prints
    print(f"Final shapes:")
    print(f"X_train: {X_train_np.shape}")
    print(f"y_train: {y_train_np.shape}")

    # Scaler per input
    scaler_X = StandardScaler()
    X_train_np = scaler_X.fit_transform(X_train.to_numpy().astype(np.float32))
    X_test_np  = scaler_X.transform(X_test.to_numpy().astype(np.float32))

    # Scaler per output (se serve normalizzare anche y)
    scaler_y = StandardScaler()
    y_train_np = scaler_y.fit_transform(series_to_2d_array(y_train, output_dim=24))
    y_test_np  = scaler_y.transform(series_to_2d_array(y_test, output_dim=24))

    # Debug prints
    print("After normalization:")
    print(f"X_train: {X_train_np.shape}, mean={X_train_np.mean():.4f}, std={X_train_np.std():.4f}")
    

    # Componenti
    space = HyperparameterSpace()
    factory = TFModelInstantiator()
    valuation = CrossValuation(factory, X_train_np, y_train_np, k=4)
    evo = Evolution(space, elitism=1, tournament_size=3, crossover_type="uniform", base_mutation_rate=0.35)

    # Popolazione iniziale
    pop_size = 64
    population = [space.sample() for _ in range(pop_size)]
    pop_decrease = 1

    # Ciclo evolutivo 
    generations = 16
    for gen in range(generations):
        print(f"------------------------------generation n: {gen}-----------------------------------\n")
        results = valuation.evaluate(population)
        if evo.early_stop_criteria():
            break
        population = evo.evolve(results, pop_size=pop_size, max_generations=generations)
        time.sleep(3)
        if ((gen)%(pop_decrease)) == 0:
            pop_size //= 2
            pop_decrease *= 2 
            print(f"---------------------population decreased at {pop_size}, next decrease in {pop_decrease - gen}")

    print("---------------------------Done.---------------------------------")

    best_of_all = lambda lst: max(lst, key=lambda x: x[1])[0]

    print(f"Params for best performance: {best_of_all(evo.history)}")

    print(X_train_np.shape[1] == X_test_np.shape[1])
    print(y_train_np.shape[1] == y_test_np.shape[1])

    # Crea il modello
    input_dim = X_train_np.shape[1]
    output_dim = y_train_np.shape[1]
    model = factory.create_model(best_of_all(evo.history), input_dim, output_dim)

    # Allena il modello
    model.train(X_train_np, y_train_np)

    # Predizioni
    y_pred = model.predict(X_test_np)
    #print(y_pred)
    #print(y_test)

    y_pred_np = np.array(y_pred).astype(np.float32)

    y_test_np = scaler_y.inverse_transform(y_test_np)
    y_pred_np = scaler_y.inverse_transform(y_pred_np)

    # controllo dimensioni
    print("y_test shape:", y_test_np.shape)
    print("y_pred shape:", y_pred_np.shape)
    y_test_flat = y_test_np.flatten()
    y_pred_flat = y_pred_np.flatten()
    

    plt.figure(figsize=(6,6))
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.5)
    plt.plot([y_test_flat.min(), y_test_flat.max()],
            [y_test_flat.min(), y_test_flat.max()],
            color='red', linewidth=2)  # linea y=x
    plt.xlabel('Valori reali')
    plt.ylabel('Predizioni')
    plt.title('Predizioni vs Valori reali')
    plt.show()
    # Valutazione finale
    score = regression_metrics(y_true=y_test_np, y_pred=y_pred_np)
    print("Final evaluation metrics:", score)

    