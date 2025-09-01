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
from sklearn.preprocessing import MinMaxScaler

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


    X_np = X.to_numpy().astype(np.float32)
    y_np = series_to_2d_array(y, output_dim=24)

    # Calcolo deviazione standard per ogni serie
    std_scores = np.std(y_np, axis=1)

    # Creo bin basati sui quantili
    bins = np.quantile(std_scores, [0.2, 0.4, 0.6, 0.8])
    std_bins = np.digitize(std_scores, bins)

    # Primo split: train (80%) vs temp (20%)
    # Passo anche std_bins per poter stratificare
    X_train, X_temp, y_train, y_temp, bins_train_temp, bins_temp = train_test_split(
        X_np, y_np, std_bins,
        test_size=0.2,
        random_state=42,
        stratify=std_bins
    )

    # bins_temp è il sottoinsieme di std_bins corrispondente a X_temp/y_temp
    # Secondo split: divido temp in val/test (50%-50% del temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=bins_temp  # usa i bin corrispondenti al sottoinsieme
    )

    print(f"Train size: {X_train.shape}, {y_train.shape}")
    print(f"Validation size: {X_val.shape}, {y_val.shape}")
    print(f"Test size: {X_test.shape}, {y_test.shape}")

    # Debug prints
    print(f"Final shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_train: {y_train.shape}")

    # Scaler per input
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.fit_transform(X_val)
    X_test  = scaler_X.transform(X_test)

    # Scaler per output (se serve normalizzare anche y)
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.fit_transform(y_val)
    y_test  = scaler_y.transform(y_test)

    print(f"y_test: {y_train.shape}, y_val: {y_val.shape}, y_train: {y_test.shape}")
    
    # Debug prints
    print("After normalization:")
    print(f"X_train: {X_train.shape}, mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    

    # Componenti
    space = HyperparameterSpace()
    factory = TFModelInstantiator()
    valuation = CrossValuation(factory, X_train, y_train, k=4)
    evo = Evolution(space, elitism=1, tournament_size=4, crossover_type="uniform", base_mutation_rate=0.25)

    # Popolazione iniziale
    pop_size = 32
    population = [space.sample() for _ in range(pop_size)]
    pop_decrease = 1

    # Ciclo evolutivo 
    generations = 8
    for gen in range(generations):
        print(f"------------------------------generation n: {gen}-----------------------------------\n")
        results = valuation.evaluate(population)

        if evo.early_stop_criteria():
            break
        
        if ((gen)%(pop_decrease)) == 0:
            pop_size //= 2
            pop_decrease *= 2
            print(f"---------------------population decreased at {pop_size}, next decrease in {pop_decrease - gen}")

        population = evo.evolve(results, pop_size=pop_size, max_generations=generations)
        time.sleep(3) 


    print("---------------------------Done.---------------------------------")

    best_of_all = lambda lst: max(lst, key=lambda x: x[1])[0]

    print(f"Params for best performance: {best_of_all(evo.history)}")

    print(X_train.shape[1] == X_test.shape[1])
    print(y_train.shape[1] == y_test.shape[1])

    # Crea il modello
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = factory.create_model(best_of_all(evo.history), input_dim, output_dim)

    # Allena il modello
    model.train(X_train, y_train, X_val, y_val)

    # Predizioni
    y_pred = model.predict(X_test)
    #print(y_pred)
    #print(y_test)

    y_pred_np = np.array(y_pred).astype(np.float32)

    y_test_np = scaler_y.inverse_transform(y_test)
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

    