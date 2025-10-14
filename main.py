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

def smart_interpolation(df_list):
    for i, df in enumerate(df_list):
        print(f"Processing dataset {i+1}...")
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            if missing_count > 0:
                print(f"  - Column '{column}': {missing_count} missing values")
                
                # Colonne numeriche
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Se pochi missing values, interpolazione
                    if missing_count / len(df) < 0.1:  # meno del 10%
                        df[column].interpolate(method='linear', inplace=True)
                    else:
                        # Se molti missing, usa la media
                        df[column].fillna(df[column].mean(), inplace=True)
                
                # Colonne stringa/categoriche
                elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                    # Usa la moda (valore più frequente)
                    mode_values = df[column].mode()
                    if not mode_values.empty:
                        df[column].fillna(mode_values[0], inplace=True)
                    else:
                        df[column].fillna('Unknown', inplace=True)
                
                # Colonne datetime
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column].interpolate(method='linear', inplace=True)
        
        print(f"Completed dataset {i+1}")

def encode_time_features(df, start_col="interval_start", end_col="interval_end"):
    df = df.copy()
    
    # Step 1: Ensure datetime columns are in the correct format first
    df[start_col] = pd.to_datetime(df[start_col], utc=True, format="mixed")
    df[end_col] = pd.to_datetime(df[end_col], utc=True, format="mixed")

    # Step 2: Extract 'month' and 'day' after conversion
    df['month'] = df[start_col].dt.month
    df['day'] = df[start_col].dt.dayofweek # Note: dayofweek is typically used for cyclical weekly patterns
    
    # Step 3: Now you can create the cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/len(df['month'].unique()))
    df['month_cos'] = np.cos(2 * np.pi * df['month']/len(df['month'].unique()))
    df['day_sin'] = np.sin(2 * np.pi * df['day']/len(df['day'].unique()))
    df['day_cos'] = np.cos(2 * np.pi * df['day']/len(df['day'].unique()))


    # Ora del giorno (0-23) → ciclo giornaliero
    df["hour"] = df[start_col].dt.hour
    angle_day_start = 2 * np.pi * df["hour"] / 24
    df["start_time_sin"] = np.sin(angle_day_start)
    df["start_time_cos"] = np.cos(angle_day_start)

    df["hour_end"] = df[end_col].dt.hour
    angle_day_end = 2 * np.pi * df["hour_end"] / 24
    df["end_time_sin"] = np.sin(angle_day_end)
    df["end_time_cos"] = np.cos(angle_day_end)

    # Durata in minuti
    df["duration_minutes"] = (df[end_col] - df[start_col]).dt.total_seconds() / 60

    # Giorno della settimana (0=Mon..6=Sun) → ciclo settimanale
    dayofweek = df[start_col].dt.dayofweek
    angle_week = 2 * np.pi * dayofweek / 7
    df["dow_sin"] = np.sin(angle_week)
    df["dow_cos"] = np.cos(angle_week)

    # Giorno dell’anno (1-365) → ciclo annuale
    dayofyear = df[start_col].dt.dayofyear
    angle_year = 2 * np.pi * dayofyear / 365
    df["doy_sin"] = np.sin(angle_year)
    df["doy_cos"] = np.cos(angle_year)

    # Rimuovo colonne temporali e variabili ausiliarie
    df = df.drop(columns=[start_col, end_col, "hour", "hour_end", "day", "month"])
    return df

def series_to_2d_array(series, output_dim=24):
    arr = np.stack(series.apply(lambda x: np.array(x, dtype=np.float32)))
    if arr.shape[1] != output_dim:
        raise ValueError(f"Ogni elemento deve avere {output_dim} valori, trovato {arr.shape[1]}")
    return arr

def is_useful_series(series, zero_ratio_thr=0.85, var_thr=1e-1, unique_thr=3):
    arr = np.array(series, dtype=float)
    
    # Varianza
    if np.var(arr) <= var_thr:
        return False
    
    # Percentuale di zeri
    zero_ratio = (arr == 0).mean()
    if zero_ratio >= zero_ratio_thr:
        return False
    
    # Numero valori distinti
    if len(np.unique(arr)) <= unique_thr:
        return False
    
    return True

if __name__ == "__main__":
    to_keep = ['type_of_TTT', 'min', 'longitude', 'var', 'median', 'interval_end', 
            'is_festive', 'mean', 'max', 'metric', 'interval_start', 'linear_trend',
            'is_weekend', 'latitude', 'nature','month', 'avg_variation', 
            'TTT','day', 'address', 'city', 'province'
            ]

    path = "../data"
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.xlsx')]

    # Caricamento dati
    selezionati = [p for p in files if not re.search(r'df',p) and re.search(r'.xlsx',p)]
    #print(f"File selezionati: {selezionati}")
    dati = [pd.read_excel(path+'/'+p, usecols=to_keep) for p in selezionati]
    
    # Interpolazione intelligente
    smart_interpolation(dati)

    # Concatenazione
    df_completo = pd.concat(dati, ignore_index=True)
    #print(dtset_completo.shape)

    # Filtra righe che non contengono "nan"
    mask = ~df_completo['TTT'].str.contains("nan", na=False)
    df_completo = df_completo[mask]

    # Preprocessing
    df_completo['TTT'] = df_completo['TTT'].apply(ast.literal_eval)
    # Filtra le righe dove la colonna TTT ha lunghezza 24
    dtset_filtrato = df_completo[df_completo["TTT"].apply(lambda x: len(x) == 24)].sample(frac=1).reset_index(drop=True)
    dtset_filtrato = dtset_filtrato[dtset_filtrato["type_of_TTT"] == "daily"]
    dtset_filtrato = dtset_filtrato.drop(columns=["type_of_TTT"])


    mask_useful = dtset_filtrato["TTT"].apply(is_useful_series)
    df_useful = dtset_filtrato[mask_useful]

    categorical = [
    'nature', 'is_weekend',
    'is_festive', 'metric', 'address', 'city', 'province'
    ]

    #print(f'useful columns: {df_useful.columns.tolist()}')
    #consiglio, per serie cicliche usare funzioni periodiche tipo seno coseno 
    df_encoded = pd.get_dummies(df_useful
                                , columns=categorical)

    df_final = encode_time_features(df_encoded)
    


    # Separazione delle feature e del target
    X = df_final.drop(columns=['TTT'])
    print(X.head())
    y = df_final['TTT']


    X_np = X.to_numpy().astype(np.float32)
    y_np = series_to_2d_array(y, output_dim=24)

    # Calcolo deviazione standard per ogni serie
    std_scores = np.std(y_np, axis=1)
    #print(std_scores)
    std_scores_log = np.log1p(std_scores)  # log(1+x) per evitare problemi con valori vicini a 0

    std_scores_normalized = (std_scores_log - np.min(std_scores_log)) / (np.max(std_scores_log) - np.min(std_scores_log))


    # Creo bin basati sui quantili
    bins = np.array([0.0, 0.4, 0.8, 1.0])
    std_bins = np.digitize(std_scores_normalized, bins)
    #print(f"bins :", std_bins)


    # Passo anche std_bins per poter stratificare
    X_train, X_test, y_train, y_test= train_test_split(
        X_np, y_np,
        test_size=0.25,
        random_state=42,
        stratify=std_bins,
        shuffle=True
    )

    # Debug prints
    print(f"Train size: {X_train.shape}, {y_train.shape}")
    print(f"Test size: {X_test.shape}, {y_test.shape}")

    # Scaler per input
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)

    # Scaler per output (se serve normalizzare anche y)
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test  = scaler_y.transform(y_test)

    print(f"y_test: {y_train.shape}, y_train: {y_test.shape}")
    
    # Debug prints
    print("After normalization:")
    print(f"X_train: {X_train.shape}, mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    

    # Componenti
    space = HyperparameterSpace()
    factory = TFModelInstantiator()
    valuation = CrossValuation(factory, X_train, y_train, iter= 5, k = 3)
    evo = Evolution(space, elitism=1, tournament_size=2, crossover_type="uniform", base_mutation_rate=0.45)

    # Popolazione iniziale
    pop_size = 2
    population = [space.sample() for _ in range(pop_size)]
    pop_decrease = 3

    # Ciclo evolutivo 
    generations = 3
    for gen in range(generations):
        print(f"------------------------------generation n: {gen}-----------------------------------\n")
        results = valuation.evaluate(population)

        if evo.early_stop_criteria(patience=2, tolerance= 5e-3):
            break
        
        if (((gen)%(pop_decrease)) == 0) and (gen != 0) and (pop_size > 4):
            pop_size //= 2
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
    model.train(X_train, y_train)

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

    