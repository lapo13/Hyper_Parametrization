import pandas as pd
import os, re, ast, time
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from Hyper_Space import HyperparameterSpace
#from Model_th import THModelInstantiator
from Model_tf import TFModelInstantiator
from valuation import CrossValuation, regression_metrics
from Evolution import Evolution
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def normalize_metrics(
                metrics_mean: Dict[str, float],
                weights: Optional[Dict[str, float]] = None
            ) -> float:
        
        if weights is None:
            weights = {
                "R2": 0.3,
                "MAE": 0.2,
                "RMSE": 0.2,
                "MAPE": 0.3,
            }

        def _inv_cost(x: float) -> float:
            return 1.0 / (1.0 + x)

        performance_score = 0.0
        for key, w in weights.items():
            if key not in metrics_mean:
                raise KeyError(f"Metrica media mancante: '{key}'")

            v = float(metrics_mean[key])

            if key == "R2":
                contrib = w * max(0.0, v)
            
            else: # Errori (RMSE, MAPE)
                contrib = w * _inv_cost(v)

            performance_score += contrib

        return float(performance_score)

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
                        df[column] = df[column].interpolate(method='linear')
                    else:
                        # Se molti missing, usa la media
                        df[column] = df[column].fillna(df[column].mean())
                
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

def spezza_serie_in_colonne(df, colonna_stringa, prefix='col_'):
# Verifica che la colonna esista
    if colonna_stringa not in df.columns:
        raise ValueError(f"La colonna '{colonna_stringa}' non esiste nel DataFrame")
    
    # Crea una copia del DataFrame
    result_df = df.copy()
    
    # Rimuove la colonna originale
    colonna_originale = result_df.pop(colonna_stringa)
    
    # Lista per raccogliere tutti gli array convertiti
    arrays_convertiti = []
    
    # Converte ogni stringa in array numerico
    for stringa in colonna_originale:
        # Gestione di valori NaN o stringhe vuote
        if pd.isna(stringa) or stringa == '':
            arrays_convertiti.append([])
            continue
            
        # Rimuove le parentesi quadre se presenti
        stringa_pulita = str(stringa).strip('[]')
        
        # Divide per spazi multipli e filtra stringhe vuote
        valori = [x for x in stringa_pulita.split() if x]
        
        # Converte in float
        try:
            array_numerico = [float(x) for x in valori]
            arrays_convertiti.append(array_numerico)
        except ValueError as e:
            print(f"Errore nella conversione di '{stringa}': {e}")
            arrays_convertiti.append([])
    
    # Trova la lunghezza massima per determinare il numero di colonne
    lunghezze = [len(arr) for arr in arrays_convertiti]
    
    if not lunghezze:
        raise ValueError("Nessun dato valido trovato nella colonna")
    
    lunghezza_max = max(lunghezze)
    lunghezza_min = min(lunghezze)
    
    # Verifica che tutti gli array abbiano la stessa lunghezza
    if lunghezza_max != lunghezza_min:
        print(f"Attenzione: array di lunghezze diverse (min: {lunghezza_min}, max: {lunghezza_max})")
        print("I valori mancanti verranno riempiti con NaN")
        
        # Riempie con NaN gli array più corti
        arrays_completi = []
        for arr in arrays_convertiti:
            if len(arr) < lunghezza_max:
                arr_completo = arr + [np.nan] * (lunghezza_max - len(arr))
                arrays_completi.append(arr_completo)
            else:
                arrays_completi.append(arr)
    else:
        arrays_completi = arrays_convertiti
    
    # Crea il DataFrame con le nuove colonne
    nuove_colonne = pd.DataFrame(
        arrays_completi,
        index=result_df.index,
        columns=[f"{prefix}{i}" for i in range(lunghezza_max)]
    )
    
    # Combina con il DataFrame originale
    result_df = pd.concat([result_df, nuove_colonne], axis=1)
    
    return result_df

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

def is_useful_series(series, zero_ratio_thr=0.85, min_nonzero_value=0.1):
    arr = np.array(series, dtype=float)
    
    # Serie troppo corta
    if len(arr) <= 1:
        return False
    
    # Troppi zeri
    zero_ratio = (arr == 0).mean()
    if zero_ratio >= zero_ratio_thr:
        return False
    
    # Valore medio trascurabile (quasi zero)
    mean_val = np.mean(arr)
    if abs(mean_val) < min_nonzero_value:
        return False
    
    # Se arriva qui, la serie ha valori significativi
    return True

if __name__ == "__main__":
    to_keep = ['type_of_TTT', 'min', 'var', 'median', 'interval_end', 
            'is_festive', 'mean', 'max', 'interval_start', 'linear_trend',
            'is_weekend', 'EMBEDDING','month', 'avg_variation', 
            'TTT','day'
            ]

    path = "../data"

    # Caricamento dati
    dati = [pd.read_excel(path+'/'+'METRO_with_embeddings.xlsx', usecols=to_keep)]


    # Interpolazione intelligente†
    smart_interpolation(dati)

    # Concatenazione
    df_completo = pd.concat(dati, ignore_index=True)

    # Filtra righe che non contengono "nan"
    mask = ~df_completo['TTT'].str.contains("nan", na=False)
    df_completo = df_completo[mask]

    

    # Preprocessing
    df_completo['TTT'] = df_completo['TTT'].apply(ast.literal_eval)
    df_completo = spezza_serie_in_colonne(df_completo, 'EMBEDDING', prefix='emb_')

    # Filtra le righe dove la colonna TTT ha lunghezza 24
    dtset_filtrato = df_completo[df_completo["TTT"].apply(lambda x: len(x) == 24)].sample(frac=1).reset_index(drop=True)
    dtset_filtrato = dtset_filtrato[dtset_filtrato["type_of_TTT"] == "daily"]
    dtset_filtrato = dtset_filtrato.drop(columns=["type_of_TTT"])

    
    mask_useful = dtset_filtrato["TTT"].apply(is_useful_series)
    df_useful = dtset_filtrato[mask_useful]

    categorical = [
    'is_weekend','is_festive'
    ]

    #print(f'useful columns: {df_useful.columns.tolist()}')
    #consiglio, per serie cicliche usare funzioni periodiche tipo seno coseno 
    df_encoded = pd.get_dummies(df_useful
                                , columns=categorical)

    df_final = encode_time_features(df_encoded)
    
    print(f"Final shape: {df_final.shape}")


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
        test_size=0.1,
        random_state=42,
        stratify=std_bins,
        shuffle=True
    )

    # Debug prints
    print(f"Train size: {X_train.shape}, {y_train.shape}")
    print(f"Test size: {X_test.shape}, {y_test.shape}")

    # Scaler per input
    scaler_X = RobustScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)

    # Scaler per output
    scaler_y = RobustScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test  = scaler_y.transform(y_test)

    print(f"y_test: {y_train.shape}, y_train: {y_test.shape}")
    
    # Debug prints
    print("After normalization:")
    print(f"X_train: {X_train.shape}, mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    

    # Componenti
    space = HyperparameterSpace()
    factory = TFModelInstantiator()
    valuation = CrossValuation(factory, X_train, y_train, iter= 3, k = 4)
    evo = Evolution(space, elitism=1, tournament_size=2, crossover_type="uniform", base_mutation_rate=0.25)

    # Popolazione iniziale
    pop_size = 256
    population = [space.sample() for _ in range(pop_size)]

    MIN_POP_SIZE = 4
    REDUCTION_FACTOR = 0.5  # riduzione 
    BASE_INTERVAL = 1
    COUNT = 1

    # Ciclo evolutivo 
    generations = 9
    for gen in range(generations):
        
        print(f"------------------------------generation n: {gen}-----------------------------------\n")
        if (COUNT % BASE_INTERVAL == 0) and (gen != 0) and (pop_size > MIN_POP_SIZE):
            new_pop_size = max(MIN_POP_SIZE, int(pop_size * REDUCTION_FACTOR))
            
            if new_pop_size < pop_size:
                pop_size = new_pop_size
                print(f"--------------------- Population decreased to {pop_size} at generation {gen}")

        results = valuation.evaluate(population, gen=gen, higher_bound=32, lower_bound=8)
        if evo.early_stop_criteria(patience=4, tolerance= 5e-3):
            break
        
        population = evo.evolve(results, pop_size=pop_size, max_generations=generations)
        COUNT += 1
        time.sleep(3) 

    print("---------------------------Done.---------------------------------")

    print(X_train.shape[1] == X_test.shape[1])
    print(y_train.shape[1] == y_test.shape[1])
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    preds = []
    # Crea il modello
    for record in evo.history:
        #print(f"Score: {record[1]}, Params: {record[0]}")
        model = factory.create_model(record[0], input_dim, output_dim)

        # Allena il modello
        model.train(X_train, y_train)

        # Predizioni
        y_pred = model.predict(X_test)
        preds.append(scaler_y.inverse_transform(y_pred))



    y_test_np = scaler_y.inverse_transform(y_test)

    scores = []
    for y_pred in preds:
        score = normalize_metrics(regression_metrics(y_test_np, y_pred))
        scores.append(score)

    best_score = max(scores)
    best_index = scores.index(best_score)
    print(f"Best model index: {best_index} with scores: {best_score} and params: {evo.history[best_index][0]}")

    # controllo dimensioni
    print("y_test shape:", y_test_np.shape)
    print("y_pred shape:", preds[best_index].shape)
    y_test_flat = y_test_np.flatten()
    y_pred_flat = preds[best_index].flatten()
    

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
    print("Final evaluation metrics:", scores[best_index])

