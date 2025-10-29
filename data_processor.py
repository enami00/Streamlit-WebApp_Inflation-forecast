import pandas as pd
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog

# --- CONFIGURATION (Colonnes) ---
# Les colonnes cibles pour lesquelles nous allons calculer le Taux d'Inflation Annuel (YoY)
TARGET_COLUMNS = ['IPC', 'IPC - non food non energy'] 
OUTPUT_FILE = 'inflation_data_YoY_quarterly.csv' # Fichier de sortie

# --- FONCTIONS DE TRAITEMENT ---

def load_data_and_correct_decimals(file_path):
    """
    Charge le fichier CSV en utilisant la virgule (,) comme séparateur décimal.
    Nettoie les noms de colonnes et convertit les colonnes numériques en float.
    """
    print(f"Chargement et correction du fichier : {file_path}")
    
    try:
        # 1. Lecture avec séparateurs spécifiques (virgule décimale)
        df = pd.read_csv(
            file_path, 
            index_col='observation_date', 
            parse_dates=True,
            sep=',',      # Séparateur de colonnes
            decimal=',',   # SÉPARATEUR DÉCIMAL CLÉ
            engine='python' 
        )
        
        # 1.1 NETTOYAGE DES NOMS DE COLONNES (Suppression des espaces inutiles)
        df.columns = df.columns.str.strip()
        
        # 2. Conversion explicite de toutes les colonnes numériques en float
        for col in df.columns:
            try:
                # Nettoyage des chaînes et conversion en numérique. 'coerce' remplace les erreurs par NaN.
                # Supprime les espaces et les tabulations dans les nombres (ex: "1 234,5" en "1234.5" après correction décimale)
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', '').str.replace('\t', ''), errors='coerce')
            except Exception:
                pass # Ignorer les colonnes non numériques

        # 3. Nettoyage initial des NaN dans les colonnes cibles
        df = df.dropna(subset=TARGET_COLUMNS)
        
        print("\n--- TYPES DE DONNÉES APRES CORRECTION ---")
        print(df.dtypes)
        print(f"Lignes chargées après nettoyage initial : {len(df)}")
        
        if len(df) == 0:
            print("AVERTISSEMENT : Le DataFrame est vide après le chargement et le nettoyage initial des cibles.")

        return df
        
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{file_path}' est introuvable.")
        return None
    except Exception as e:
        print(f"ERREUR lors du chargement : {e}")
        return None


def calculate_yoy_inflation(df_original):
    """
    Calcule et remplace les taux d'inflation trimestriels par les Taux d'Inflation Annuels (YoY)
    pour toutes les colonnes spécifiées dans TARGET_COLUMNS.
    """
    
    if df_original.empty:
        print("ATTENTION: Le DataFrame initial est vide, impossible de procéder au calcul YoY.")
        return pd.DataFrame()

    # Copie pour travailler et éviter les warnings de chaînage
    df_processed = df_original.copy()
    
    for col in TARGET_COLUMNS:
        # 1. Préparation pour le calcul de l'indice cumulé
        # CORRECTION CLÉ: Division par 100 pour convertir le pourcentage (ex: 0.1976) en décimal (0.001976).
        # fillna(0) est utilisé pour permettre au cumprod() de continuer sans s'arrêter en cas de NaN.
        ipc_rates = df_processed[col].fillna(0) / 100
        
        # 2. Calcul de l'Indice des Prix à la Consommation (IPC) cumulé
        # L'indice commence à 1.0 (100) à la première observation
        df_ipc_idx = (1 + ipc_rates).cumprod() 
        
        # 3. Calcul du Taux d'Inflation Annuel (Year-on-Year, YoY)
        # Formule : (Indice_t / Indice_{t-4} - 1) * 100
        # On compare l'indice du trimestre actuel avec l'indice du même trimestre il y a 4 périodes (un an)
        new_col_name = f'Inflation_YoY - {col}'
        df_processed[new_col_name] = (df_ipc_idx / df_ipc_idx.shift(4) - 1) * 100
        
        # 4. Suppression de la colonne trimestrielle originale
        df_processed = df_processed.drop(columns=[col])

    # 5. Nettoyage des 4 premières lignes qui auront des NaN après le calcul YoY.
    # On applique dropna UNIQUEMENT aux nouvelles colonnes YoY.
    yoy_cols = [f'Inflation_YoY - {col}' for col in TARGET_COLUMNS]
    df_processed.dropna(subset=yoy_cols, inplace=True) 
    
    return df_processed

# --- FONCTION PRINCIPALE ---

def main():
    # Déterminer le fichier d'entrée (Utilisation du dialogue par défaut)
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # --- Utiliser un dialogue de sélection de fichier (file.choose équivalent) ---
        print("\n--- SÉLECTION DU FICHIER À TRAITER (Dialogue) ---")
        
        root = tk.Tk()
        root.withdraw() 
        
        input_file = filedialog.askopenfilename(
            title="Sélectionnez le fichier CSV trimestriel à traiter",
            filetypes=(("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*"))
        )
        
        if not input_file:
            print("\nFATAL : Sélection du fichier annulée ou aucun fichier choisi.")
            return

    if not os.path.exists(input_file):
        print(f"\nFATAL : Le fichier d'entrée '{input_file}' n'existe pas. Veuillez vérifier le chemin.")
        return

    # Étape 1: Chargement et correction des décimales
    df_initial_corrected = load_data_and_correct_decimals(input_file)

    if df_initial_corrected is not None and not df_initial_corrected.empty:
        # Étape 2: Calcul de l'inflation YoY et mise à jour du DataFrame
        df_processed = calculate_yoy_inflation(df_initial_corrected)
        
        if not df_processed.empty:
            # Étape 3: Sauvegarde du nouveau fichier
            df_processed.to_csv(OUTPUT_FILE, index=True, date_format='%Y-%m-%d')
            
            print("\n--- TRAITEMENT TERMINÉ ---")
            print(f"Fichier source traité : '{input_file}'")
            print(f"Base de données trimestrielle corrigée créée : '{OUTPUT_FILE}'")
            print(f"Dimensions de la nouvelle base : {df_processed.shape[0]} trimestres x {df_processed.shape[1]} variables")
            print("\nLe nouveau fichier CSV est prêt pour votre application de modélisation.")
        else:
             print("\nÉCHEC DU TRAITEMENT : Le fichier CSV de sortie n'a pas été créé car le DataFrame traité est vide.")

if __name__ == "__main__":
    main()
