import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

# Streamlit page configuration
st.set_page_config(
    page_title="Prévision de l'inflation",
    layout="wide"
)

# Constantes et chemins de fichiers
DATA_FILE = 'inflation_data_YoY_quarterly.csv'
TARGET_CHOICES = {
    "Inflation Globale (IPC YoY)": 'Inflation_YoY - IPC',
}

# --- FONCTIONS DE TRAITEMENT DES DONNÉES ---

@st.cache_data
def load_data(file_path):
    """Charge les données et effectue le nettoyage initial."""
    try:
        # Charge les données, définit observation_date comme index et parse les dates
        df = pd.read_csv(file_path, index_col='observation_date', parse_dates=True)
        df = df.dropna()
        return df
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier de données '{file_path}' est introuvable.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement ou du traitement du fichier : {e}")
        return None

def prepare_data(df, target_column, lag_order):
    """Prépare les données pour le modèle: log-transformation et sélection des features."""
    df_clean = df.copy()

    # 1. Log-transformation du PIB
    if 'GDP' in df_clean.columns:
        df_clean['GDP'] = df_clean['GDP'].replace(0, np.nan) 
        df_clean['log_GDP'] = np.log(df_clean['GDP'])
        df_clean = df_clean.drop(columns=['GDP'])
    
    # 2. Définition des variables X et y
    y = df_clean[target_column]
    
    # Exclure les cibles d'inflation de X
    all_targets = list(TARGET_CHOICES.values())
    X_cols = [col for col in df_clean.columns if col not in all_targets]
    X = df_clean[X_cols]
    
    # 3. Ajout du retard de l'inflation (Lag) - DYNAMIQUE
    lag_col_name = f'{target_column}_L{lag_order}'
    X[lag_col_name] = y.shift(lag_order) # Retard de 'lag_order' trimestres
    X = X.dropna()
    y = y[X.index] # Synchroniser y avec le nouvel index de X

    return X, y, X.columns, lag_col_name

def train_and_evaluate(X, y, target_column, test_ratio_percent, lag_col_name):
    """Entraîne le modèle, évalue et effectue la prévision N+2 (8 périodes)."""
    
    # 1. Standardisation de X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # 2. Séparation des données (sans shuffle pour les séries temporelles) - DYNAMIQUE
    test_ratio = test_ratio_percent / 100
    split_index = int(len(X_scaled_df) * (1 - test_ratio))
    X_train, X_test = X_scaled_df[:split_index], X_scaled_df[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 3. Entraînement du Modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Prédictions
    y_pred_full = model.predict(X_scaled_df)
    y_pred_test = model.predict(X_test)
    
    # 5. Métriques
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    # 6. Prévision N+2 (8 trimestres suivants)
    forecast_dates = pd.date_range(start=X.index[-1], periods=9, freq='QS-OCT')[1:] 
    forecast_data = []

    # Dernière observation standardisée pour l'initialisation de la prévision
    future_X_scaled = X_scaled_df.iloc[-1].copy() 
    
    # La dernière valeur observée de la variable de retard est le point de départ
    lag_index = X.columns.get_loc(lag_col_name) # Obtient l'indice de la colonne de retard
    current_lag_value_scaled = X_scaled_df.iloc[-1][lag_col_name]
    
    for i in range(8):
        # Utiliser la prévision du trimestre précédent comme valeur de retard pour le calcul
        future_X_scaled_array = future_X_scaled.values.copy()
        future_X_scaled_array[lag_index] = current_lag_value_scaled
        
        # Prédiction
        pred_value = model.predict(future_X_scaled_array.reshape(1, -1))[0]
        
        forecast_data.append({
            'Date': forecast_dates[i],
            'Prévision (YoY)': pred_value
        })
        
        # Mettre à jour la valeur de retard pour l'itération suivante.
        # 1. Trouver l'index de la colonne lag dans le scaler (déjà fait: lag_index)
        # 2. Utiliser les statistiques de la variable `lag_col_name` qui a été standardisée
        mean_lag = scaler.mean_[lag_index]
        std_lag = scaler.scale_[lag_index]
        
        # 3. La prochaine valeur du lag doit être la prédiction non-scalée (pred_value).
        # On la standardise pour l'utiliser comme input à T+1.
        current_lag_value_scaled = (pred_value - mean_lag) / std_lag


    forecast_df = pd.DataFrame(forecast_data).set_index('Date')
    
    return model, r2, mse, X.columns, y_pred_full, forecast_df, y, split_index, X.index


# --- INTERFACE STREAMLIT ---

def app():
    st.title("Prévision de l'inflation")

    # 1. Chargement des Données
    df = load_data(DATA_FILE)

    if df is None or df.empty:
        st.stop()
        
    # --- BARRE LATÉRALE (Contrôles Interactifs) ---
    st.sidebar.header("Configuration du Modèle")
    
    target_name = list(TARGET_CHOICES.keys())[0]
    target_column = TARGET_CHOICES[target_name]

    # Contrôle 1: Ratio Train/Test
    test_ratio_percent = st.sidebar.slider(
        "Proportion des Données de Test (%)",
        min_value=10, max_value=40, value=20, step=5,
        help="Détermine le pourcentage de la série temporelle historique utilisé pour valider le modèle (Test). Le reste est utilisé pour l'entraînement (Train)."
    )
    
    # Contrôle 2: Ordre du Retard (Lag)
    lag_order = st.sidebar.slider(
        "Ordre de Retard de l'Inflation (Trimestres)",
        min_value=1, max_value=8, value=4, step=1,
        help="Spécifie combien de trimestres en arrière l'inflation passée est utilisée comme variable explicative. Le lag de 4 est souvent utilisé pour modéliser l'inertie annuelle."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Cible analysée : **{target_name}**")
    
    # 2. Préparation des Données
    X, y, feature_names, lag_col_name = prepare_data(df, target_column, lag_order)

    if X.shape[0] < 10:
        st.warning("Pas assez de données pour l'entraînement après nettoyage et ajout du retard.")
        st.stop()
        
    # 3. Entraînement et Évaluation
    model, r2, mse, feature_names, y_pred_full, forecast_df, y_original, split_index, index_dates = train_and_evaluate(
        X, y, target_column, test_ratio_percent, lag_col_name
    )

    # --- TABS ---
    tab1, tab2 = st.tabs(["Présentation & Données", "Modélisation & Prévision (N+2)"])

    with tab1:
        st.header("1. Objectif et Configuration du modèle")
        
        st.subheader("Qu'est-ce que l'inflation ?")
        st.markdown("""
        L'**inflation** est la hausse générale et durable des prix des biens et services. Elle se traduit par une diminution du pouvoir d'achat de la monnaie.
        
        L'indicateur utilisé ici est l'**IPC (Indice des Prix à la Consommation) en glissement annuel (YoY)**.
        """)
        
        st.markdown(f"""
        Cette application utilise un modèle de **Régression Linéaire Multiple** pour analyser et projeter l'**Inflation Globale (IPC YoY)**. 
        
        L'objectif est d'étudier l'impact des variables macroéconomiques sur les prix et de générer une prévision en intégrant un facteur d'inertie (l'inflation passée).
        
        ---
        
        ### Configuration actuelle
        
        * **Ordre de Retard (Lag) :** Actuellement fixé à **{lag_order} trimestres** (modifiable dans la barre latérale). Cette valeur est cruciale pour capturer l'effet d'inertie de l'inflation passée.
        * **Ratio Train/Test :** Le modèle utilise **{100 - test_ratio_percent}%** des données historiques pour l'entraînement et **{test_ratio_percent}%** pour la validation (modifiable dans la barre latérale).
        """)

        st.subheader("Les Variables macroéconomiques utilisées")
        st.markdown(f"""
        Le modèle utilise les facteurs suivants comme déterminants (features) :
        * **Inflation Passée (Lag) :** L'inflation il y a **{lag_order}** trimestres est incluse comme variable clé pour capturer l'effet d'inertie (le choix du lag est modifiable dans la barre latérale).
        * **GDP (Produit Intérieur Brut) :** Mesure de l'activité économique.
        * **Taux d'intérêt (Rendements des obligations) :** Facteur monétaire.
        * **Taux de Chômage (Total unemployment rate) :** Indicateur de la tension sur le marché du travail.
        * **Prix du Pétrole Brut (Euro) :** Indicateur des chocs d'offre énergétique.
        * **Taux de Change (€/$).**
        * **Masse Monétaire (Monetary aggregate).**
        """)
        
        st.subheader("Aperçu du jeu de données utilisé")
        st.caption(f"Le jeu de données aligné contient {len(df)} observations trimestrielles, de {df.index.min().strftime('%Y-%m')} à {df.index.max().strftime('%Y-%m')}.")
        st.dataframe(df.head(5).style.format(precision=2), width='stretch')
        
        st.markdown("---")
        st.header("2. Exploration des séries temporelles")
        
        st.subheader("Série temporelle : Inflation globale (IPC YoY)")
        st.markdown("Ce graphique montre l'évolution de l'inflation ciblée. Observez la volatilité et la tendance à long terme avant l'entraînement du modèle.")

        # Graphique Plotly de la Série Temporelle
        fig_ts = px.line(y, 
                         title=f'Évolution de {target_name}',
                         labels={'index': 'Date', 'value': 'Taux en % (YoY)'},
                         height=400,
                         template="plotly_white") # Style plus propre
        fig_ts.update_traces(mode='lines', line=dict(color='#0052A3', width=2))
        fig_ts.update_layout(hovermode="x unified", title_x=0.5)
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("Matrice de corrélation (Déterminants)")
        st.markdown("Visualisation de la relation linéaire entre toutes les variables macroéconomiques. Les valeurs élevées (proches de 1 ou -1) indiquent une forte relation qui peut être exploitée par le modèle.")
        
        # Matrice de corrélation
        corr = df.corr().round(2)
        fig_corr = px.imshow(corr, 
                             text_auto=True, 
                             aspect="auto",
                             color_continuous_scale='RdBu_r', 
                             title="Corrélation entre les Variables",
                             template="plotly_white")
        fig_corr.update_layout(title_x=0.5)
        st.plotly_chart(fig_corr, use_container_width=True)


    # --- TAB 2: MODÉLISATION & PRÉVISION ---
    with tab2:
        st.header("3. Performance du modèle et interprétation")

        # Display performance metrics (Style plus 'finance')
        st.subheader("Mesures d'ajustement et de fiabilité")
        
        col1, col2, col3 = st.columns(3)
        
        col1.markdown(
            f"""
            <div style="padding: 10px; border-radius: 8px; border-left: 5px solid #0052A3; background-color: #f0f2f6;">
                <p style="margin: 0; font-size: 14px; color: #555;">R-carré (R²)</p>
                <h3 style="margin: 0; color: #0052A3;">{r2:.4f}</h3>
                <p style="margin: 0; font-size: 10px; color: #888;">% de variance expliquée sur le set de test</p>
            </div>
            """, unsafe_allow_html=True
        )
        
        col2.markdown(
            f"""
            <div style="padding: 10px; border-radius: 8px; border-left: 5px solid #FF4B4B; background-color: #f0f2f6;">
                <p style="margin: 0; font-size: 14px; color: #555;">Erreur Quadratique Moyenne (MSE)</p>
                <h3 style="margin: 0; color: #FF4B4B;">{mse:.4f}</h3>
                <p style="margin: 0; font-size: 10px; color: #888;">Sur le set de test (plus faible est mieux)</p>
            </div>
            """, unsafe_allow_html=True
        )

        col3.markdown(
            f"""
            <div style="padding: 10px; border-radius: 8px; border-left: 5px solid #008000; background-color: #f0f2f6;">
                <p style="margin: 0; font-size: 14px; color: #555;">Taille de l'Échantillon Train</p>
                <h3 style="margin: 0; color: #008000;">{split_index} périodes</h3>
                <p style="margin: 0; font-size: 10px; color: #888;">Ratio Train/Test : {100 - test_ratio_percent}% / {test_ratio_percent}%</p>
            </div>
            """, unsafe_allow_html=True
        )
        
        st.subheader("Modèle économétrique utilisé")
        st.markdown(r"""
        Le modèle de régression linéaire utilisé est une forme simplifiée de modèle autorégressif (ARX) :
        $$ \text{Inflation}_t = \beta_0 + \sum_{i} \beta_i \cdot \text{Facteur\_Macro}_i + \beta_{lag} \cdot \text{Inflation}_{t-L} + \varepsilon_t $$
        Où $L$ est l'ordre de retard sélectionné dans le panneau latéral (actuellement $L={lag_order}$ trimestres), $\text{Facteur\_Macro}_i$ sont les autres variables (PIB, Taux, etc.), et $\beta_i$ sont les coefficients.
        """)


        st.subheader("Coefficients standardisés du modèle")
        
        # Créer le DataFrame des coefficients
        coef_df = pd.DataFrame({
            'Variable': feature_names,
            'Coefficient Standardisé': model.coef_
        })
        coef_df['Impact Absolu'] = np.abs(coef_df['Coefficient Standardisé'])
        coef_df = coef_df.sort_values(by='Impact Absolu', ascending=False)
        coef_df = coef_df.drop(columns=['Impact Absolu'])
        
        st.dataframe(coef_df.reset_index(drop=True).style.format({'Coefficient Standardisé': "{:.4f}"}), width='stretch')

        st.caption("""
        **Interprétation des coefficients :** Les coefficients sont standardisés (mis à l'échelle), ce qui permet de comparer l'importance relative des variables :
        * **Signe :** Un coefficient positif signifie qu'une augmentation de la variable explicative tend à augmenter l'inflation. Inversement pour un signe négatif.
        * **Magnitude :** Plus la valeur absolue est grande, plus la variable a une influence forte sur la cible.
        * **Variable de retard :** La variable `{lag_col_name}` modélise l'inertie de l'inflation passée sur l'inflation actuelle.
        """)
        
        st.header("4. Prévisions pour les 8 prochains trimestres (N+2)")
        
        # Tableau des Prédictions N+2
        st.subheader(f"Prédictions pour {target_name} (Horizon N+2)")
        st.markdown("Prévision séquentielle (chaque trimestre prédit est utilisé comme input pour le lag du trimestre suivant) basée sur l'état actuel des variables.")
        
        # Formater le tableau pour la lisibilité
        forecast_display = forecast_df.copy()
        forecast_display.index = forecast_df.index.strftime('%Y - T%q')
        forecast_display.columns = ['Taux Prévu (YoY) en %']
        
        st.dataframe(forecast_display.style.format("{:.3f}"), width='stretch')

        # Graphique de Prévision (Historique + Forecast)
        st.subheader("Historique et projection du modèle")
        st.markdown("""
        Ce graphique met en contraste l'historique réel de l'inflation avec l'ajustement du modèle sur le passé, et la projection future pour les 8 trimestres suivants.
        """)
        
        # Préparation des données pour le graphique
        plot_df = pd.DataFrame({
            'Historique Réel': y_original,
            'Historique Prédit': y_pred_full
        })
        
        # Utilisation de Plotly pour le graphique de prévision (interactive)
        fig_forecast = go.Figure()

        # 1. Historique Réel (Bleu)
        fig_forecast.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['Historique Réel'],
            mode='lines', name='Inflation Réelle',
            line=dict(color='#0052A3', width=2) 
        ))

        # 2. Historique Prédit (Vert clair - Ajustement Modèle)
        fig_forecast.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['Historique Prédit'],
            mode='lines', name='Ajustement Modèle',
            line=dict(color='#32CD32', dash='dash', width=1.5)
        ))

        # 3. Prévision N+2 (Rouge vif)
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df['Prévision (YoY)'],
            mode='lines+markers', name='Prévision (N+2)',
            line=dict(color='#FF4B4B', width=3), marker=dict(size=6, symbol='circle')
        ))

        # Ajout de la ligne verticale (séparation historique/futur)
        last_date_str = index_dates[-1].strftime('%Y-%m-%d')
        
        fig_forecast.add_vline(x=last_date_str, 
                               line_width=2, 
                               line_dash="dot", 
                               line_color="darkgray")

        # Annotation de la séparation
        fig_forecast.add_annotation(
            x=last_date_str,
            y=1.05,  
            yref="paper", 
            text="Fin des Données Réelles",
            showarrow=False,
            font=dict(color="darkgray", size=10),
            bgcolor="rgba(255, 255, 255, 0.7)",
        )

        fig_forecast.update_layout(
            title=f'Projection de l\'Inflation : Historique vs. Prévision',
            xaxis_title='Date Trimestrielle',
            yaxis_title='Taux d\'Inflation (YoY, en %)',
            hovermode="x unified",
            title_x=0.5,
            height=600,
            template="plotly_white"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)


if __name__ == '__main__':
    app()
