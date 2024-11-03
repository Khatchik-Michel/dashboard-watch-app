import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.title("Application de Scoring de Crédit")

# Liste des features disponibles pour l'analyse
features = ['AMT_GOODS_PRICE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'INCOME_PER_PERSON']

# Initialiser les variables de session pour les prédictions et les graphes d'importance
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'local_importance_generated' not in st.session_state:
    st.session_state['local_importance_generated'] = False
if 'global_importance_generated' not in st.session_state:
    st.session_state['global_importance_generated'] = False

# Initialisation du fichier uploadé et des données
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données")
    st.write(data.head())

    # Nettoyage des données
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Conversion des colonnes en types numériques
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Calculer `INCOME_PER_PERSON` si elle est absente
    if 'INCOME_PER_PERSON' not in data.columns:
        data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
        data['INCOME_PER_PERSON'].fillna(0, inplace=True)

    # Choisir entre prédire pour tous les IDs ou un ID spécifique
    predire_tous = st.checkbox("Prédire pour tous les IDs", value=True)
    selected_id = None  # Initialiser selected_id
    selected_feature = None  # Initialiser selected_feature

    if predire_tous:
        # Préparation des données pour l'API (tous les IDs)
        data_json = data.to_dict(orient='records')
        st.write("Données envoyées à l'API:", data_json) 
    else:
        # Liste déroulante pour choisir l'ID
        selected_id = st.selectbox("Choisissez un ID", data['SK_ID_CURR'].unique())
        
        # Préparation des données pour l'API (ID spécifique)
        selected_data = data[data['SK_ID_CURR'] == selected_id]
        data_json = selected_data.to_dict(orient='records')
        st.write("Données envoyées à l'API:", data_json)  

    # Choisir la feature à analyser
    selected_feature = st.selectbox("Choisissez une feature pour l'analyse de distribution", features)

    # Bouton pour lancer les prédictions
    if st.button("Prédire"):
        # Appel à l'API Flask en local sur le port 5000
        api_url = "http://127.0.0.1:5000/predict"
        response = requests.post(api_url, json=data_json, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            # Stocker la réponse de l'API dans le state
            st.session_state['predictions'] = response.json()
            st.session_state['local_importance_generated'] = True  # Marquer que l'importance locale a été générée
            st.write("Réponse brute de l'API:", st.session_state['predictions'])  # Affichage de la réponse complète
        else:
            st.write("Erreur dans l'appel à l'API")
            st.write(f"Status Code: {response.status_code}")
            st.write(f"Message: {response.text}")

# Afficher les prédictions si elles existent
if st.session_state['predictions'] is not None:
    st.write("Prédictions")
    
    # Extraire les valeurs de prédiction
    predictions = st.session_state['predictions']['prediction']

    for idx, pred in enumerate(predictions):
        score = float(pred[1])  # Convertir la probabilité de classe 1 en float
        st.write(f"Prédiction pour l'entrée {idx + 1}: {'Crédit accepté' if score >= 0.5 else 'Crédit refusé'} (score: {score:.2f})")

        # Afficher la jauge avec la bande blanche de 0 au score, et une ligne pour indiquer 0.5
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Score de Crédit"},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "green" if score >= 0.5 else "red"},  # Couleur de fond selon acceptation/refus
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, score], 'color': "white"}],  # Bande blanche de 0 au score
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5}  # Ligne pour indiquer 0.5
            }
        ))

        st.plotly_chart(fig)

    # Afficher le graphique d'importance locale seulement après que la prédiction a été faite
    if st.session_state['local_importance_generated'] and selected_id is not None:
        st.write("Affichage de l'importance locale des caractéristiques")
        image_url = f"http://127.0.0.1:5000/get_local_importance/{selected_id}"
        st.image(image_url, caption=f"Importance locale des caractéristiques pour l'ID {selected_id}", use_column_width=True)

# Bouton pour afficher l'importance globale des caractéristiques
if st.button("Afficher l'importance globale des caractéristiques"):
    with st.spinner('Chargement du graphique...'):
        # Appel à l'API Flask pour obtenir l'image de l'importance globale
        api_url_global = "http://127.0.0.1:5000/get_global_importance"
        response_global = requests.get(api_url_global)

        if response_global.status_code == 200:
            # Afficher l'image d'importance globale
            img_url_global = "http://127.0.0.1:5000/get_global_importance"
            st.image(img_url_global, caption="Importance globale des caractéristiques", use_column_width=True)
        else:
            st.write("Erreur lors de la génération de l'importance globale.")

# Graphiques de distribution de feature pour un client spécifique
if data is not None and selected_id is not None and selected_feature is not None:
    client_value = data.loc[data['SK_ID_CURR'] == selected_id, selected_feature].values[0]
    
    # Appel à l'API pour obtenir la distribution de la feature
    api_feature_url = "http://127.0.0.1:5000/get_feature_distribution"
    response = requests.post(api_feature_url, json={"feature": selected_feature, "client_value": client_value})
    
    if response.status_code == 200:
        feature_data = response.json()
        values = feature_data["values"]
        
        # Afficher le graphique de distribution
        fig = px.histogram(values, nbins=50, opacity=0.75, labels={'value': selected_feature}, title=f"Distribution de {selected_feature}")
        fig.add_vline(x=client_value, line=dict(color="blue"), annotation_text="Valeur Client")
        st.plotly_chart(fig)
    else:
        st.write("Erreur lors de la récupération des données de distribution pour la feature.")

# Nouvelle section pour l'analyse bivariée
if data is not None and selected_id is not None:
    st.write("### Analyse bivariée")
    selected_feature_x = st.selectbox("Choisissez la première feature pour l'analyse bivariée", features, key='feature_x')
    selected_feature_y = st.selectbox("Choisissez la deuxième feature pour l'analyse bivariée", features, key='feature_y')

    if selected_feature_x != selected_feature_y:
        client_data_row = data[data['SK_ID_CURR'] == selected_id].iloc[0]
        client_feature_x = client_data_row[selected_feature_x]
        client_feature_y = client_data_row[selected_feature_y]
        client_score = st.session_state['predictions']['prediction'][0][1]  # Score du client

        # Préparer les données pour l'API
        api_bivariate_url = "http://127.0.0.1:5000/get_bivariate_data"
        payload = {
            "feature_x": selected_feature_x,
            "feature_y": selected_feature_y,
            "client_data": {
                selected_feature_x: client_feature_x,
                selected_feature_y: client_feature_y,
                'score': client_score
            }
        }

        # Appel à l'API
        response = requests.post(api_bivariate_url, json=payload)
        if response.status_code == 200:
            bivariate_data = response.json()
            df_plot = pd.DataFrame({
                selected_feature_x: bivariate_data['feature_x'],
                selected_feature_y: bivariate_data['feature_y'],
                'score': bivariate_data['scores']
            })
            # Afficher le graphique
            fig = px.scatter(df_plot, x=selected_feature_x, y=selected_feature_y, color='score', 
                             color_continuous_scale='RdYlGn', title="Analyse bivariée avec dégradé de couleur selon le score")
            # Ajouter la position du client
            fig.add_trace(go.Scatter(
                x=[bivariate_data['client_x']],
                y=[bivariate_data['client_y']],
                mode='markers',
                marker=dict(color='black', size=12, symbol='x'),
                name='Client'
            ))
            st.plotly_chart(fig)
        else:
            st.write("Erreur lors de la récupération des données pour l'analyse bivariée.")
    else:
        st.write("Veuillez choisir deux features différentes pour l'analyse bivariée.")
