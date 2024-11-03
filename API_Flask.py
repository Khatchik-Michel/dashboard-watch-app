from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np

# Ignorer les warnings
warnings.simplefilter(action='ignore', category=AttributeError)
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

# Obtenir le chemin de travail actuel
base_path = os.path.dirname(os.path.abspath(__file__))

# Utiliser des chemins relatifs pour le modèle et les images
model_path = os.path.join(base_path, 'model.pkl')
global_importance_image_path = 'C:/Users/Pc Portable Michel/Downloads/P7/global_importance.png'

# Charger le modèle initial
model = joblib.load(model_path)

# Charger un échantillon des données pour servir de background dataset pour SHAP
background_data = pd.DataFrame([[0]*len(model.feature_names_in_)], columns=model.feature_names_in_)

# Créer un masker basé sur les données
masker = shap.maskers.Independent(background_data)

# Créer un explainer SHAP avec LinearExplainer pour les modèles linéaires (ex: LogisticRegression)
explainer = shap.LinearExplainer(model, masker, feature_perturbation="interventional")

# Obtenir les noms des caractéristiques du modèle
expected_features = model.feature_names_in_

# Simuler des données de classification pour les classes (ici pour l'exemple)
synthetic_data = pd.DataFrame({
    'AMT_GOODS_PRICE': np.random.normal(100000, 25000, 1000),
    'AMT_CREDIT': np.random.normal(150000, 50000, 1000),
    'AMT_INCOME_TOTAL': np.random.normal(200000, 75000, 1000),
    'INCOME_PER_PERSON': np.random.normal(50000, 20000, 1000),
    # Ajouter d'autres features si nécessaire
})

# Prétraiter synthetic_data pour correspondre aux features du modèle
def preprocess_synthetic_data(df):
    # Ajouter les features manquantes avec des zéros
    missing_cols = [feature for feature in expected_features if feature not in df.columns]
    if missing_cols:
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)
    # Réordonner les colonnes pour correspondre à l'ordre attendu par le modèle
    df = df[expected_features]
    # Convertir toutes les colonnes en types numériques
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

# Prétraiter et calculer les scores pour synthetic_data
synthetic_data_processed = preprocess_synthetic_data(synthetic_data.copy())
synthetic_data_scores = model.predict_proba(synthetic_data_processed)[:, 1]
synthetic_data['score'] = synthetic_data_scores

@app.route('/')
def home():
    return "Bienvenue sur l'API de Scoring de Crédit!"

def pretraitement_test(df):
    # Conserver 'SK_ID_CURR' séparément pour éviter qu'elle soit affectée
    sk_id_curr = df['SK_ID_CURR']
    
    # Traitement des colonnes catégorielles binaires
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])
    
    # One-hot encoding pour les colonnes catégorielles
    df = pd.get_dummies(df, dummy_na=True)
    
    # Remplacement des valeurs spécifiques et ajout de nouvelles colonnes
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # Réintégrer 'SK_ID_CURR' après le prétraitement
    df['SK_ID_CURR'] = sk_id_curr
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route pour effectuer des prédictions sur des données envoyées via POST.
    """
    data = request.get_json(force=True)
    try:
        df = pd.DataFrame(data)  # Convertir les données en DataFrame

        # Vérifier si la colonne 'SK_ID_CURR' est présente dans les données
        if 'SK_ID_CURR' not in df.columns:
            return jsonify({"error": "'SK_ID_CURR' manquant dans les données envoyées."}), 400

        # Prétraitement des données
        df = pretraitement_test(df)

        # Extraire l'ID pour usage ultérieur
        sk_id = df['SK_ID_CURR'].iloc[0]

        # Supprimer la colonne 'SK_ID_CURR' des caractéristiques utilisées pour la prédiction
        df = df.drop(columns=['SK_ID_CURR'])

        # Vérifier si des colonnes sont manquantes et les ajouter en bloc
        missing_cols = [feature for feature in expected_features if feature not in df.columns]
        if missing_cols:
            df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)

        # Réordonner les colonnes pour correspondre à l'ordre attendu par le modèle
        df = df[expected_features]

        # Convertir toutes les colonnes en types numériques
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Prédire la probabilité
        prediction = model.predict_proba(df).tolist()  # Convertir les prédictions en liste

        # Calculer l'importance locale des features via SHAP
        shap_values = explainer(df)

        # Sauvegarder l'image d'importance locale sous un nom dynamique basé sur l'ID sélectionné
        local_importance_path = os.path.join(base_path, f'local_importance_{sk_id}.png')

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0])  # Afficher le graphique d'importance locale
        plt.subplots_adjust(left=0.3)
        plt.savefig(local_importance_path)  # Sauvegarder sous le nom dynamique

        print(f"Image générée à l'adresse : {local_importance_path}")

        # Renvoi correct de la réponse avec les prédictions
        return jsonify({"prediction": prediction, "local_importance": f"local_importance_{sk_id}.png"})
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")  # Loguer l'erreur
        return jsonify({"error": str(e)}), 500

# Route pour obtenir l'image d'importance locale pour un ID donné
@app.route('/get_local_importance/<int:id>', methods=['GET'])
def get_local_importance(id):
    """
    Route pour télécharger l'image d'importance locale générée pour un ID spécifique.
    """
    local_importance_path = os.path.join(base_path, f'local_importance_{id}.png')
    if os.path.exists(local_importance_path):
        return send_file(local_importance_path, mimetype='image/png')
    else:
        return jsonify({"error": f"L'image d'importance locale pour l'ID {id} n'a pas été générée."}), 404

# Route pour lire et retourner l'image d'importance globale directement depuis un fichier existant
@app.route('/get_global_importance', methods=['GET'])
def get_global_importance():
    """
    Route pour lire et retourner l'image d'importance globale directement depuis un fichier existant.
    """
    if os.path.exists(global_importance_image_path):
        return send_file(global_importance_image_path, mimetype='image/png')
    else:
        return jsonify({"error": "L'image d'importance globale n'a pas été trouvée."}), 404

# Route pour obtenir la distribution d'une feature sélectionnée
@app.route('/get_feature_distribution', methods=['POST'])
def get_feature_distribution():
    """
    Retourner la distribution d'une feature donnée ainsi que la valeur du client.
    """
    data = request.get_json(force=True)
    feature = data.get('feature', None)
    client_value = data.get('client_value', None)

    if feature not in synthetic_data.columns:
        return jsonify({"error": "Feature non reconnue."}), 400

    # Obtenir la distribution des valeurs de la feature
    feature_values = synthetic_data[feature]

    response = {
        "feature": feature,
        "values": feature_values.tolist(),
        "client_value": client_value
    }

    return jsonify(response)

# Nouvelle route pour obtenir les données pour le graphique bivarié
@app.route('/get_bivariate_data', methods=['POST'])
def get_bivariate_data():
    """
    Retourner les données pour un graphique bivarié entre deux features, avec les scores des clients.
    """
    data = request.get_json(force=True)
    feature_x = data.get('feature_x', None)
    feature_y = data.get('feature_y', None)
    client_data = data.get('client_data', None)  # Contient les valeurs du client pour les features

    if feature_x not in synthetic_data.columns or feature_y not in synthetic_data.columns:
        return jsonify({"error": "Features non reconnues."}), 400

    try:
        # Récupérer les données nécessaires
        df = synthetic_data[[feature_x, feature_y, 'score']]

        response = {
            "feature_x": df[feature_x].tolist(),
            "feature_y": df[feature_y].tolist(),
            "scores": df['score'].tolist(),
            "client_x": client_data.get(feature_x),
            "client_y": client_data.get(feature_y),
            "client_score": client_data.get('score')
        }
        return jsonify(response)
    except Exception as e:
        print(f"Erreur lors de la génération des données bivariées: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Port par défaut : 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
