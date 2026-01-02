import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import joblib  # Optionnel si on charge le modèle localement

# Config page
st.set_page_config(page_title="Détection Fraude MLOps", page_icon="🕵️‍♂️", layout="wide")

st.title("Détection de Fraude Bancaire en Temps Réel")
st.markdown("""
Interface interactive pour tester des transactions. Entrez les features ou uploadez un CSV.
Stack : XGBoost | FastAPI | Streamlit | Plotly
""")

# Sidebar pour inputs
with st.sidebar:
    st.header("Paramètres")
    api_url = st.text_input("URL API", "http://localhost:8000/predict")
    threshold = st.slider("Seuil de fraude", 0.0, 1.0, 0.5)

# Onglets pour modes
tab1, tab2, tab3 = st.tabs(["Prédiction Simple", "Batch CSV", "Visualisations"])

with tab1:
    st.subheader("Testez une seule transaction")
    
    default_fraud_example = "-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985596,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,0.3262713720154,-0.189115517021459,0.133558376740387,-0.0210530534538215,149.62,0.0,0.0,0.0"
    
    features_str = st.text_area(
        "30 features séparées par des virgules",
        value=default_fraud_example,
        height=120,
        help="Copiez-collez exactement 30 valeurs numériques. Pas d'espaces avant/après les virgules."
    )
    
    if st.button("🔍 Prédire la fraude", type="primary"):
        try:
            # Nettoyage très robuste
            parts = [p.strip() for p in features_str.split(",") if p.strip()]
            
            if len(parts) != 30:
                st.error(f"⚠️ Erreur : {len(parts)} valeurs détectées (besoin de exactement 30) !")
                st.info("Vérifiez qu'il n'y a pas de virgule en trop ou de texte parasite.")
            else:
                features = [float(p) for p in parts]
                
                response = requests.post(api_url, json={"features": features})
                response.raise_for_status()  # Détecte les erreurs HTTP
                
                result = response.json()
                proba = result["probability_fraud"]
                is_fraud = result["is_fraud"]
                
                st.success(f"**Probabilité de fraude : {proba:.5f}**")
                st.write(f"**Classification** : {'🟥 Fraude détectée' if is_fraud else '🟩 Transaction légitime'}")
                
                # Gauge circulaire
                fig = px.pie(values=[proba, 1 - proba], names=['Fraude', 'Légitime'], hole=0.5,
                             color_discrete_sequence=["#ff4444", "#44ff44"])
                fig.update_traces(textinfo='none')
                fig.add_annotation(text=f"{proba:.1%}", x=0.5, y=0.5, font_size=30, showarrow=False)
                st.plotly_chart(fig, use_container_width=True)
                
        except ValueError as ve:
            st.error("⚠️ Erreur de format : toutes les valeurs doivent être des nombres décimaux valides.")
            st.info("Conseil : copiez-collez l'exemple sans modification.")
        except requests.exceptions.RequestException as re:
            st.error(f"Erreur de connexion à l'API : {str(re)}")
            st.info("Vérifiez que uvicorn est lancé sur le port 8000.")
        except Exception as e:
            st.error(f"Erreur inattendue : {str(e)}")
with tab2:
    st.subheader("Prédiction Batch via CSV")
    uploaded_file = st.file_uploader("Uploadez un CSV (colonnes : features 1-30)")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        results = []
        for _, row in df.iterrows():
            features = row.tolist()
            response = requests.post(api_url, json={"features": features})
            results.append(response.json())
        st.dataframe(pd.DataFrame(results))
        # Viz batch : histogramme des probas
        fig_batch = px.histogram(results, x="probability_fraud", color="is_fraud", title="Distribution Batch")
        st.plotly_chart(fig_batch)

with tab3:
    st.subheader("Visualisations du Modèle")
    # Charge et affiche rapports interactifs (de visualize.py)
    st.markdown("Ouvrez reports/roc_curve.html pour interagir !")
    # Exemple embed : suppose tu as un modèle chargé
    model = joblib.load("models/xgb_fraud.pkl")  
    st.plotly_chart(px.bar(x=['Feature1', 'Feature2'], y=[0.8, 0.6], title="Importance des Features"))

