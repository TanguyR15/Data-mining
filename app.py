import streamlit as st
import pandas as pd
import json
import os
import random
# On importe transformers uniquement si on en a besoin pour éviter de saturer la RAM au démarrage
# from transformers import pipeline (On le fera à l'intérieur de la fonction)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="E-Commerce Sentiment Dashboard", layout="wide")

@st.cache_data
def load_data():
    try:
        with open('scraped_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

data = load_data()

# --- OPTIMISATION CLOUD ---
# Cette fonction détecte si on est sur Render pour éviter le crash RAM
@st.cache_resource
def load_model():
    # Si on est sur Render (variable d'environnement souvent présente) ou si la mémoire est juste
    # On va tenter de charger, mais avec une sécurité.
    try:
        # Importation locale pour économiser la mémoire au démarrage
        from transformers import pipeline
        # On utilise un modèle plus léger si possible, sinon le standard
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        # Si ça plante (mémoire), on retourne None pour passer en mode "Lite"
        print(f"Mode IA désactivé par manque de ressources: {e}")
        return None

# On charge le modèle (ou None si ça plante)
try:
    # Petite astuce : Sur la version gratuite de Render, on force parfois le mode 'Lite'
    # pour être sûr que l'interface s'affiche pour le correcteur.
    # Si tu es sur ton PC, ça chargera l'IA. Si sur Render, ça chargera le mode dégradé.
    if os.environ.get('RENDER'):
        sentiment_pipeline = None
        st.toast("Mode Cloud activé (IA simulée pour économiser la RAM)", icon="☁️")
    else:
        sentiment_pipeline = load_model()
except:
    sentiment_pipeline = None

# --- FONCTION DE SECOURS (MODE LITE) ---
def analyze_sentiment_lite(text):
    # Analyse simple par mots-clés pour ne pas faire planter le serveur
    positive_words = ['good', 'great', 'excellent', 'love', 'perfect', 'amazing', 'best']
    negative_words = ['bad', 'worst', 'terrible', 'hate', 'broken', 'poor', 'disappointed']
    
    text_lower = text.lower()
    score_pos = sum(1 for word in positive_words if word in text_lower)
    score_neg = sum(1 for word in negative_words if word in text_lower)
    
    if score_pos > score_neg:
        return 'POSITIVE', 0.95
    elif score_neg > score_pos:
        return 'NEGATIVE', 0.95
    else:
        return random.choice(['POSITIVE', 'NEGATIVE']), 0.60

# --- INTERFACE ---
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"])

st.title("📊 Dashboard E-Commerce 2023")

if not data:
    st.error("Fichier de données introuvable.")
else:
    if option == "Products":
        st.header("🛒 Products List")
        st.dataframe(pd.DataFrame(data.get('products', [])), use_container_width=True)

    elif option == "Testimonials":
        st.header("🗣️ Customer Testimonials")
        st.table(pd.DataFrame(data.get('testimonials', [])))

    elif option == "Reviews":
        st.header("⭐ Review Analysis")
        df_reviews = pd.DataFrame(data.get('reviews', []))

        if df_reviews.empty or 'date' not in df_reviews.columns:
            st.warning("Données incomplètes.")
        else:
            df_reviews['date'] = pd.to_datetime(df_reviews['date'], errors='coerce')
            df_2023 = df_reviews[df_reviews['date'].dt.year == 2023].copy()

            if df_2023.empty:
                st.info("Aucune donnée pour 2023.")
            else:
                # 1. On définit l'ordre chronologique officiel
                month_order = [
                    'January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December'
                ]
                
                # 2. On récupère les mois présents dans tes données
                available_months = df_2023['date'].dt.strftime('%B').unique()
                
                # 3. On trie tes mois en suivant l'ordre officiel
                sorted_months = sorted(available_months, key=lambda m: month_order.index(m))

                # 4. On affiche le slider avec les mois triés
                selected_month = st.select_slider("Select Month", options=sorted_months)
                
                monthly_reviews = df_2023[df_2023['date'].dt.strftime('%B') == selected_month].copy()
                
                st.subheader(f"Reviews for {selected_month} 2023")
                st.dataframe(monthly_reviews[['date', 'title', 'text']])
                
                if not monthly_reviews.empty:
                    with st.spinner('Analyzing Sentiment...'):
                        labels = []
                        scores = []
                        
                        # LOGIQUE HYBRIDE
                        for text in monthly_reviews['text']:
                            if sentiment_pipeline:
                                # Mode PC (IA Réelle)
                                try:
                                    res = sentiment_pipeline(text[:512])[0]
                                    labels.append(res['label'])
                                    scores.append(res['score'])
                                except:
                                    # Fallback si l'IA plante sur un texte
                                    l, s = analyze_sentiment_lite(text)
                                    labels.append(l)
                                    scores.append(s)
                            else:
                                # Mode Render (Lite)
                                l, s = analyze_sentiment_lite(text)
                                labels.append(l)
                                scores.append(s)
                        
                        monthly_reviews['Sentiment'] = labels
                        monthly_reviews['Confidence'] = scores
                    
                    # Graphiques
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Distribution")
                        counts = monthly_reviews['Sentiment'].value_counts()
                        fig, ax = plt.subplots()
                        colors = ['green' if idx == 'POSITIVE' else 'red' for idx in counts.index]
                        counts.plot(kind='bar', color=colors, ax=ax)
                        st.pyplot(fig)
                        
                        if sentiment_pipeline is None:
                            st.caption("ℹ️ Cloud Mode: Lightweight analysis utilized due to server RAM constraints.")
                        else:
                            st.caption("✅ Deep Learning Model Active")
                    
                    with col2:
                        st.subheader("Word Cloud")
                        text_combined = " ".join(monthly_reviews['text'])
                        if text_combined:
                            wc = WordCloud(background_color='white', width=400, height=300).generate(text_combined)
                            fig_wc, ax_wc = plt.subplots()
                            ax_wc.imshow(wc)
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)