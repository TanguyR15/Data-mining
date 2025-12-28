import streamlit as st
import pandas as pd
import json
from transformers import pipeline
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

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_model()

st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"])

st.title("📊 Dashboard E-Commerce 2023")

if not data:
    st.error("Fichier 'scraped_data.json' introuvable. Lancez d'abord le scraper.")
else:
    if option == "Products":
        st.header("🛒 Products List")
        st.dataframe(pd.DataFrame(data.get('products', [])), use_container_width=True)

    elif option == "Testimonials":
        st.header("🗣️ Customer Testimonials")
        st.table(pd.DataFrame(data.get('testimonials', [])))

    elif option == "Reviews":
        st.header("⭐ Review Analysis (Deep Learning)")
        df_reviews = pd.DataFrame(data.get('reviews', []))

        # --- CORRECTION ICI ---
        if df_reviews.empty:
            st.warning("Aucun avis trouvé dans les données scrapées.")
        elif 'date' not in df_reviews.columns:
            st.error("Erreur : La colonne 'date' est absente des données. Vérifiez votre script de scraping.")
            st.write("Colonnes trouvées :", list(df_reviews.columns))
        else:
            # Conversion sécurisée
            df_reviews['date'] = pd.to_datetime(df_reviews['date'], errors='coerce')
            df_2023 = df_reviews[df_reviews['date'].dt.year == 2023].copy()

            if df_2023.empty:
                st.info("Aucun avis trouvé pour l'année 2023. Essayez de scraper plus de pages.")
            else:
                months = df_2023['date'].dt.strftime('%B').unique()
                selected_month = st.select_slider("Select Month in 2023", options=sorted(months))
                
                monthly_reviews = df_2023[df_2023['date'].dt.strftime('%B') == selected_month].copy()
                
                st.subheader(f"Reviews for {selected_month} 2023")
                st.dataframe(monthly_reviews[['date', 'title', 'text']])
                
                if not monthly_reviews.empty:
                    with st.spinner('Analyzing...'):
                        results = sentiment_pipeline(monthly_reviews['text'].tolist(), truncation=True)
                        monthly_reviews['Sentiment'] = [r['label'] for r in results]
                        monthly_reviews['Confidence'] = [r['score'] for r in results]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Sentiment Distribution")
                        counts = monthly_reviews['Sentiment'].value_counts()
                        fig, ax = plt.subplots()
                        counts.plot(kind='bar', color=['green', 'red'], ax=ax)
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Word Cloud")
                        text_combined = " ".join(monthly_reviews['text'])
                        if text_combined.strip():
                            wc = WordCloud(background_color='white').generate(text_combined)
                            fig_wc, ax_wc = plt.subplots()
                            ax_wc.imshow(wc)
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)