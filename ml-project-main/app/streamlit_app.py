"""
Streamlit application for interactive sentiment analysis.

This app allows users to enter individual reviews or upload a CSV file
containing multiple reviews and returns the predicted sentiment along
with confidence scores. Results are colour‑coded for clarity and a
bar chart summarises the distribution of sentiments across batch
inputs. The app dynamically loads the best performing model selected
during training.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure that the src package is discoverable
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(ROOT_DIR))

from src.predict import SentimentPredictor  # type: ignore


@st.cache_resource()
def load_predictor(model_dir: str) -> SentimentPredictor:
    """Load the sentiment predictor once per session.

    Streamlit caches the predictor to avoid reloading models on each
    interaction. The cache is keyed by ``model_dir``.
    """
    return SentimentPredictor(model_dir)


def display_top_features(predictor: SentimentPredictor, top_n: int = 10) -> None:
    """Display the most influential features for logistic regression.

    For each class, show the top ``n`` words with the highest
    coefficients. Note that this only applies to the logistic
    regression model; SVM weights are not directly interpretable and
    transformer models are opaque.
    """
    if predictor.best_model != 'logreg':
        st.sidebar.info("Les mots influents ne sont disponibles que pour le modèle LogReg.")
        return
    vectorizer = predictor.pipeline.named_steps['tfidf']
    clf = predictor.pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()
    for class_index, class_name in enumerate(predictor.class_names):
        coefs = clf.coef_[class_index]
        top_indices = coefs.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        st.sidebar.write(f"Mots les plus influents pour '{class_name}':")
        st.sidebar.write(", ".join(top_words))


def main():
    st.set_page_config(page_title="Analyse de sentiment", layout="centered")
    st.title("🔍 Analyse de sentiment des avis clients")
    st.write(
        "Ce système prédit si un avis est **positif**, **négatif** ou **neutre** "
        "et fournit un score de confiance. Utilisez le champ ci‑dessous pour "
        "tester un avis ou importez un fichier CSV pour une analyse en lot."
    )

    model_dir = ROOT_DIR / 'models'
    predictor = load_predictor(str(model_dir))

    st.sidebar.header("Options")
    # Optionally show top features for logistic regression
    if predictor.best_model == 'logreg':
        if st.sidebar.checkbox("Afficher les mots influents (LogReg)"):
            display_top_features(predictor, top_n=10)

    # Example texts
    examples = {
        'Positif': "J'ai adoré ce produit, la qualité est excellente.",
        'Neutre': "Le service était correct mais sans plus.",
        'Négatif': "Je suis très déçu, ce produit ne fonctionne pas."
    }
    example_choice = st.selectbox(
        "Ou choisissez un exemple d'avis:",
        ["(aucun)"] + list(examples.keys())
    )
    if example_choice != "(aucun)":
        example_text = examples[example_choice]
    else:
        example_text = ""

    # Text input
    user_text = st.text_area(
        "Saisissez un avis client (en français ou en anglais):",
        value=example_text,
        height=150
    )
    if st.button("Analyser l'avis"):
        if not user_text.strip():
            st.warning("Veuillez entrer un avis à analyser.")
        else:
            label, confidence = predictor.predict(user_text)
            # Colour‑coded display
            if label.lower() == 'positive' or label.lower().startswith('pos'):
                st.success(f"Sentiment détecté: Positif (confiance: {confidence:.2f})")
            elif label.lower() == 'negative' or label.lower().startswith('neg'):
                st.error(f"Sentiment détecté: Négatif (confiance: {confidence:.2f})")
            else:
                st.info(f"Sentiment détecté: Neutre (confiance: {confidence:.2f})")

    st.markdown("---")
    st.subheader("Analyse en lot via CSV")
    st.write(
        "Importez un fichier CSV contenant au minimum une colonne appelée "
        "`review`. Chaque ligne de cette colonne sera analysée."
    )
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Impossible de lire le fichier: {e}")
        else:
            if 'review' not in df.columns:
                st.error("La colonne 'review' est manquante dans votre fichier.")
            else:
                texts = df['review'].astype(str).tolist()
                with st.spinner("Analyse en cours..."):
                    labels, confidences = predictor.predict_batch(texts)
                df['predicted_sentiment'] = labels
                df['confidence'] = confidences
                st.success(f"Analyse terminée pour {len(df)} avis.")
                st.dataframe(df.head(100))
                # Distribution plot
                sentiment_counts = df['predicted_sentiment'].value_counts().reindex(predictor.class_names, fill_value=0)
                st.subheader("Distribution des sentiments prédits")
                st.bar_chart(sentiment_counts)
                # Download button
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats en CSV",
                    data=csv_data,
                    file_name="predictions_sentiment.csv",
                    mime="text/csv"
                )


if __name__ == '__main__':
    main()