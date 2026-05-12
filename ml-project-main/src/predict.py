"""
Prediction utility for the sentiment analysis project.

This module provides a unified interface to load a trained sentiment
analysis model (logistic regression, SVM or transformer) and produce
predictions on new text. It supports probability/confidence scores
where available and gracefully handles models without calibrated
probabilities (e.g. linear SVM).

Usage
-----
Predict a single sentence using the best model saved in a directory:

::

    python -m src.predict --model_dir models --text "Le produit est fantastique !"

Predict all reviews in a CSV file and save the results:

::

    python -m src.predict --model_dir models --input_csv data/my_reviews.csv --output_csv results.csv

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    # transformers is optional for non‑BERT models
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


class SentimentPredictor:
    """Load and run predictions with a previously trained model.

    The constructor inspects ``model_dir/best_model.txt`` to determine which
    model was selected as the best during training. It then loads the
    appropriate artefacts. Supported models are ``logreg``, ``svm`` and
    ``bert``.
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory '{model_dir}' does not exist")
        best_file = self.model_dir / 'best_model.txt'
        if not best_file.exists():
            raise FileNotFoundError(
                f"Missing best_model.txt in '{model_dir}'. Please train models first."
            )
        with open(best_file, 'r') as f:
            self.best_model = f.read().strip()
        if self.best_model not in {'logreg', 'svm', 'bert'}:
            raise ValueError(f"Unknown model type '{self.best_model}'.")

        self.class_names: List[str] = []
        if self.best_model == 'logreg':
            self.pipeline = joblib.load(self.model_dir / 'logreg_model.joblib')
            self.class_names = list(self.pipeline.named_steps['clf'].classes_)
        elif self.best_model == 'svm':
            self.pipeline = joblib.load(self.model_dir / 'svm_model.joblib')
            self.class_names = list(self.pipeline.named_steps['clf'].classes_)
        else:
            # Load transformer
            bert_path = self.model_dir / 'bert_model'
            if AutoTokenizer is None or AutoModelForSequenceClassification is None:
                raise ImportError(
                    "transformers is required for using a BERT model. Install it via pip."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
            # Derive class names from model config
            # huggingface stores id2label mapping in config if present
            id2label = self.bert_model.config.id2label
            # Sort by id to get consistent ordering
            self.class_names = [id2label[i] for i in sorted(id2label)]

    def _predict_pipeline(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Predict using a scikit‑learn pipeline (logreg or svm)."""
        # Clean input type
        if isinstance(texts, str):
            texts = [texts]
        # Get raw predictions
        preds = self.pipeline.predict(texts)
        # Compute confidences
        if self.best_model == 'logreg' and hasattr(self.pipeline.named_steps['clf'], 'predict_proba'):
            probas = self.pipeline.predict_proba(texts)
            max_probas = probas.max(axis=1)
        else:
            # SVM does not provide probabilities; approximate via decision_function
            if hasattr(self.pipeline.named_steps['clf'], 'decision_function'):
                decision = self.pipeline.decision_function(texts)
                # decision can be shape (n_samples, n_classes) or (n_samples,) for binary
                if decision.ndim == 1:
                    decision = np.vstack([-decision, decision]).T
                # apply softmax to get pseudo‑probabilities
                exp_scores = np.exp(decision)
                probas = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                max_probas = probas.max(axis=1)
            else:
                # fallback: assign uniform confidence
                max_probas = np.ones(len(preds)) / len(self.class_names)
        return list(preds), list(max_probas)

    def _predict_bert(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Predict using a transformer model."""
        if isinstance(texts, str):
            texts = [texts]
        # Tokenise
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=1)
        pred_labels = [self.class_names[p.item()] for p in predictions]
        return pred_labels, confidences.tolist()

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict the sentiment for a single piece of text."""
        if self.best_model == 'bert':
            preds, confs = self._predict_bert([text])
        else:
            preds, confs = self._predict_pipeline([text])
        return preds[0], float(confs[0])

    def predict_batch(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Predict sentiments for a list of texts."""
        if self.best_model == 'bert':
            return self._predict_bert(texts)
        else:
            return self._predict_pipeline(texts)


def main(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run sentiment predictions')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained models and best_model.txt')
    parser.add_argument('--text', type=str, default=None,
                        help='Single text to classify')
    parser.add_argument('--input_csv', type=str, default=None,
                        help='Path to a CSV file containing a column named "review"')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save predictions if --input_csv is provided')

    if args is None:
        args = parser.parse_args()

    predictor = SentimentPredictor(args.model_dir)

    if args.text:
        label, confidence = predictor.predict(args.text)
        print(f"Prediction: {label} (confidence: {confidence:.2f})")

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if 'review' not in df.columns:
            raise ValueError("Input CSV must contain a column named 'review'.")
        texts = df['review'].astype(str).tolist()
        labels, confidences = predictor.predict_batch(texts)
        df['predicted_sentiment'] = labels
        df['confidence'] = confidences
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"Saved predictions to {args.output_csv}")
        else:
            print(df.head())

    if not args.text and not args.input_csv:
        parser.print_help()


if __name__ == '__main__':
    main()