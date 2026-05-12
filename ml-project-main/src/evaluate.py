"""
Evaluation script for the sentiment analysis project.

This script loads the dataset and the trained models, computes
evaluation metrics on the held‑out test set and optionally produces
visualisations such as confusion matrices. It is intended to be run
after training to provide an objective assessment of model
performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)

from .preprocess import load_dataset, preprocess_dataframe
from .train import split_data


def evaluate_model(model, X_test: pd.Series, y_test: pd.Series) -> Tuple[dict, np.ndarray]:
    """Evaluate a scikit‑learn model on the test set.

    Returns metrics (accuracy, precision, recall, f1) and the confusion matrix.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }, cm


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, save_path: Path) -> None:
    """Plot and save a confusion matrix using seaborn."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description='Evaluate trained sentiment models')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the CSV dataset used for training')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained models and best_model.txt')
    parser.add_argument('--out_dir', type=str, default='reports',
                        help='Directory to save evaluation reports and plots')

    if args is None:
        args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataset(args.data, text_column='review', label_column='sentiment')
    df_processed = preprocess_dataframe(df, text_column='review')
    _, _, X_test, _, _, y_test = split_data(
        df_processed, 'clean_text', 'sentiment', test_size=0.15, val_size=0.15
    )

    # Evaluate logistic regression
    logreg_model = joblib.load(Path(args.model_dir) / 'logreg_model.joblib')
    logreg_metrics, logreg_cm = evaluate_model(logreg_model, X_test, y_test)
    # Save confusion matrix
    plot_confusion_matrix(
        logreg_cm, list(logreg_model.classes_), 'Logistic Regression Confusion Matrix',
        out_dir / 'confusion_logreg.png'
    )

    # Evaluate SVM
    svm_model = joblib.load(Path(args.model_dir) / 'svm_model.joblib')
    svm_metrics, svm_cm = evaluate_model(svm_model, X_test, y_test)
    plot_confusion_matrix(
        svm_cm, list(svm_model.classes_), 'SVM Confusion Matrix',
        out_dir / 'confusion_svm.png'
    )

    # Write report
    report_path = out_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write('Evaluation Report\n')
        f.write('=================\n\n')
        f.write('Logistic Regression:\n')
        for k, v in logreg_metrics.items():
            f.write(f'  {k}: {v:.4f}\n')
        f.write('\nSVM:\n')
        for k, v in svm_metrics.items():
            f.write(f'  {k}: {v:.4f}\n')
    print(f"Saved evaluation report to {report_path}")


if __name__ == '__main__':
    main()