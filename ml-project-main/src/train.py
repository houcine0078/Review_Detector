"""
Training script for customer review sentiment analysis.

This script supports training multiple models on a labelled reviews dataset
to classify text into ``positive``, ``negative`` or ``neutral`` categories.
It currently implements classical machine‑learning approaches (TF‑IDF +
logistic regression and TF‑IDF + linear SVM) and provides a stub for
fine‑tuning a transformer model (e.g. CamemBERT or DistilBERT). The
script splits the data into train/validation/test sets, performs
training, evaluates performance on the validation set, and chooses the
best performing model. Models and associated vectorisers/tokenisers are
saved to disk for later use.

Example
-------
To train baseline models on a CSV file and save the results to the
``models`` directory:

::

    python -m src.train --data data/sample_reviews.csv --out_dir models

To include transformer fine‑tuning (this can take some time and may
require a GPU):

::

    python -m src.train --data data/my_dataset.csv --out_dir models \
        --train_bert --bert_model camembert-base

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

from .preprocess import load_dataset, preprocess_dataframe


def split_data(df: pd.DataFrame,
               text_column: str,
               label_column: str,
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42) -> Tuple:
    """Split the dataset into train, validation and test sets.

    A two‑stage split is performed: first a combined validation/test set
    is extracted, then that set is split into validation and test splits.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    text_column : str
        Name of the column containing the processed text.
    label_column : str
        Name of the column containing the labels.
    test_size : float, default 0.15
        Proportion of the data to reserve for the test set.
    val_size : float, default 0.15
        Proportion of the data to reserve for the validation set. This is
        computed relative to the original dataset size.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df[text_column]
    y = df[label_column]
    # first split off combined validation+test set
    combined_size = val_size + test_size
    X_train, X_combined, y_train, y_combined = train_test_split(
        X, y, test_size=combined_size, stratify=y, random_state=random_state
    )
    # compute relative sizes for val/test split
    val_prop = val_size / combined_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_combined, y_combined, test_size=1 - val_prop, stratify=y_combined,
        random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_logistic_pipeline(max_features: int = 20000) -> Pipeline:
    """Construct a TF‑IDF + logistic regression pipeline.

    Parameters
    ----------
    max_features : int, optional
        Maximum number of features for the TF‑IDF vectoriser. Raising this
        value can improve performance on large datasets at the cost of
        memory.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The configured pipeline.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            stop_words=None,
            ngram_range=(1, 2),
            max_features=max_features
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            solver='lbfgs'
        ))
    ])


def build_svm_pipeline(max_features: int = 20000) -> Pipeline:
    """Construct a TF‑IDF + linear SVM pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            stop_words=None,
            ngram_range=(1, 2),
            max_features=max_features
        )),
        ('clf', LinearSVC())
    ])


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics for a set of predictions.

    Returns accuracy, precision, recall and F1 score (weighted by
    support).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_and_evaluate_model(model: Pipeline,
                             X_train: pd.Series,
                             y_train: pd.Series,
                             X_val: pd.Series,
                             y_val: pd.Series) -> Tuple[Pipeline, Dict[str, float]]:
    """Fit a pipeline on the training data and evaluate on the validation set."""
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    metrics = evaluate_predictions(y_val, y_pred_val)
    return model, metrics


def train_bert(df: pd.DataFrame,
               text_column: str,
               label_column: str,
               model_name: str = 'camembert-base',
               output_dir: str = 'models/bert_model',
               num_train_epochs: int = 3,
               batch_size: int = 8,
               lr: float = 2e-5,
               weight_decay: float = 0.01) -> Dict[str, float]:
    """Fine‑tune a transformer model for sequence classification.

    This function uses Hugging Face's ``transformers`` and ``datasets``
    libraries. It splits the data internally into train/validation/test
    sets and returns evaluation metrics on the validation set. The
    trained model and tokenizer are saved to ``output_dir``.

    Note that training a transformer can be computationally intensive.
    When using this function ensure you have sufficient resources and
    consider using a GPU for acceleration.
    """
    try:
        from datasets import Dataset
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  Trainer, TrainingArguments)
    except ImportError as e:
        raise ImportError(
            "transformers and datasets libraries are required for BERT training."
        ) from e

    # Map labels to integers
    labels = sorted(df[label_column].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Create train/val/test splits
    processed_df = df[[text_column, label_column]].copy()
    processed_df['label_id'] = processed_df[label_column].map(label2id)
    X_train, X_val, X_test, _, _, _ = split_data(
        processed_df, text_column, 'label_id'
    )

    train_ds = Dataset.from_pandas(pd.DataFrame({
        text_column: X_train,
        'label': df.loc[X_train.index, label_column].map(label2id).values
    }))
    val_ds = Dataset.from_pandas(pd.DataFrame({
        text_column: X_val,
        'label': df.loc[X_val.index, label_column].map(label2id).values
    }))
    # We won't evaluate on test here; evaluation can be done separately

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding='max_length', truncation=True, max_length=128
        )

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        save_total_limit=2,
        report_to=[]
    )

    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        metrics = evaluate_predictions(labels, predictions)
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn
    )

    trainer.train()
    # Evaluate on validation set
    eval_results = trainer.evaluate()

    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return eval_results


def main(args: Optional[argparse.Namespace] = None) -> None:
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the CSV dataset with ``review`` and ``sentiment`` columns')
    parser.add_argument('--out_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--train_bert', action='store_true',
                        help='Whether to train a transformer model (can be slow)')
    parser.add_argument('--bert_model', type=str, default='camembert-base',
                        help='Pretrained transformer model to fine‑tune')
    parser.add_argument('--max_features', type=int, default=20000,
                        help='Maximum number of TF‑IDF features')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')

    if args is None:
        args = parser.parse_args()

    # Ensure output directory exists
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    df = load_dataset(args.data, text_column='review', label_column='sentiment')
    df_processed = preprocess_dataframe(df, text_column='review')

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_processed, 'clean_text', 'sentiment',
        test_size=0.15, val_size=0.15, random_state=args.random_state
    )

    # Train logistic regression
    log_reg_pipe = build_logistic_pipeline(max_features=args.max_features)
    log_reg_model, log_reg_metrics = train_and_evaluate_model(
        log_reg_pipe, X_train, y_train, X_val, y_val
    )
    # Train SVM
    svm_pipe = build_svm_pipeline(max_features=args.max_features)
    svm_model, svm_metrics = train_and_evaluate_model(
        svm_pipe, X_train, y_train, X_val, y_val
    )

    print('Validation metrics:')
    print('Logistic Regression:', log_reg_metrics)
    print('SVM:', svm_metrics)

    # Select best baseline model based on weighted F1
    best_baseline = 'logreg' if log_reg_metrics['f1'] >= svm_metrics['f1'] else 'svm'

    # Save baseline models
    joblib.dump(log_reg_model, out_dir / 'logreg_model.joblib')
    joblib.dump(svm_model, out_dir / 'svm_model.joblib')

    # Optionally train BERT
    if args.train_bert:
        print(f"Training transformer model '{args.bert_model}'...")
        bert_output_dir = out_dir / 'bert_model'
        bert_output_dir.mkdir(exist_ok=True)
        bert_metrics = train_bert(
            df_processed, 'review', 'sentiment', model_name=args.bert_model,
            output_dir=str(bert_output_dir)
        )
        print('Transformer validation metrics:', bert_metrics)
        # Compare models (baseline best vs BERT)
        # For comparability, use F1 from bert_metrics if available
        bert_f1 = bert_metrics.get('eval_f1') if 'eval_f1' in bert_metrics else 0
        if bert_f1 >= max(log_reg_metrics['f1'], svm_metrics['f1']):
            best_model_name = 'bert'
        else:
            best_model_name = best_baseline
    else:
        bert_f1 = None
        best_model_name = best_baseline

    # Report best model
    print(f"Best model based on validation F1: {best_model_name}")
    # Save a simple text file indicating the best model
    best_txt = out_dir / 'best_model.txt'
    with open(best_txt, 'w') as f:
        f.write(best_model_name)


if __name__ == '__main__':
    main()