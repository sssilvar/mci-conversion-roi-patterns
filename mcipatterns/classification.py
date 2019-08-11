#!/bin/env python3
import os
from os.path import basename, dirname, join

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, auc, roc_curve, accuracy_score, precision_score, f1_score


def classify(features_file, subj_train, subj_test):
    # Load and arse data
    df = pd.read_csv(features_file, index_col=0)
    X = df.drop('label', axis='columns')
    y = df['label'].astype('category')

    X_train, X_test = X.reindex(subj_train).dropna(), X.reindex(subj_test).dropna()
    y_train, y_test = y.reindex(subj_train).dropna(), y.reindex(subj_test).dropna()

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Test the model
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 0]  # MCIc

    # Report the results
    report = classification_report(y_test, y_pred)
    print(report)

    # Plot ROC and save it
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='MCIc')

    metrics = pd.Series(name='metrics')
    metrics['acc'] = accuracy_score(y_test, y_pred)
    metrics['pre'] = precision_score(y_test, y_pred, pos_label='MCIc')
    metrics['f1'] = f1_score(y_test, y_pred, pos_label='MCIc')
    metrics['auc'] = auc(y_test, y_pred_proba)

    return metrics, fpr, tpr
