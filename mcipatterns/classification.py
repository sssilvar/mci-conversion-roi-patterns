#!/bin/env python3
import os

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


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
        ('clf', RandomForestClassifier())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Test the model
    y_pred = pipeline.predict(X_test)

    # Report the results
    report = classification_report(y_test, y_pred)
    print(report)
