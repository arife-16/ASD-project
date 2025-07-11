## Project Overview

This project uses **brain imaging data** (fMRI time series) and a **brain map** (atlas/ROIs) to calculate how different brain areas interact (functional connectivity matrix).
This "brain fingerprint" is then combined with **patient information** (phenotypic data), preprocessed, and used to train machine learning models to predict whether someone has ASD.
The project also uses feature selection techniques, to find the most important brain connections for this prediction.
```mermaid
flowchart TD
    A0["fMRI Time Series
"]
    A1["Brain Atlas / ROIs (Regions of Interest)
"]
    A2["Functional Connectivity (FC) Matrix
"]
    A3["Phenotypic Data
"]
    A4["Preprocessing and Filtering
"]
    A5["Machine Learning Models (Logistic Regression, SVM, CNN, etc.)
"]
    A6["Model Evaluation (Cross-Validation, Metrics)
"]
    A7["Feature Selection (SelectKBest, RFECV)
"]
    A1 -- "Defines/Extracts" --> A0
    A4 -- "Filters/Processes" --> A0
    A4 -- "Filters/Processes" --> A3
    A0 -- "Calculates" --> A2
    A2 -- "Provides Features" --> A7
    A7 -- "Selects Features For" --> A5
    A3 -- "Provides Labels" --> A5
    A5 -- "Evaluates" --> A6
```

