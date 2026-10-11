# Master’s Thesis: Food Protein Allergenicity Prediction

## Overview 
This project was developed as part of my master's thesis in Bioengineering in Chemistry and Bio-industries. It explores the use of machine learning to predict food protein allergenicity by combining AlphaFold2-derived structural embeddings with sequence-similarity features based on FAO/WHO Codex Alimentarius criteria.

## Methods
* Protein representation using AlphaFold2 structural embeddings
* Feature engineering and integration of sequence-similarity features
* Class imbalance handling and feature selection (ANOVA, mutual information and RFE)
* Model training and cross-validation using SVM, Gradient Boosting and XGBoost
* Performance evaluation using adequate metrics

## Repository structure
* 0_Database/ —> Protein datasets and database preparation
* 1_Embeddings/ —> Structural embeddings
* 2_Pooling/ —> Pooling of embeddings into fixed-length representations
* 3_Handling_imbalance/ —> Handling class imbalance and related experiments
* 4A_FAO_features_calculation/ —> Calculation of FAO/WHO Codex sequence-similarity features
* 4B_FAO_feature_implementation/ —> Integration of the calculated features
* 4C_FAO_CV_training/ —> Model training and cross-validation
* 4D_FAO_performances/ —> Model performance evaluation on test sets

## Repository Contents
This repository contains the Python scripts, initial datasets used in the project and some results. Most intermediate files and generated data are not included.
