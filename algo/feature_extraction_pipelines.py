# -*- coding: utf-8 -*-

"""
Custom feature extraction pipelines

__author__ : Koushik Khan [koushikkhan38@gmail.com]
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")


# ------------ Set Path ------------- #
home_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(home_path, "data")
algo_path = os.path.join(home_path, "algo")
model_path = os.path.join(home_path, "model")
sys.path.append(home_path)

# Import custom classes
from algo.data_transformers import CleanTextData, CreateDummies

# Predictor and Target variables
PREDICTOR = ["description", "component"]
TARGET = "ticketcategory"


def generate_feature_and_model_pipeline(source_data_fname, n_gram_size=1, svd_n_components=20,
                                     classifier="dtree", out_file_name=None, save_pipe=False):
    """
    # Arguments
        source_data_fname: source datafile name
        classifier: name of the classifier to be used, default is DecisionTree Classifier
        out_file_name: File name where the final pipeline has to saved 
        save_pipe: If True, final pipeline will be saved to disk, else it will
                   be returned as in-memory object

    # Returns
        scikit learn pipeline object
    """

    try:
        src_df = pd.read_csv(os.path.join(data_path, source_data_fname))
    except FileNotFoundError:
        print("Source datafile not found!")
        sys.exit(1)

    clf_map = {"dtree":DecisionTreeClassifier, "mnb":MultinomialNB, "svc":SVC}

    # For 'description' text
    pipe_1 = Pipeline(
        [
            ('cleaning', CleanTextData('description')),
            ('vect', TfidfVectorizer(ngram_range=(1, n_gram_size))),
            ('reduce_dim', TruncatedSVD(n_components=svd_n_components))
        ]
    )

    # For 'component' variable
    pipe_2 = Pipeline(
        [
            ('get_dummy', CreateDummies('component'))
        ]
    )

    # Create FeatureUnion Pipeline
    feature_union_pipe = FeatureUnion(
        [
            ('tfidf_feature_description', pipe_1),
            ('dummy_feature_component', pipe_2)
        ]
    )

    # Pipeline for feeding features into classifier
    clf_pipe = Pipeline(
        [
            ('feature_union', feature_union_pipe),
            ('clf', clf_map[classifier]())
        ]
    )

    # create predictor and target
    X = src_df[PREDICTOR]
    y = src_df[TARGET]

    # Fit pipeline to data
    print("INFO: Started fitting classification pipeline on training data.\n")
    
    clf_pipe.fit(X, y)
    
    print("INFO: Training completed.\n")

    if save_pipe:        
        if out_file_name is None:
            with open(os.path.join(model_path, "classification_pipeline.pkl"), "wb") as clf_pipeline_f:
                pickle.dump(clf_pipe, clf_pipeline_f)
        else:
            with open(os.path.join(model_path, out_file_name + ".pkl"), "wb") as clf_pipeline_f:
                pickle.dump(clf_pipe, clf_pipeline_f)
        print("INFO: Fitted classification pipeline has been saved.\n")

        return None
    else:
        return clf_pipe


if __name__ == "__main__":
    model = generate_feature_and_model_pipeline(source_data_fname="ticket_data_sp.csv")

    # load test data
    test_df = pd.read_csv(os.path.join(data_path, "test_data.csv"))
    y_test = test_df['labels'].tolist()

    # predict using model
    try:
        pred = model.predict(test_df)
        print("Predicted Categories: ", pred)
        print("Actual Test Categories: ", y_test)
        print("Accuracy Score: {}".format(accuracy_score(y_test, pred)))
        report = classification_report(y_test, pred)
        with open(os.path.join(model_path, "classification_report.txt"), "w") as clf_rep_f:
            clf_rep_f.writelines(report)

    except Exception as e:
        print("Error: {}".format(e))
        exit(2)