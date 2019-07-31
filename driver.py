# -*- coding: utf-8 -*-

"""
Driver program to generate the model file

__author__ : Koushik Khan [koushikkhan38@gmail.com]
"""

import os
import sys
from configparser import RawConfigParser


# ------------ Set Path ------------- #
home_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(home_path, "data")
algo_path = os.path.join(home_path, "algo")
model_path = os.path.join(home_path, "model")
config_path = os.path.join(home_path, "config")
sys.path.append(home_path)

from algo.data_transformers import CleanTextData, CreateDummies
from algo.feature_extraction_pipelines import generate_feature_and_model_pipeline

config = RawConfigParser()
config.read(os.path.join(config_path, "config.ini"))
clf_id = config['model_building']['classifier']
n_gram_size = int(config['feature_generation']['n_gram_size'])
svd_n_components = int(config['feature_generation']['svd_n_components'])
source = config['data_source']['train_data']
# min_word_len = int(config['data_transformation']['min_word_len'])

# params = {
#             "source_data_fname":SOURCE, 
#             "classifier":clf_id,
#             "save_pipe":True
#          }
# SOURCE = "ticket_data_sp.csv"


if __name__ == "__main__":
    generate_feature_and_model_pipeline(source_data_fname=source, classifier=clf_id, n_gram_size=n_gram_size,
                                         svd_n_components=svd_n_components, save_pipe=True)