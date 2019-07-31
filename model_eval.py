# -*- coding: utf-8 -*-

"""
Model Evaluation

__author__ : Koushik Khan [koushikkhan38@gmail.com]
"""

import os
import sys
import pickle
import argparse
import pandas as pd

# ------- Set Path --------- #
home_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(home_path, "data")
algo_path = os.path.join(home_path, "algo")
model_path = os.path.join(home_path, "model")
sys.path.append(home_path)


def main():
    # Read positional and optional arguments from commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='file containing the input data')
    args = parser.parse_args()

    # if args.input is not None:
    try:
        src_path = str(os.path.join(data_path, args.input))
        # print("debug:", src_path)
        input_df = pd.read_csv(src_path)
    except FileNotFoundError:
        print('Input file not found!')
        sys.exit(1)


    # create feature for test data
    # X_test = create_feature(corpus=input_text, mode="test", gram_size=2)
    test_df = input_df[['description', 'component']]
    
    try:
        with open(os.path.join(model_path, "classification_pipeline.pkl"), "rb") as clf_pipe_f:
            clf_pipe = pickle.load(clf_pipe_f)
    except FileNotFoundError:
        print("classification pipeline not found!")
        sys.exit(3)

    # load model
    # with open(os.path.join(model_path, "model.pkl"), "rb") as model_f:
    #     model = pickle.load(model_f)

    # predict new labels
    # labels_pred = clf_pipe.predict(test_df)
    formatted_output = """{}""".format("\n".join(clf_pipe.predict(test_df)))

    with open(os.path.join(data_path, "output.txt"), 'w') as out_f:
        out_f.writelines(formatted_output)

    return None


if __name__ == "__main__":
    main()