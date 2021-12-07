"""
Run this module as `python main.py`
Args:
--mode
--train-path
--eval-path
--test-path
Mode can either be train or predict. In train mode, train path and eval
paths are necessary. In predict mode, test path is necessary.
"""


import argparse
from model import Model


FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]


def get_args():
    """Read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=False, default="predict")
    parser.add_argument("--train-path", type=str, required=False, default="data/train.csv")
    parser.add_argument("--eval-path", type=str, required=False, default="data/test.csv")
    parser.add_argument("--test-path", type=str, required=False, default="data/test.csv")
    known_arguments, unknown_arguments = parser.parse_known_args()
    known_arguments = {
        key: value for key, value in vars(known_arguments).items()
        if value is not None}
    return known_arguments, unknown_arguments


if __name__ == "__main__":
    args, unknown_args = get_args()
    train_path = args.get("train_path")
    eval_path = args.get("eval_path")
    test_path = args.get("test_path")
    mode = args.get("mode")

    model = Model()
    if mode == "train":
        input_df = model.spark.read.csv(train_path, header=True, inferSchema=True, sep=";")
        eval_df = model.spark.read.csv(eval_path, header=True, inferSchema=True, sep=";")
        model.train(input_df, eval_df, FEATURES)
    elif mode == "predict":
        test_df = model.spark.read.csv(test_path, header=True, inferSchema=True, sep=";")
        model.predict(test_df, FEATURES)
    else:
        print("Mode should be either train or predict. Given: %s.", mode)
