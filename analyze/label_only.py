import os
import random
import sys
from itertools import chain

import lightgbm as lgb
import numpy as np
import seqeval.metrics
import sklearn_crfsuite
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from luke.LUKE_utils import get_lgb_dataset
from utils.utils import path_to_data

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
tags = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}


def main():
    path = "./outputs100/output_2023.txt"

    df = get_lgb_dataset(path)

    # dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conll2023.txt")
    # dataset = load_dataset("conll2003")
    # dataset = dataset["validation"]

    light_gbm(df)
    # crf_train(df,dataset)


def light_gbm(df):
    df_train, df_val = train_test_split(df, test_size=0.2, shuffle=False)

    train_y = df_train["new_label"]
    train_x = df_train.drop("new_label", axis=1).drop("label", axis=1)

    val_y = df_val["new_label"]
    val_x = df_val.drop("new_label", axis=1).drop("label", axis=1)

    weight = compute_sample_weight("balanced", train_y).astype("float32")
    train_set = lgb.Dataset(train_x, train_y, weight=weight)
    test_set = lgb.Dataset(val_x, val_y)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        # "num_class": 9,
        "learning_rate": 0.0005,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=10000,
        valid_sets=[train_set, test_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(period=50),
        ],
        # feval=f1,
    )

    path = "./outputs100/output_valid.txt"
    df = get_lgb_dataset(path)
    labels = df["label"].to_numpy()
    df = df.drop("new_label", axis=1).drop("label", axis=1)
    test_pred = (model.predict(df) > 0.5).astype(int)
    test_pred_str = []
    label_str = []
    for t_i, t_p, label in zip(df.to_dict(orient="records"), test_pred, labels):
        t_p = t_i["const"] if t_p == "0" else t_i["maj2"]
        t_p = [tag for tag, id in tags.items() if id == t_p][0]
        label = [tag for tag, id in tags.items() if id == label][0]
        test_pred_str.append(t_p)
        label_str.append(label)
    print(seqeval.metrics.classification_report([label_str], [test_pred_str], digits=4))

    path = "./outputs100/output_test.txt"
    df = get_lgb_dataset(path)
    labels = df["label"].to_numpy()
    df = df.drop("new_label", axis=1).drop("label", axis=1)
    test_pred = (model.predict(df) > 0.5).astype(int)
    test_pred_str = []
    label_str = []
    for t_i, t_p, label in zip(df.to_dict(orient="records"), test_pred, labels):
        t_p = t_i["const"] if t_p == "0" else t_i["maj2"]
        t_p = [tag for tag, id in tags.items() if id == t_p][0]
        label = [tag for tag, id in tags.items() if id == label][0]
        test_pred_str.append(t_p)
        label_str.append(label)

    print(seqeval.metrics.classification_report([label_str], [test_pred_str], digits=4))


def crf_dataset(dataset, df):
    tokens = dataset["tokens"]
    x = df.drop("new_label", axis=1).drop("label", axis=1).to_dict(orient="records")
    y = df["label"].astype("str").to_list()
    new_y = df["new_label"].astype("str").to_list()

    dataset = []
    labels = []
    new_labels = []
    pred_pos = 0
    for token in tokens:
        dataset.append(x[pred_pos : pred_pos + len(token)])
        labels.append(y[pred_pos : pred_pos + len(token)])
        new_labels.append(new_y[pred_pos : pred_pos + len(token)])
        pred_pos += len(token)
    return dataset, labels, new_labels


def crf_train(df, dataset):
    df_x, df_y, df_new_y = crf_dataset(dataset, df)
    train_x, val_x, train_y, val_y, train_new_y, val_new_y = train_test_split(
        df_x, df_y, df_new_y, test_size=0.2, shuffle=False
    )

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=1000,
        all_possible_transitions=False,
    )
    crf.fit(train_x, train_new_y, val_x, val_new_y)

    test_pred = crf.predict(train_x)
    test_input = list(chain.from_iterable(train_x))
    test_pred = list(chain.from_iterable(test_pred))
    labels = list(chain.from_iterable(train_y))
    test_pred_str = []
    label_str = []
    for t_i, t_p, label in zip(test_input, test_pred, labels):
        t_p = t_i["const"] if t_p == "0" else t_i["maj2"]
        t_p = [tag for tag, id in tags.items() if id == int(t_p)][0]
        label = [tag for tag, id in tags.items() if id == int(label)][0]
        test_pred_str.append(t_p)
        label_str.append(label)

    train_result = seqeval.metrics.classification_report([label_str], [test_pred_str], digits=4).split("\n")

    test_pred = crf.predict(val_x)
    test_input = list(chain.from_iterable(val_x))
    test_pred = chain.from_iterable(test_pred)
    labels = chain.from_iterable(val_y)
    test_pred_str = []
    label_str = []

    for t_i, t_p, label in zip(test_input, test_pred, labels):
        t_p = t_i["const"] if t_p == "0" else t_i["maj2"]
        t_p = [tag for tag, id in tags.items() if id == int(t_p)][0]
        label = [tag for tag, id in tags.items() if id == int(label)][0]
        test_pred_str.append(t_p)
        label_str.append(label)
    valid_result = seqeval.metrics.classification_report([label_str], [test_pred_str], digits=4).split("\n")

    path = "./outputs100/output_test.txt"
    df = get_lgb_dataset(path)

    dataset = load_dataset("conll2003")
    dataset = dataset["test"]
    test_x, test_y, _ = crf_dataset(dataset, df)

    test_pred = crf.predict(test_x)
    test_input = list(chain.from_iterable(test_x))
    test_pred = chain.from_iterable(test_pred)
    labels = chain.from_iterable(test_y)
    test_pred_str = []
    label_str = []
    for t_i, t_p, label in zip(test_input, test_pred, labels):
        t_p = t_i["const"] if t_p == "0" else t_i["maj2"]
        t_p = [tag for tag, id in tags.items() if id == int(t_p)][0]
        label = [tag for tag, id in tags.items() if id == int(label)][0]
        test_pred_str.append(t_p)
        label_str.append(label)

    test_result = seqeval.metrics.classification_report([label_str], [test_pred_str], digits=4).split("\n")
    result = "\n".join([f"{tr}\t{va}\t{te}" for tr, va, te in zip(train_result, valid_result, test_result)])
    print(result)


if __name__ == "__main__":
    main()
