import os
from collections import Counter
from itertools import chain
from logging import getLogger

import hydra
import seqeval.metrics

root_path = os.getcwd()
logger = getLogger(__name__)


def majority(preds):
    count = Counter(preds)
    count = count.most_common()

    return count[0][0]


def upper_bound(label, preds):
    if label in preds:
        return label
    else:
        return majority(preds)


def get_const(preds):
    return preds[-1]


def round_table(file_iter, vote="majority"):
    eval_preds = []
    eval_labels = []

    eval_pred = []
    eval_label = []
    for line in file_iter:
        if line[-1:] == "\n":
            line = line[:-1]

        if len(line) < 5:
            if len(eval_pred) != 0:
                eval_preds.append(eval_pred)
                eval_labels.append(eval_label)
                eval_pred = []
                eval_label = []
            continue

        line = line.split(" ")
        label = line[1]
        preds = line[2:]

        if vote == "majority":
            pred = majority(preds)
        elif vote == "upper_bound":
            pred = upper_bound(label, preds)
        elif vote == "const":
            pred = get_const(preds)
        else:
            print("Vote Error")
            exit(1)
        eval_pred.append(pred)
        eval_label.append(label)

    eval_preds = list(chain.from_iterable(eval_preds))
    eval_labels = list(chain.from_iterable(eval_labels))
    logger.info("\n" + seqeval.metrics.classification_report([eval_labels], [eval_preds], digits=4))


def many_round_table(file_iters, model_names, vote="majority"):
    eval_preds = []
    eval_labels = []

    eval_pred = []
    eval_label = []
    for i, _ in enumerate(file_iters[0]):
        preds = []
        for model_name, file_iter in zip(model_names, file_iters):
            if file_iter[i][-1:] == "\n":
                file_iter[i] = file_iter[i][:-1]

            line = file_iter[i].split(" ")
            if model_name.startswith("LUKE"):
                # label = line[0]
                preds += line[1:]
            else:
                label = line[1]
                preds += line[2:]

        if vote == "majority":
            pred = majority(preds)
        elif vote == "upper_bound":
            pred = upper_bound(label, preds)
        elif vote == "const":
            pred = get_const(preds)
        else:
            print("Vote Error")
            exit(1)
        eval_pred.append(pred)
        eval_label.append(label)

    if len(eval_pred) != 0:
        eval_preds.append(eval_pred)
        eval_labels.append(eval_label)

    eval_preds = list(chain.from_iterable(eval_preds))
    eval_labels = list(chain.from_iterable(eval_labels))
    logger.info("\n" + seqeval.metrics.classification_report([eval_labels], [eval_preds], digits=4))


def main():
    input_pathes = [
        os.path.join(root_path, "outputs100\\BertB\\Regtest.txt"),
        os.path.join(root_path, "outputs100\\BertL\\Regtest\\many_preds.txt"),
        os.path.join(root_path, "outputs100\\RobertaB\\Regtest.txt"),
        os.path.join(root_path, "outputs100\\RobertaL\\Regtest.txt"),
        os.path.join(root_path, "outputs100\\BertB\\Normaltest.txt"),
        os.path.join(root_path, "outputs100\\BertL\\Normaltest\\many_preds.txt"),
        os.path.join(root_path, "outputs100\\RobertaB\\Normaltest.txt"),
        os.path.join(root_path, "outputs100\\RobertaL\\Normaltest.txt"),
        os.path.join(root_path, "outputs100\\BertB\\Reg3test.txt"),
        os.path.join(root_path, "outputs100\\BertL\\Reg3test\\many_preds.txt"),
        os.path.join(root_path, "outputs100\\RobertaB\\Reg3test.txt"),
        os.path.join(root_path, "outputs100\\RobertaL\\Reg3test.txt"),
        # os.path.join(root_path, "outputs100\\LUKE\\Normaltest.txt"),
    ]
    file_iters = []
    model_names = []
    for input_path in input_pathes:
        with open(input_path) as f:
            text = f.read()
        # with open("./input_pred.txt", "w") as f:
        #    f.write(text)

        text = text.replace("\n\n", "\n").replace("\n\n", "\n")[:-1]

        file_iters.append(text.split("\n"))
        model_names.append(input_path.split("\\")[-2])
    many_round_table(file_iters, model_names, vote="majority")


def main2():
    input_path = os.path.join(root_path, "outputs100\\RobertaL\\Normaltest.txt")

    with open(input_path) as f:
        text = f.readlines()
    round_table(text, vote="const")


if __name__ == "__main__":
    main2()
