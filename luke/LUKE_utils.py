import random
from collections import Counter

import pandas as pd
from sklearn.metrics import f1_score

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


def boi1_to_2(labels):
    new_labels = []
    pre_tag = ""
    for label in labels:
        if label[0] == "I":
            if pre_tag in ["", "O"]:
                label = "B" + label[1:]
            elif len(pre_tag) > 4 and pre_tag[-3:] != label[-3:]:
                label = "B" + label[1:]

        new_labels.append(label)
        pre_tag = label

    return new_labels


def f1(y_pred, data):
    y_true = data.get_label()
    y_pred = y_pred.argmax(axis=1)
    score = f1_score(y_true, y_pred, average="micro")

    return "f1", score, True


class round_preds:
    def __init__(self) -> None:
        self.past_label = "O"

    def majority(self, preds, top_k=1, pad="TOP"):
        count = Counter(preds)
        count = count.most_common()
        if top_k == 1:
            return count[0][0]
        else:
            top_preds = [c[0] for c in count[:top_k]]
            if len(top_preds) != top_preds:
                pad = top_preds[0] if pad == "TOP" else pad
                top_preds += [pad] * (top_k - len(top_preds))
            return top_preds

    def upper_bound(self, label, preds, const=None, random_num=10):
        preds = random.sample(preds, k=random_num)
        if const is not None:
            preds.append(const)

        if label in preds:
            self.past_label = label
            return label
        elif label[0] == "I" and self.past_label != label:
            _label = "B" + label[1:]
            self.past_label = label
            if _label in preds:
                return _label
            else:
                return self.majority(preds)
        else:
            self.past_label = label
            return self.majority(preds)

    def get_ratio(self, preds):
        count = Counter(preds)
        ratio = [count[t] / len(preds) for t in tags]

        return ratio


def get_lgb_dataset(path):
    consts = []
    randoms = []
    labels = []
    with open(path) as f:
        data = f.read().split("\n")
        for line in data:
            line = line.split(" ")
            labels.append(line[0])
            randoms.append(line[1:-1])
            consts.append(line[-1])
    labels = boi1_to_2(labels)

    pred_func = round_preds()
    ratios = [pred_func.get_ratio(r) for r in randoms]
    majorities = [pred_func.majority(r, top_k=3) for r in randoms]

    df = {"const": [], "maj1": [], "maj2": [], "maj3": [], "label": [], "new_label": []}
    for i in range(9):
        df[f"ratio{i}"] = []

    for majority, ratio, const, label in zip(majorities, ratios, consts, labels):
        df["const"].append(tags[const])
        df["maj1"].append(tags[majority[0]])
        df["maj2"].append(tags[majority[1]])
        df["maj3"].append(tags[majority[2]])
        for i in range(9):
            df[f"ratio{i}"].append(ratio[i])
        df["label"].append(tags[label])
        df["new_label"].append(0 if label == const else 1)
    df = pd.DataFrame(df)

    return df
