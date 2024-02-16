import random

from LUKE_utils import boi1_to_2, round_preds

random.seed(42)

path = "./outputs100/luke/output_2023.txt"
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
majorities = [pred_func.majority(r, top_k=9, pad="PAD") for r in randoms]
ratios = [pred_func.get_ratio(r) for r in randoms]

round_prediction = [
    pred_func.upper_bound(la, ra, const=co, random_num=3) for ra, co, la in zip(majorities, consts, labels)
]
# print(seqeval.metrics.classification_report([labels], [round_prediction], digits=4))

maj_count = [0, 0, 0, 0, 0, 0, 0, 0]
miss_count = 0
for majority, const, label in zip(majorities, consts, labels):
    flag = False
    for i in range(8):
        if label == majority[i]:
            maj_count[i] += 1
            flag = True
    if not flag:
        miss_count += 1

print("all data majority count: ", maj_count, " and miss num: ", miss_count)

maj_count = [0, 0, 0, 0, 0, 0, 0, 0]
for majority, const, label in zip(majorities, consts, labels):
    if const != label:
        for i in range(8):
            if label == majority[i]:
                maj_count[i] += 1
print("const miss data majority count: ", maj_count)

for tag in tags.keys():
    maj_count = [0, 0, 0, 0, 0, 0, 0, 0]
    for majority, const, label in zip(majorities, consts, labels):
        if label == tag:
            for i in range(8):
                if label == majority[i]:
                    maj_count[i] += 1
    print(f"{tag} data majority count: ", maj_count)
