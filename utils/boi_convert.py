import copy


def boi2_to_1(labels):
    new_labels = []
    for label in labels:
        # ner = [O,B-LOC,I-LOC,I-LOC,B-LOC,I-LOC,B-MISC,B-MISC]
        pre_tag = -1
        new_ner = []
        for la in label:
            if la in [1, 3, 5, 7] and pre_tag not in [la + 1, la]:
                pre_tag = copy.deepcopy(la)
                la += 1
            else:
                pre_tag = copy.deepcopy(la)
            new_ner.append(copy.deepcopy(la))
        # new_ner = [O,I-LOC,I-LOC,I-LOC,B-LOC,I-LOC,I-MISC,B-MISC]
        new_labels.append(new_ner)

    return new_labels


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


if __name__ == "__main__":
    labels = ["I-ORG", "I-ORG", "O", "I-ORG", "I-MISC", "I-LOC"]
    print(boi1_to_2(labels))
