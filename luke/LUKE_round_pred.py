import seqeval.metrics
from LUKE_utils import round_preds

if __name__ == "__main__":
    path = "./outputs100/LUKE/Normal2023.txt"
    consts = []
    randoms = []
    labels = []
    with open(path) as f:
        data = f.read().split("\n")
        for line in data:
            if len(line) == 0:
                continue
            line = line.split(" ")
            labels.append(line[0])
            randoms.append(line[1:])
            consts.append(line[-1])
    pred_func = round_preds()
    round_prediction = [
        pred_func.upper_bound(la, ra, co, random_num=101) for ra, co, la in zip(randoms, consts, labels)
    ]
    print(seqeval.metrics.classification_report([labels], [round_prediction], digits=4))
