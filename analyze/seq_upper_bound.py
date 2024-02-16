import pickle

import numpy as np
import seqeval.metrics


def main():
    dataset_name = "valid"
    with open(f"./outputs100/dataset_{dataset_name}.pkl", mode="rb") as f:
        dataset = pickle.load(f)

    pred = {}
    predicted_labels = []
    golden_labels = []

    for sentence_id, rate, predicted_label, golden_label in zip(
        dataset["Sentence_id"],
        dataset["rate"],
        dataset["Predicted_label"],
        dataset["Golden_label"],
    ):
        if sentence_id not in pred.keys():
            pred[sentence_id] = {
                "Golden_label": golden_label,
                "prob": [],
                "Predicted_label": [],
            }
        pred[sentence_id]["Predicted_label"].append(predicted_label)
        pred[sentence_id]["prob"].append(rate)

    for p in pred.values():
        max_prob_index = np.argmax(np.array(p["prob"]))
        final_pred = p["Predicted_label"][max_prob_index]
        predicted_labels += final_pred
        golden_labels += p["Golden_label"]

    print(seqeval.metrics.classification_report([golden_labels], [predicted_labels], digits=4))


if __name__ == "__main__":
    main()
