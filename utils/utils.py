import copy
import os
import sys

import torch
from seqeval import metrics
from sklearn.utils import compute_sample_weight
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from boi_convert import boi1_to_2

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""
root_path = os.getcwd()
ner_dict = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
    "PAD": 9,
}

subword_dict = {
    0: 0,
    1: 2,
    2: 2,
    3: 4,
    4: 4,
    5: 6,
    6: 6,
    7: 8,
    8: 8,
    9: 9,
}

t5_sp_token_dict = {
    "<PER>": "<extra_id_0>",
    "<ORG>": "<extra_id_1>",
    "<LOC>": "<extra_id_2>",
    "<MISC>": "<extra_id_3>",
    "</PER>": "<extra_id_4>",
    "</ORG>": "<extra_id_5>",
    "</LOC>": "<extra_id_6>",
    "</MISC>": "<extra_id_7>",
}


class BertDataset(Dataset):
    def __init__(self, data) -> None:
        X, mask, type_ids, y, predict_y = (
            data["input_ids"],
            data["attention_mask"],
            data["token_type_ids"],
            data["subword_labels"],
            data["predict_labels"],
        )
        super().__init__()
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        if type(mask) is not torch.Tensor:
            mask = torch.tensor(mask)
        if type(type_ids) is not torch.Tensor:
            type_ids = torch.tensor(type_ids)
        if type(y) is not torch.Tensor:
            y = torch.tensor(y)
        if type(predict_y) is not torch.Tensor:
            predict_y = torch.tensor(predict_y)

        self.X = X
        self.mask = mask
        self.type_ids = type_ids
        self.y = y
        self.predict_y = predict_y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.type_ids[idx], self.y[idx], self.predict_y[idx]


class MultiViewDataset(Dataset):
    def __init__(self, datasets) -> None:
        super().__init__()
        self.datasets = datasets
        # 0: const
        # 1: random

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)


def recall_score(target, pred, average="micro", skip=-1):
    new_target = []
    new_pred = []
    for p, t in zip(pred, target):
        if t != skip:
            new_target.append(val_to_key(t, ner_dict))
            p = p if p != skip else 0
            new_pred.append(val_to_key(p, ner_dict))

    return metrics.recall_score([new_target], [new_pred], average=average)


def precision_score(target, pred, average="micro", skip=-1):
    new_target = []
    new_pred = []
    for p, t in zip(pred, target):
        if t != skip:
            new_target.append(val_to_key(t, ner_dict))
            p = p if p != skip else 0
            new_pred.append(val_to_key(p, ner_dict))

    return metrics.precision_score([new_target], [new_pred], average=average)


def f1_score(target, pred, skip=-1):
    new_target = []
    new_pred = []
    for p, t in zip(pred, target):
        if t != skip:
            new_target.append(val_to_key(t, ner_dict))
            p = p if p != skip else 0
            new_pred.append(val_to_key(p, ner_dict))

    return metrics.f1_score([new_target], [new_pred])


def get_texts_and_labels(dataset):
    tokens = dataset["tokens"]
    labels = dataset["ner_tags"]
    data = {
        "tokens": tokens,
        "labels": labels,
    }

    return data


def get_dataloader(data, batch_size, shuffle=True, drop_last=True):
    dataset = BertDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


def get_mv_dataloader(const_data, random_data, batch_size, shuffle=True, drop_last=True):
    datasets = [BertDataset(d) for d in [const_data, random_data]]
    dataset = MultiViewDataset(datasets)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


def key_to_val(key, dic):
    return dic[key] if key in dic else dic["UNK"]


def val_to_key(val, dic, pad_key="PAD"):
    keys = [k for k, v in dic.items() if v == val]
    if keys:
        return keys[0]
    return "UNK"


def path_to_data(path):
    with open(path, "r", encoding="utf-8") as f:
        row_data = f.readlines()

    data = []
    doc_index = []
    tokens = []
    labels = []
    replaced_labels = []
    pre_doc_end = 0
    for line in row_data:
        if "-DOCSTART-" in line:
            if len(tokens) != 0:
                labels = [key_to_val(la, ner_dict) for la in replaced_labels]
                document = dict(tokens=tokens, labels=labels, doc_index=doc_index)
                data.append(document)
                tokens = []
                replaced_labels = []
                labels = []
                doc_index = []
                pre_doc_end = 0

        elif len(line) <= 5:
            if len(labels) != 0:
                doc_start = pre_doc_end
                doc_end = len(tokens)
                doc_index.append((doc_start, doc_end))

                pre_doc_end = doc_end
                replaced_labels += boi1_to_2(labels)
                labels = []
        else:
            line = line.strip().split()

            tokens.append(line[0])
            labels.append(line[-1])

    if len(tokens) != 0:
        labels = [key_to_val(la, ner_dict) for la in replaced_labels]
        document = dict(tokens=tokens, labels=labels, doc_index=doc_index)
        data.append(document)
    return data


def get_label(word_ids, label, subword_label):
    previous_word_idx = -100
    label_ids = []
    for word_idx in word_ids:
        if word_idx == -100:
            label_ids.append(ner_dict["PAD"])
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(label[word_idx])
        else:
            if subword_label == "I":
                label_ids.append(subword_dict[label[word_idx]])
            elif subword_label == "B":
                label_ids.append(label[word_idx])
            elif subword_label == "PAD":
                label_ids.append(ner_dict["PAD"])
            else:
                print("subword_label must be 'I', 'B' or 'PAD'.")
                exit(1)

        previous_word_idx = word_idx
    return label_ids


def dataset_encode(
    tokenizer,
    data,
    p=None,
    padding=512,
    return_tensor=True,
    subword_label="I",
    post_sentence_padding=False,
    add_sep_between_sentences=False,
):
    if tokenizer.__class__.__name__ == "MaxMatchTokenizer":
        return tokenizer.dataset_encode(
            data,
            p=p,
            post_sentence_padding=post_sentence_padding,
            add_sep_between_sentences=add_sep_between_sentences,
        )
    if p is None or p == 0:
        tokenizer.const_tokenize()
    else:
        tokenizer.random_tokenize()

    row_tokens = []
    row_labels = []
    input_ids = []
    attention_mask = []
    subword_labels = []
    predict_labels = []
    weight_y = []
    for document in data:
        text = document["tokens"]
        labels = document["labels"]
        max_length = padding - 2 if padding else len(text)

        for i, j in document["doc_index"]:
            subwords, word_ids = tokenizer.tokenzizeSentence(" ".join(text[i:j]))
            row_tokens.append(text[i:j])
            row_labels.append(labels[i:j])
            masked_ids = copy.deepcopy(word_ids)

            if post_sentence_padding:
                while len(subwords) < max_length and j < len(text):
                    if add_sep_between_sentences and j in [d[0] for d in document["doc_index"]]:
                        subwords.append(tokenizer.sep_token)
                        word_ids.append(-100)
                        masked_ids.append(-100)
                    ex_subwords = tokenizer.tokenize(" " + text[j])
                    subwords = subwords + ex_subwords
                    word_ids = word_ids + [max(word_ids) + 1] * len(ex_subwords)
                    masked_ids = masked_ids + [-100] * len(ex_subwords)
                    j += 1
                    if len(subwords) < max_length:
                        subwords = subwords[:max_length]
                        word_ids = word_ids[:max_length]
                        masked_ids = masked_ids[:max_length]
            subwords = (
                [tokenizer.cls_token_id]
                + [tokenizer._convert_token_to_id(w) for w in subwords]
                + [tokenizer.sep_token_id]
            )
            word_ids = [-100] + word_ids + [-100]
            masked_ids = [-100] + masked_ids + [-100]

            if len(subwords) >= padding:
                subwords = subwords[:padding]
                word_ids = word_ids[:padding]
                masked_ids = masked_ids[:padding]
                mask = [1] * padding

            else:
                attention_len = len(subwords)
                pad_len = padding - len(subwords)
                subwords += [tokenizer.pad_token_id] * pad_len
                word_ids += [-100] * pad_len
                masked_ids += [-100] * pad_len
                mask = [1] * attention_len + [0] * pad_len

            input_ids.append(subwords)
            attention_mask.append(mask)

            label = labels[i:j]
            label_ids = get_label(word_ids, label, subword_label)
            subword_labels.append(label_ids)

            masked_label = row_labels[-1]
            masked_label_ids = get_label(masked_ids, masked_label, "PAD")
            predict_labels.append(masked_label_ids)

            weight_y += [l_i for l_i in label_ids if l_i != ner_dict["PAD"]]

    loss_rate = compute_sample_weight("balanced", y=weight_y)
    weight = [loss_rate[weight_y.index(i)] for i in range(len(set(weight_y)))]
    weight.append(0)

    tokenizer.random_tokenize()

    if return_tensor:
        data = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int),
            "subword_labels": torch.tensor(subword_labels, dtype=torch.long),
            "predict_labels": torch.tensor(predict_labels, dtype=torch.long),
            "tokens": row_tokens,
            "labels": row_labels,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }
        data["token_type_ids"] = torch.zeros_like(data["attention_mask"])
    else:
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "subword_labels": subword_labels,
            "predict_labels": predict_labels,
            "tokens": row_tokens,
            "labels": row_labels,
            "weight": weight,
        }
    return data
