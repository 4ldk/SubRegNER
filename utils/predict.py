import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.round_table import round_table
from utils.utils import (
    dataset_encode,
    get_dataloader,
    path_to_data,
    val_to_key,
)

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

model_dict = {
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


def loop_pred(
    length,
    model_name,
    test,
    tokenizer,
    loop=100,
    batch_size=4,
    p=0.1,
    vote="majority",
    local_model=False,
    post_sentence_padding=False,
    add_sep_between_sentences=False,
    device="cuda",
    output_path="./many_preds.txt",
):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    if test == "2003":
        test_dataset = path_to_data(os.path.join(root_path, "test_datasets/eng.testb"))
        encoding = "utf-8"

    elif test == "valid":
        test_dataset = path_to_data(os.path.join(root_path, "test_datasets/eng.testa"))
        encoding = "utf-8"

    elif test == "2023":
        test_dataset = path_to_data(os.path.join(root_path, "test_datasets/conllpp.txt"))
        encoding = "utf-8"

    elif test == "crossweigh":
        test_dataset = path_to_data(os.path.join(root_path, "test_datasets/conllcw.txt"))
        encoding = "utf-8"

    if local_model:
        if not os.path.exists(local_model):
            local_model = os.path.join(root_path, local_model)
        config = AutoConfig.from_pretrained(model_name, num_labels=len(ner_dict))
        model = AutoModelForTokenClassification.from_config(config)
        model.load_state_dict(torch.load(local_model))
        model = model.to(device)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

    model.eval()
    outputs = []
    for i in tqdm(range(loop)):
        if i == loop - 1:
            p = 0

        output = pred(
            tokenizer,
            test_dataset,
            model,
            model_name,
            device,
            batch_size=batch_size,
            length=length,
            p=p,
            post_sentence_padding=post_sentence_padding,
            add_sep_between_sentences=add_sep_between_sentences,
        )
        outputs.append(output)

    joined_out = []
    for index in range(len(outputs[0])):
        if len(outputs[0][index]) < 5:
            joined_out.append("")
        else:
            pred_ners = []
            for output in outputs:
                token, ner, pred_ner = output[index].split(" ")
                if len(pred_ners) == 0:
                    pred_ners.append(token)
                    pred_ners.append(ner)
                pred_ners.append(pred_ner)
            pred_ners = " ".join(pred_ners)
            joined_out.append(pred_ners)

    joined_out = "\n".join(joined_out)

    with open(output_path, "w", encoding=encoding) as f_out:
        f_out.write(joined_out)

    with open(output_path, encoding=encoding) as f:
        file_iter = f.readlines()
    round_table(file_iter, vote)


def pred(
    tokenizer,
    test_dataset,
    model,
    model_name,
    device="cuda",
    batch_size=4,
    length=512,
    p=0,
    post_sentence_padding=False,
    add_sep_between_sentences=False,
):
    encoded_dataset = dataset_encode(
        tokenizer,
        test_dataset,
        p=p,
        padding=length,
        subword_label="PAD",
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
    )
    dataloader = get_dataloader(encoded_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    labels, out_tokens, out_labels = (
        encoded_dataset["predict_labels"].tolist(),
        encoded_dataset["tokens"],
        encoded_dataset["labels"],
    )

    output = []
    with torch.no_grad():
        for i, (input, mask, type_ids, label, _) in enumerate(tqdm(dataloader, leave=False)):
            input, mask = (
                input.to(device),
                mask.to(device),
            )
            if model_name.startswith("bert-"):
                preds = model(input, mask, type_ids.to(device)).logits.argmax(-1).to("cpu").tolist()
            else:
                preds = model(input, mask).logits.argmax(-1).to("cpu").tolist()
            for j, pred in enumerate(preds):
                label = labels[i * batch_size + j]
                out_label = out_labels[i * batch_size + j]
                out_token = out_tokens[i * batch_size + j]

                pred = [val_to_key(prd, model_dict) for (prd, lbl) in zip(pred, label) if lbl != ner_dict["PAD"]]
                pred = [c if c != "PAD" else "O" for c in pred]

                out_label = [val_to_key(o_n, ner_dict) for o_n in out_label]
                out = [" ".join([t, c, p]) for t, c, p in zip(out_token, out_label, pred)]
                out = "\n".join(out)
                output.append(out)
                output.append("\n\n")
    output = "".join(output).split("\n")

    return output
