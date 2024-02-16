import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import seqeval.metrics
import torch
from bpe_dropout import LukeTokenizerDropout
from tqdm import tqdm, trange
from transformers import LukeForEntitySpanClassification

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
# Load the model checkpoint
model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
model.eval()
model.to("cuda")

# Load the tokenizer
tokenizer = LukeTokenizerDropout.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003", alpha=0.1)


def load_documents(dataset_file):
    documents = []
    words = []
    labels = []
    sentence_boundaries = []
    with open(dataset_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if words:
                    documents.append(
                        dict(
                            words=words,
                            labels=labels,
                            sentence_boundaries=sentence_boundaries,
                        )
                    )
                    words = []
                    labels = []
                    sentence_boundaries = []
                continue

            if not line:
                if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                    sentence_boundaries.append(len(words))
            else:
                items = line.split(" ")
                words.append(items[0])
                labels.append(items[-1])

    if words:
        documents.append(dict(words=words, labels=labels, sentence_boundaries=sentence_boundaries))

    return documents


def load_examples(documents, max_token_length=510):
    examples = []
    max_mention_length = 30
    tokenizer.const_tokenize()

    for document in tqdm(documents):
        words = document["words"]
        subword_lengths = [len(tokenizer.tokenize(w)) for w in words]
        total_subword_length = sum(subword_lengths)
        sentence_boundaries = document["sentence_boundaries"]

        for i in range(len(sentence_boundaries) - 1):
            sentence_start, sentence_end = sentence_boundaries[i : i + 2]
            if total_subword_length <= max_token_length:
                # if the total sequence length of the document is shorter than the
                # maximum token length, we simply use all words to build the sequence
                context_start = 0
                context_end = len(words)
            else:
                # if the total sequence length is longer than the maximum length, we add
                # the surrounding words of the target sentence　to the sequence until it
                # reaches the maximum length
                context_start = sentence_start
                context_end = sentence_end
                cur_length = sum(subword_lengths[context_start:context_end])
                while True:
                    if context_start > 0:
                        if cur_length + subword_lengths[context_start - 1] <= max_token_length:
                            cur_length += subword_lengths[context_start - 1]
                            context_start -= 1
                        else:
                            break
                    if context_end < len(words):
                        if cur_length + subword_lengths[context_end] <= max_token_length:
                            cur_length += subword_lengths[context_end]
                            context_end += 1
                        else:
                            break

            text = ""
            for word in words[context_start:sentence_start]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "

            sentence_words = words[sentence_start:sentence_end]
            sentence_subword_lengths = subword_lengths[sentence_start:sentence_end]

            word_start_char_positions = []
            word_end_char_positions = []
            for word in sentence_words:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                word_start_char_positions.append(len(text))
                text += word
                word_end_char_positions.append(len(text))
                text += " "

            for word in words[sentence_end:context_end]:
                if word[0] == "'" or (len(word) == 1 and is_punctuation(word)):
                    text = text.rstrip()
                text += word
                text += " "
            text = text.rstrip()

            entity_spans = []
            original_word_spans = []
            for word_start in range(len(sentence_words)):
                for word_end in range(word_start, len(sentence_words)):
                    if sum(sentence_subword_lengths[word_start : word_end + 1]) <= max_mention_length:
                        entity_spans.append(
                            (
                                word_start_char_positions[word_start],
                                word_end_char_positions[word_end],
                            )
                        )
                        original_word_spans.append((word_start, word_end + 1))

            examples.append(
                dict(
                    text=text,
                    words=sentence_words,
                    entity_spans=entity_spans,
                    original_word_spans=original_word_spans,
                )
            )

    return examples


def is_punctuation(char):
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


test_documents = load_documents("C:\\Users\\chenr\\Desktop\\python\\subword_regularization\\eng.testa")
test_examples = load_examples(test_documents, 410)
tokenizer.random_tokenize()

round_prediction = []
num_round = 1
for i in range(num_round):
    if i == num_round - 1:
        test_documents = load_documents("C:\\Users\\chenr\\Desktop\\python\\subword_regularization\\eng.testa")
        test_examples = load_examples(test_documents, 510)
        tokenizer.const_tokenize()
    batch_size = 2
    all_logits = []

    for batch_start_idx in trange(0, len(test_examples), batch_size):
        batch_examples = test_examples[batch_start_idx : batch_start_idx + batch_size]
        texts = [example["text"] for example in batch_examples]
        entity_spans = [example["entity_spans"] for example in batch_examples]

        while True:
            inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)

            if inputs["input_ids"].shape[-1] <= 512:
                break

        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        all_logits.extend(outputs.logits.tolist())

    final_labels = [label for document in test_documents for label in document["labels"]]

    final_predictions = []
    for example_index, example in enumerate(test_examples):
        logits = all_logits[example_index]
        max_logits = np.max(logits, axis=1)
        max_indices = np.argmax(logits, axis=1)
        original_spans = example["original_word_spans"]
        predictions = []
        for logit, index, span in zip(max_logits, max_indices, original_spans):
            if index != 0:  # the span is not NIL
                predictions.append((logit, span, model.config.id2label[index]))

        # construct an IOB2 label sequence
        predicted_sequence = ["O"] * len(example["words"])
        for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
            if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                predicted_sequence[span[0]] = "B-" + label
                if span[1] - span[0] > 1:
                    predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

        final_predictions += predicted_sequence
    round_prediction.append(final_predictions)

round_prediction = pd.DataFrame(round_prediction).T.values.tolist()


def majority(preds):
    count = Counter(preds)
    count = count.most_common()

    return count[0][0]


def upper_bound(label, preds):
    if label in preds:
        return label
    else:
        return majority(preds)


path = "./output_valid.txt"
with open(path, "w") as f_out:
    pred = "\n".join(f_l + " " + " ".join(r_p) for r_p, f_l in zip(round_prediction, final_labels))
    f_out.write(pred)

round_prediction = [upper_bound(f_l, r_p) for r_p, f_l in zip(round_prediction, final_labels)]

print(seqeval.metrics.classification_report([final_labels], [round_prediction], digits=4))
