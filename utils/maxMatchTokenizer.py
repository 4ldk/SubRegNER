import copy
import random

import torch
from sklearn.utils.class_weight import compute_sample_weight
from transformers import AutoTokenizer


class MaxMatchTokenizer:
    def __init__(
        self,
        vocab=None,
        midPref="##",
        headPref="",
        p=0.0,
        padding=False,
        ner_dict=None,
    ):
        self.midPref = midPref
        self.headPref = headPref
        self.doNaivePreproc = False
        if vocab:
            self.__build(vocab)

        self.p = p
        self.padding = padding
        self.ner_dict = ner_dict
        self.subword_dict = {
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
        }  # Chage from subword B-xxx label to I-xxx label

    def __build(self, vocab):
        self.unkToken = "[UNK]"
        self.vocab = set(vocab)
        self.vocab.add(self.unkToken)
        self.vocabSize = len(self.vocab)

        self.maxLength = max(len(w) for w in self.vocab)

        self.word2id = {}
        self.id2word = {}
        for w in sorted(self.vocab):
            self.word2id[w] = len(self.word2id)
            self.id2word[self.word2id[w]] = w

    # This function corresponds to Algorithm 1 in the paper.
    def tokenizeWord(self, word, p=None):
        p = p if p is not None else self.p

        subwords = []
        i = 0
        wordLength = len(word)
        while i < wordLength:
            subword = None
            for j in range(1, min(self.maxLength + 1, wordLength - i + 1)):
                w = word[i : i + j]

                if 0 == i:
                    w = self.headPref + w
                if 0 < i:
                    w = self.midPref + w

                if w in self.vocab:
                    # random for subword regularization
                    if j == 1 or p < random.random():
                        # drop acception with p
                        subword = w

            if subword is None:
                # return unk if including unk
                return [self.unkToken]
            else:
                i += len(subword) - len(self.midPref) if 0 < i else len(subword) - len(self.headPref)
                subwords.append(subword)
        return subwords

    def tokenize(self, text, p=None):
        p = p if p is not None else self.p
        if type(text) is list:
            return [self.tokenize(line, p) for line in text]
        if self.doNaivePreproc:
            text = self.naivePreproc(text)
            subwords = []
            for i, word in enumerate(text.split()):
                subword = self.tokenizeWord(word, p)
                for sw in subword:
                    subwords.append((i, sw))

        else:
            subwords = [(i, subword) for i, word in enumerate(text.split()) for subword in self.tokenizeWord(word, p)]
        word_ids = [sw[0] for sw in subwords]
        subwords = [sw[1] for sw in subwords]
        return subwords, word_ids

    def encode(self, text, p=None):
        p = p if p is not None else self.p
        if type(text) is list:
            subwords = [self.clsTokenId] + [
                self.word2id[w] for line in text for w in self.tokenize(line, p)[0] + [self.sepToken]
            ]
            word_ids = [-100] + [self.tokenize(line, p)[1] for line in text + [-100]]

            return subwords, word_ids
        subwords, word_ids = self.tokenize(text, p)
        subwords = [self.clsTokenId] + [self.word2id[w] for w in subwords] + [self.sepTokenId]
        word_ids = [-100] + word_ids + [-100]
        if self.padding:
            if len(subwords) >= self.padding:
                subwords = subwords[: self.padding]
                word_ids = word_ids[: self.padding]
                attention_mask = [1] * self.padding
            else:
                attention_len = len(subwords)
                pad_len = self.padding - len(subwords)
                subwords += [self.padTokenId] * pad_len
                word_ids += [-100] * pad_len
                attention_mask = [1] * attention_len + [0] * pad_len

        else:
            attention_mask = [1] * len(subwords)

        return subwords, word_ids, attention_mask

    def loadVocab(self, path):
        words = [line.strip() for line in open(path)]
        self.vocab = set()
        self.word2id = {}
        self.id2word = {}
        for i, w in enumerate(words):
            self.vocab.add(w)
            self.word2id[w] = i
            self.id2word[i] = w
        self.vocabSize = len(self.vocab)
        self.maxLength = max(len(w) for w in self.vocab)

        self.unkToken = "[UNK]"
        self.unkTokenId = self.word2id[self.unkToken]
        self.clsToken = "[CLS]"
        self.clsTokenId = self.word2id[self.clsToken]
        self.sepToken = "[SEP]"
        self.sepTokenId = self.word2id[self.sepToken]

    def loadBertTokenizer(self, bertTokenizer, doNaivePreproc=False):
        if doNaivePreproc:
            self.doNaivePreproc = doNaivePreproc
            self.bertTokenizer = bertTokenizer

        self.midPref = "##"
        self.vocab = set()
        self.word2id = {}
        self.id2word = {}

        for w, i in bertTokenizer.vocab.items():
            self.vocab.add(w)
            self.word2id[w] = i
            self.id2word[i] = w
        self.vocabSize = len(self.vocab)
        self.maxLength = max(len(w) for w in self.vocab)

        self.unkToken = bertTokenizer.unk_token
        self.unkTokenId = bertTokenizer.unk_token_id
        self.clsToken = bertTokenizer.cls_token
        self.clsTokenId = bertTokenizer.cls_token_id
        self.sepToken = bertTokenizer.sep_token
        self.sepTokenId = bertTokenizer.sep_token_id
        self.bosToken = bertTokenizer.bos_token
        self.bosTokenId = bertTokenizer.bos_token_id
        self.eosToken = bertTokenizer.eos_token
        self.eosTokenId = bertTokenizer.eos_token_id
        self.padToken = bertTokenizer.pad_token
        self.padTokenId = bertTokenizer.pad_token_id

    def naivePreproc(self, text):
        return " ".join(self.bertTokenizer.tokenize(text)).replace(" " + self.midPref, "")

    def get_label(self, word_ids, label, subword_label):
        previous_word_idx = -100
        label_ids = []
        for word_idx in word_ids:
            if word_idx == -100:
                label_ids.append(self.ner_dict["PAD"])
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                if subword_label == "I":
                    label_ids.append(self.subword_dict[label[word_idx]])
                elif subword_label == "B":
                    label_ids.append(label[word_idx])
                elif subword_label == "PAD":
                    label_ids.append(self.ner_dict["PAD"])
                else:
                    print("subword_label must be 'I', 'B' or 'PAD'.")
                    exit(1)

            previous_word_idx = word_idx
        return label_ids

    def dataset_encode(
        self,
        data,
        p=None,
        return_tensor=True,
        subword_label="I",
        post_sentence_padding=False,
        add_sep_between_sentences=False,
    ):
        p = p if p is not None else self.p

        row_tokens = []
        row_labels = []
        input_ids = []
        attention_mask = []
        token_type_ids = []
        subword_labels = []
        predict_labels = []
        weight_y = []
        for document in data:
            text = document["tokens"]
            labels = document["labels"]
            max_length = self.padding - 2 if self.padding else len(text)

            for i, j in document["doc_index"]:
                subwords, word_ids = self.tokenize(" ".join(text[i:j]), p)
                row_tokens.append(text[i:j])
                row_labels.append(labels[i:j])
                masked_ids = copy.deepcopy(word_ids)
                token_type_id = [0] * len(subwords)

                if post_sentence_padding:
                    while len(subwords) < max_length and j < len(text):
                        if add_sep_between_sentences and j in [d[0] for d in document["doc_index"]]:
                            subwords.append(self.sepToken)
                            word_ids.append(-100)
                            masked_ids.append(-100)
                            token_type_id.append(token_type_id[-1])
                        ex_subwords = self.tokenizeWord(text[j], p=p)
                        subwords = subwords + ex_subwords
                        word_ids = word_ids + [max(word_ids) + 1] * len(ex_subwords)
                        masked_ids = masked_ids + [-100] * len(ex_subwords)
                        token_type_id = token_type_id + [1 if add_sep_between_sentences else 0] * len(ex_subwords)
                        j += 1
                        if len(subwords) < max_length:
                            subwords = subwords[:max_length]
                            word_ids = word_ids[:max_length]
                            masked_ids = masked_ids[:max_length]
                            token_type_id = token_type_id[:max_length]

                subwords = [self.clsTokenId] + [self.word2id[w] for w in subwords] + [self.sepTokenId]
                word_ids = [-100] + word_ids + [-100]
                masked_ids = [-100] + masked_ids + [-100]
                token_type_id = [0] + token_type_id + [token_type_id[-1]]

                if self.padding:
                    if len(subwords) >= self.padding:
                        subwords = subwords[: self.padding]
                        word_ids = word_ids[: self.padding]
                        masked_ids = masked_ids[: self.padding]
                        token_type_id = token_type_id[: self.padding]
                        mask = [1] * self.padding

                    else:
                        attention_len = len(subwords)
                        pad_len = self.padding - len(subwords)
                        subwords += [self.padTokenId] * pad_len
                        word_ids += [-100] * pad_len
                        masked_ids += [-100] * pad_len
                        token_type_id += [1] * pad_len
                        mask = [1] * attention_len + [0] * pad_len
                else:
                    mask = [1] * len(subwords)

                input_ids.append(subwords)
                attention_mask.append(mask)
                token_type_ids.append(token_type_id)

                label = labels[i:j]
                label_ids = self.get_label(word_ids, label, subword_label)
                subword_labels.append(label_ids)

                masked_label = row_labels[-1]
                masked_label_ids = self.get_label(masked_ids, masked_label, "PAD")
                predict_labels.append(masked_label_ids)

                weight_y += [l_i for l_i in label_ids if l_i != self.ner_dict["PAD"]]

        loss_rate = compute_sample_weight("balanced", y=weight_y)
        weight = [loss_rate[weight_y.index(i)] for i in range(len(set(weight_y)))]
        weight.append(0)

        if return_tensor:
            data = {
                "input_ids": torch.tensor(input_ids, dtype=torch.int),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.int),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.int),
                "subword_labels": torch.tensor(subword_labels, dtype=torch.long),
                "predict_labels": torch.tensor(predict_labels, dtype=torch.long),
                "tokens": row_tokens,
                "labels": row_labels,
                "weight": torch.tensor(weight, dtype=torch.float32),
            }
        else:
            data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "subword_labels": subword_labels,
                "predict_labels": predict_labels,
                "tokens": row_tokens,
                "labels": row_labels,
                "weight": weight,
            }
        return data


if __name__ == "__main__":
    sent = "The English data is a collection 's of news wire articles from the Reuters Corpus ."
    sent2 = "Converts a single index or a sequence of indices in a token or a sequence of tokens ."
    print(sent)

    mmt = MaxMatchTokenizer(p=0.3, padding=40)
    bert_tokeninzer = AutoTokenizer.from_pretrained("bert-base-cased")
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer, doNaivePreproc=True)

    tokens = mmt.tokenize([sent, sent2])
    # tokens = [list(chain.from_iterable(tkns)) for tkns in tokens]
    ids = [mmt.encode(tkns) for tkns in [sent, sent2]]
    # , truncation=True return_tensors="pt", max_length=80, padding="max_length")
    print(ids[0][0])
    print(ids[0][1])
    print(len(ids[0][1]))
    # print(ids.word_ids())
    print(bert_tokeninzer.decode(ids[0][0]))
