import copy
import random

from transformers import LukeTokenizer


def get_pairs(word, alpha=0):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        if alpha < random.random() or prev_char == "Ġ":
            pairs.add((prev_char, char))
        else:
            pairs.add((prev_char, "@@" + char))
        prev_char = char
    return pairs


class LukeTokenizerDropout(LukeTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        entity_vocab_file,
        task=None,
        max_entity_length=32,
        max_mention_length=30,
        entity_token_1="<ent>",
        entity_token_2="<ent2>",
        entity_unk_token="[UNK]",
        entity_pad_token="[PAD]",
        entity_mask_token="[MASK]",
        entity_mask2_token="[MASK2]",
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        alpha=0,
        seed=42,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            entity_vocab_file,
            task,
            max_entity_length,
            max_mention_length,
            entity_token_1,
            entity_token_2,
            entity_unk_token,
            entity_pad_token,
            entity_mask_token,
            entity_mask2_token,
            errors,
            bos_token,
            eos_token,
            sep_token,
            cls_token,
            unk_token,
            pad_token,
            mask_token,
            add_prefix_space,
            **kwargs,
        )
        self.alpha = alpha
        self._alpha = alpha
        self.seed = seed
        random.seed(self.seed)

    def const_tokenize(self):
        self._alpha = copy.deepcopy(self.alpha)
        self.alpha = 0

    def random_tokenize(self):
        self.alpha = self._alpha

    def reset_seed(self):
        random.seed(self.seed)

    def bpe(self, token):
        if (token in self.cache) and (self.alpha == 0):
            # サブワード正則化時はキャッシュから返さない
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word, self.alpha)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word, self.alpha)
        word = " ".join(word)
        if self.alpha == 0:
            # サブワード正則化時の分割はキャッシュしない
            self.cache[token] = word
        return word


if __name__ == "__main__":
    tokenizer: LukeTokenizerDropout = LukeTokenizerDropout.from_pretrained(
        "studio-ousia/luke-large-finetuned-conll-2003", alpha=0.2
    )
    text = "This is a rtd pen."
    span = [
        (0, 4),
        (0, 7),
        (0, 9),
        (5, 7),
        (5, 9),
        (5, 13),
        (8, 9),
        (8, 13),
        (8, 17),
        (10, 13),
        (10, 17),
        (14, 17),
    ]

    print("random tokenize")
    for i in range(10):
        a = tokenizer.tokenize(text)
        print(a)

    print("random tokenize")
    tokenizer.reset_seed()
    for i in range(10):
        a = tokenizer.tokenize(text)
        print(a)

    tokenizer.const_tokenize()
    print("const tokenize")
    for i in range(10):
        a = tokenizer.tokenize(text)
        print(a)
    b = tokenizer([text], entity_spans=[span], return_tensors="pt", padding=True)
    print(b)
