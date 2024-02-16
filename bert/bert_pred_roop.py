import os
import sys
from logging import getLogger

import hydra
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.maxMatchTokenizer import MaxMatchTokenizer
from utils.predict import loop_pred

logger = getLogger(__name__)
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


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.visible_devices
    if cfg.huggingface_cache:
        os.environ["TRANSFORMERS_CACHE"] = cfg.huggingface_cache

    mmt = MaxMatchTokenizer(ner_dict=ner_dict, p=cfg.pred_p, padding=cfg.length)
    bert_tokeninzer = AutoTokenizer.from_pretrained(cfg.model_name)
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer)

    loop_pred(
        cfg.length,
        cfg.model_name,
        cfg.test,
        tokenizer=mmt,
        loop=cfg.loop,
        batch_size=cfg.batch_size,
        p=cfg.pred_p,
        vote=cfg.vote,
        local_model=cfg.local_model,
        post_sentence_padding=cfg.post_sentence_padding,
        add_sep_between_sentences=cfg.add_sep_between_sentences,
        device=cfg.device,
        output_path=cfg.output_path,
    )


if __name__ == "__main__":
    main()
