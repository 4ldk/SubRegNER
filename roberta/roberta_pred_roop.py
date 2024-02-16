import os
import sys
from logging import getLogger

import hydra

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.bpe_dropout import RobertaTokenizerDropout
from utils.predict import loop_pred

logger = getLogger(__name__)
root_path = os.getcwd()


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.visible_devices
    if cfg.huggingface_cache:
        os.environ["TRANSFORMERS_CACHE"] = cfg.huggingface_cache

    tokenizer = RobertaTokenizerDropout.from_pretrained(cfg.model_name, alpha=cfg.pred_p)
    loop_pred(
        length=cfg.length,
        model_name=cfg.model_name,
        test=cfg.test,
        tokenizer=tokenizer,
        loop=cfg.loop,
        batch_size=cfg.test_batch,
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
