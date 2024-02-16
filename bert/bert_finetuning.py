import os
import random
import sys

import hydra
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.maxMatchTokenizer import MaxMatchTokenizer
from utils.trainer import trainer
from utils.utils import get_dataloader, get_mv_dataloader, path_to_data

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
    train(
        cfg.batch_size,
        cfg.lr,
        cfg.num_epoch,
        cfg.length,
        cfg.p,
        cfg.seed,
        cfg.model_name,
        accum_iter=cfg.accum_iter,
        weight_decay=cfg.weight_decay,
        use_loss_weight=cfg.use_loss_weight,
        use_scheduler=cfg.use_scheduler,
        warmup_late=cfg.warmup_late,
        post_sentence_padding=cfg.post_sentence_padding,
        add_sep_between_sentences=cfg.add_sep_between_sentences,
        multi_view=cfg.multi_view,
        kl_weight=cfg.kl_weight,
        kl_t=cfg.kl_t,
    )


def train(
    batch_size,
    lr,
    num_epoch,
    length,
    p,
    seed,
    model_name,
    accum_iter=4,
    weight_decay=0,
    use_loss_weight=False,
    use_scheduler=False,
    warmup_late=0.01,
    post_sentence_padding=False,
    add_sep_between_sentences=False,
    multi_view=False,
    kl_weight=0.6,
    kl_t=1,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = "cuda"
    init_scale = 4096
    os.makedirs("./model")

    mmt = MaxMatchTokenizer(ner_dict=ner_dict, p=p, padding=length)
    bert_tokeninzer = AutoTokenizer.from_pretrained(model_name)
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer)

    train_dataset = path_to_data(os.path.join(root_path, "test_datasets/eng.train"))
    train_data = mmt.dataset_encode(
        train_dataset,
        p=p,
        subword_label="PAD",
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
    )
    if multi_view:
        const_data = mmt.dataset_encode(
            train_dataset,
            p=0,
            subword_label="PAD",
            post_sentence_padding=post_sentence_padding,
            add_sep_between_sentences=add_sep_between_sentences,
        )
        train_loader = get_mv_dataloader(const_data, train_data, batch_size=batch_size, shuffle=True)
    else:
        const_data = None
        train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)

    valid_dataset = path_to_data(os.path.join(root_path, "test_datasets/eng.testa"))
    valid_data = mmt.dataset_encode(
        valid_dataset,
        p=0,
        subword_label="PAD",
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
    )
    valid_loader = get_dataloader(valid_data, batch_size=batch_size, shuffle=False, drop_last=False)

    test_dataset = path_to_data(os.path.join(root_path, "test_datasets/eng.testb"))
    test_data = mmt.dataset_encode(
        test_dataset,
        p=0,
        subword_label="PAD",
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
    )
    test_loader = get_dataloader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    weight = train_data["weight"].to(device) if use_loss_weight else None
    num_training_steps = int(len(train_loader) / accum_iter) * num_epoch
    num_warmup_steps = int(num_training_steps * warmup_late)

    net = trainer(
        model_name=model_name,
        lr=lr,
        batch_size=batch_size,
        length=length,
        accum_iter=accum_iter,
        weight_decay=weight_decay,
        weight=weight,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        use_scheduler=use_scheduler,
        init_scale=init_scale,
        post_sentence_padding=post_sentence_padding,
        add_sep_between_sentences=add_sep_between_sentences,
        multi_view=multi_view,
        kl_weight=kl_weight,
        kl_t=kl_t,
        const_data=const_data,
        device=device,
    )
    net.train(mmt, train_dataset, train_loader, num_epoch, valid_loader=valid_loader, test_loader=test_loader, p=p)


if __name__ == "__main__":
    main()
