"""
Overfit SwinClassifier on a single batch to verify model/loss/target wiring.
No augmentation, no weight decay, no grad clip, no warmup.

If loss drops toward 0 and accuracy climbs toward 1.0 within ~200 steps, the
pipeline is wired correctly and the pretraining stall is a hyperparameter or
schedule issue. If loss stays at ln(num_classes), something structural is
broken (head disconnected, targets wrong, gradients not flowing).

Usage:
    python -m panoptic_perception.scripts.train.swin_smoke \\
        --config panoptic_perception/configs/trainer/train_kwargs_swin_pretrain.json
"""

import argparse
import copy
import math

import torch

from panoptic_perception.dataset.imagenet_dataset import DataLoaderBuilder
from panoptic_perception.losses.loss_factory import LossFactory
from panoptic_perception.models import ModelFactory
from panoptic_perception.utils.config_parser import load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    cfg = load_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_kwargs = copy.deepcopy(cfg["dataset_kwargs"])
    dl_kwargs["train_preprocessor_kwargs"]["perform_augmentation"] = False
    train_dl = DataLoaderBuilder(dl_kwargs, logger=None)._build_train()

    model = ModelFactory.from_config(cfg["model_kwargs"]).to(device)
    model.loss_function = LossFactory.build(cfg["loss_kwargs"])
    model.train()

    batch = next(iter(train_dl))
    imgs = batch["images"].to(device)
    tgts = batch["labels"].to(device)

    print(f"[smoke] batch: images {tuple(imgs.shape)} dtype={imgs.dtype} "
          f"range=[{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
    print(f"[smoke] labels: min={tgts.min().item()} max={tgts.max().item()} "
          f"unique={tgts.unique().numel()}/{tgts.numel()} dtype={tgts.dtype}")
    print(f"[smoke] num_classes={model.num_classes}, "
          f"random floor CE = ln(N) = {math.log(model.num_classes):.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.999))
    for step in range(args.steps):
        out = model(imgs, tgts)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            acc = (out.logits.argmax(1) == tgts).float().mean().item()
            gnorm = math.sqrt(sum(p.grad.pow(2).sum().item()
                                  for p in model.parameters() if p.grad is not None))
            print(f"[smoke] step {step:3d} loss {out.loss.item():.4f} "
                  f"acc {acc:.3f} grad_norm {gnorm:.3f}")


if __name__ == "__main__":
    main()
