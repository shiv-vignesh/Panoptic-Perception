import torch

from panoptic_perception.models.swin_model import SwinClassifier
from panoptic_perception.trainer.trainer import Trainer
from panoptic_perception.utils.logger import Logger
from panoptic_perception.models.utils import WeightsManager


def load_swin_classifier(model: SwinClassifier, model_path: str, logger: Logger):
    wm = WeightsManager(verbose=False)
    result = wm.load(model, model_path, strict=False, key_prefix=None)
    if result is None:
        if logger is not None:
            logger.log_message(f"[swin classifier init]: checkpoint not found at {model_path}; skipping")
        return

    missing, unexpected, loaded_keys = result
    if logger is not None:
        logger.log_message(
            f"[swin classifier init]: loaded from {model_path} "
            f"(loaded={len(loaded_keys)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )
    if missing:
        logger.log_message(f"[swin classifier init] missing keys (first 10): {missing[:10]}")
    if unexpected:
        logger.log_message(f"[swin classifier init] unexpected keys (first 10): {unexpected[:10]}")


class SwinTrainerClassifier(Trainer):

    def _forward_model(self, data_items):
        return self.model(
            data_items["images"],
            data_items["labels"],
        )

    def _init_train_window(self):
        super()._init_train_window()
        self._window_correct = 0
        self._window_total = 0

    def _accumulate_train_iter(self, loss, model_outputs, step_time):
        super()._accumulate_train_iter(loss, model_outputs, step_time)
        if model_outputs is None or getattr(model_outputs, "logits", None) is None:
            return
        targets = getattr(model_outputs, "targets", None)
        if targets is None:
            return
        with torch.no_grad():
            pred = model_outputs.logits.argmax(dim=1)
            self._window_correct += (pred == targets).sum().item()
            self._window_total += pred.size(0)

    def _log_train_window(self, current_lr, batch_idx):
        n = self._window_batches or 1
        avg_loss = self._window_loss / n
        top1 = self._window_correct / max(self._window_total, 1)

        self.logger.log_message(
            f"Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} "
            f"- loss {avg_loss:.4f} - top1 {top1:.4f} -- lr: {current_lr}"
        )
        self.wandb_logger.log_metrics({
            "train/loss_10pct": avg_loss,
            "train/top1_10pct": top1,
            "train/lr": current_lr,
        }, step=self.cur_epoch * self.total_train_batch + batch_idx)

    def _populate_eval_batch_ctx(self, data_items, outputs):
        super()._populate_eval_batch_ctx(data_items, outputs)
        self.eval_batch_ctx.cur_eval_gt_cls_labels = data_items["labels"]

    