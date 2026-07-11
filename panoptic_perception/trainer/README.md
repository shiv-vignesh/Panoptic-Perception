# Trainer Refactor — Template Method for Task-Specific Loops

## Motivation

The current `Trainer._train_one_epoch` (and `_eval_one_epoch`) reads task-specific fields off the `PanopticModelOutputs` dataclass — `outputs.detection_loss`, `outputs.drivable_segmentation_loss`, `outputs.lane_detection_loss`, etc. This works for detection but breaks the moment a new task (classification, distillation, etc.) returns a different output type.

Symptom already seen on the classifier pretraining path:

```
AttributeError: 'ImageClassifierOutputs' object has no attribute 'detection_loss'
```

The subclass (`SwinTrainerClassifier`) overrode `_forward_model` and `_train_one_step` but inherited `_train_one_epoch`, which still reads detection fields.

The fix is not to override `_train_one_epoch` in every subclass. That duplicates the entire scaffold (grad accumulation, warmup, callbacks, per-window logging, LR stepping) and diverges over time. The fix is to split the base into (a) task-agnostic scaffolding and (b) task-specific hooks the subclass implements.

## Principle — Template Method

Base owns:
- Iterating the loader
- Calling `_train_one_step` (already subclass-provided)
- Optimizer step + gradient accumulation
- Warmup schedule
- Callbacks
- LR scheduler epoch step
- Smoke-cap short-circuit

Subclass owns:
- What fields to accumulate per batch
- What to log per iteration
- What to log every 10%
- What to log at epoch end

No `if isinstance(outputs, ...)` in the base. No `getattr(outputs, 'detection_loss', None)` fallbacks. The base doesn't know or care what shape the output has.

## Hook Surface — `_train_one_epoch`

Six hooks. Defaults handle the generic case (running loss). Subclasses override the ones that need task-specific fields.

```python
class Trainer:

    def _init_train_window(self):
        """Reset window-scoped accumulators. Called once at epoch start and
        again after each 10% window fires."""
        self._window_loss = 0.0
        self._window_step_time = 0.0
        self._window_batches = 0

    def _accumulate_train_iter(self, loss, model_outputs, step_time):
        """Per-batch accumulation. Base handles loss + timing. Subclass extends
        with task-specific fields (per-task losses, top-k accuracy, etc)."""
        self._window_loss += loss.item()
        self._window_step_time += step_time
        self._window_batches += 1

    def _log_train_iter(self, batch_idx, loss, model_outputs, current_lr):
        """Per-iter log line (fires every log_every_n_iters). Default no-op."""
        pass

    def _log_train_window(self, current_lr, batch_idx):
        """Called every 10% of the epoch. Writes the summary + wandb metrics.
        Subclass override provides task-specific averages and metric names."""
        n = self._window_batches or 1
        avg_loss = self._window_loss / n
        avg_time = self._window_step_time / n
        self.logger.log_message(
            f"Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} "
            f"- loss {avg_loss:.4f} -- lr: {current_lr}"
        )
        self.wandb_logger.log_metrics(
            {"train/loss_10pct": avg_loss,
             "train/lr": current_lr,
             "train/avg_step_time": avg_time},
            step=self.cur_epoch * self.total_train_batch + batch_idx,
        )

    def _log_train_epoch(self, avg_epoch_loss, current_lr, epoch_time):
        """End-of-epoch summary. Generic; rarely overridden."""
        self.logger.log_message(
            f"Epoch {self.cur_epoch} - Average Loss {avg_epoch_loss:.4f} -- current_lr: {current_lr}"
        )
        self.wandb_logger.log_metrics(
            {"train/epoch_loss": avg_epoch_loss,
             "train/epoch_time": epoch_time,
             "train/epoch": self.cur_epoch},
            step=self.cur_epoch,
        )
```

## The Task-Agnostic Scaffold

`_train_one_epoch` reduces to pure orchestration. No field reads on `model_outputs`.

```python
def _train_one_epoch(self):
    self.model.train()
    self._init_train_window()
    total_loss = 0.0
    total_time = 0.0
    current_lr = self.optimizer.param_groups[0]['lr']

    self.total_train_batch = len(self.train_dataloader)
    self.ten_percent_train_batch = max(1, self.total_train_batch // 10)

    pbar = tqdm(self.train_dataloader, desc=f'Training Epoch: {self.cur_epoch}')

    for batch_idx, data_items in enumerate(pbar):
        self.train_batch_idx = batch_idx

        t0 = time.time()
        loss, model_outputs = self._train_one_step(data_items)
        step_time = time.time() - t0

        if (batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0:
            if self.training_args.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_args.gradient_clipping
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        total_loss += loss.item()
        total_time += step_time

        self._accumulate_train_iter(loss, model_outputs, step_time)

        if (self.training_args.log_every_n_iters and
                (batch_idx + 1) % self.training_args.log_every_n_iters == 0):
            self._log_train_iter(batch_idx, loss, model_outputs, current_lr)

        if (batch_idx + 1) % self.ten_percent_train_batch == 0:
            self._apply_warmup()
            self._log_train_window(current_lr, batch_idx)
            self._init_train_window()

        self.callbacks.on_step_end(self)

        if (self.training_args.max_train_iters and
                (batch_idx + 1) >= self.training_args.max_train_iters):
            self.logger.log_message(
                f"[smoke] max_train_iters={self.training_args.max_train_iters} reached"
            )
            break

    observed = (batch_idx + 1) if self.total_train_batch else 1
    avg_epoch_loss = total_loss / observed
    self._log_train_epoch(avg_epoch_loss, current_lr, total_time)

    if hasattr(self, 'lr_scheduler'):
        if (self.training_args.lr_scheduler_start_epoch != -1 and
                self.cur_epoch > self.training_args.lr_scheduler_start_epoch):
            self.lr_scheduler.step()
```

## Detection Trainer — moves the removed reads here

All `outputs.detection_loss` etc. reads relocate from the base to a `DetectionTrainer` subclass. YOLOP + BDD paths inherit from `DetectionTrainer` instead of `Trainer`.

```python
class DetectionTrainer(Trainer):

    def _init_train_window(self):
        super()._init_train_window()
        self._window_det = 0.0
        self._window_drv = 0.0
        self._window_lane_seg = 0.0
        self._window_lane_det = 0.0
        self._window_lane_items = {}

    def _accumulate_train_iter(self, loss, model_outputs, step_time):
        super()._accumulate_train_iter(loss, model_outputs, step_time)
        if model_outputs is None:
            return
        if model_outputs.detection_loss is not None:
            self._window_det += model_outputs.detection_loss.item()
        if model_outputs.drivable_segmentation_loss is not None:
            self._window_drv += model_outputs.drivable_segmentation_loss.item()
        if model_outputs.lane_segmentation_loss is not None:
            self._window_lane_seg += model_outputs.lane_segmentation_loss.item()
        if model_outputs.lane_detection_loss is not None:
            self._window_lane_det += model_outputs.lane_detection_loss.item()
        if model_outputs.lane_detection_loss_items is not None:
            for k, v in model_outputs.lane_detection_loss_items.items():
                self._window_lane_items[k] = self._window_lane_items.get(k, 0.0) + v

    def _log_train_iter(self, batch_idx, loss, model_outputs, current_lr):
        if model_outputs is None:
            return
        det = model_outputs.detection_loss.item() if model_outputs.detection_loss is not None else 0.0
        drv = model_outputs.drivable_segmentation_loss.item() if model_outputs.drivable_segmentation_loss is not None else 0.0
        self.logger.log_message(
            f"[iter {batch_idx:4d}] loss={loss.item():.4f} det={det:.4f} drv={drv:.4f}"
        )

    def _log_train_window(self, current_lr, batch_idx):
        n = self._window_batches or 1
        avg_loss = self._window_loss / n
        avg_det = self._window_det / n
        avg_drv = self._window_drv / n
        avg_lane_seg = self._window_lane_seg / n
        avg_lane_det = self._window_lane_det / n

        parts = [f'total {avg_loss:.4f}']
        if avg_det > 0:      parts.append(f'det {avg_det:.4f}')
        if avg_drv > 0:      parts.append(f'drv {avg_drv:.4f}')
        if avg_lane_seg > 0: parts.append(f'lane_seg {avg_lane_seg:.4f}')
        if avg_lane_det > 0: parts.append(f'lane_det {avg_lane_det:.4f}')

        self.logger.log_message(
            f'Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} '
            f'- {" | ".join(parts)} -- lr: {current_lr}'
        )

        metrics = {"train/loss_10pct": avg_loss, "train/lr": current_lr}
        if avg_det > 0:      metrics["train/det_loss"] = avg_det
        if avg_drv > 0:      metrics["train/drivable_loss"] = avg_drv
        if avg_lane_seg > 0: metrics["train/lane_seg_loss"] = avg_lane_seg
        if avg_lane_det > 0: metrics["train/lane_det_loss"] = avg_lane_det
        for k, v in self._window_lane_items.items():
            metrics[f"train/{k}"] = v / n

        self.wandb_logger.log_metrics(
            metrics, step=self.cur_epoch * self.total_train_batch + batch_idx
        )
```

## Classifier Trainer — small, only what applies

`SwinTrainerClassifier` overrides four hooks; the rest inherit from base.

```python
class SwinTrainerClassifier(Trainer):

    def _init_train_window(self):
        super()._init_train_window()
        self._window_correct = 0
        self._window_total = 0

    def _accumulate_train_iter(self, loss, model_outputs, step_time):
        super()._accumulate_train_iter(loss, model_outputs, step_time)
        if model_outputs is None or model_outputs.logits is None:
            return
        with torch.no_grad():
            pred = model_outputs.logits.argmax(dim=1)
            self._window_correct += (pred == model_outputs.targets).sum().item()
            self._window_total += pred.size(0)

    def _log_train_window(self, current_lr, batch_idx):
        n = self._window_batches or 1
        avg_loss = self._window_loss / n
        top1 = self._window_correct / max(self._window_total, 1)

        self.logger.log_message(
            f"Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} "
            f"- loss {avg_loss:.4f} - top1 {top1:.4f} -- lr: {current_lr}"
        )
        self.wandb_logger.log_metrics(
            {"train/loss_10pct": avg_loss,
             "train/top1_10pct": top1,
             "train/lr": current_lr},
            step=self.cur_epoch * self.total_train_batch + batch_idx,
        )
```

No `_train_one_epoch` override needed.

## `_eval_one_epoch` — Same Pattern

Same split applies. Extract hooks:

- `_init_eval_state(prefix)` — reset per-epoch metric accumulators
- `_accumulate_eval_iter(model_outputs, data_items)` — per-batch metric update
- `_log_eval_epoch(prefix, avg_loss)` — write summary

`DetectionTrainer` overrides these to accumulate detection AP, IoU, per-class breakdowns.
`SwinTrainerClassifier` overrides to accumulate top-1 / top-5 accuracy.

Detailed code omitted here; follow the same pattern as `_train_one_epoch` above.

## Migration Plan

Do these in order. Each step is independently mergeable.

1. **Introduce the hooks on `Trainer` with default implementations that mimic today's generic behaviour.** Base's `_train_one_epoch` and `_eval_one_epoch` become task-agnostic. Existing behaviour preserved for anything that inherits directly from `Trainer` without a task-specific parent.

2. **Extract the current detection-specific reads into `DetectionTrainer(Trainer)`.** YOLOP + BDD paths change their base class from `Trainer` to `DetectionTrainer`. Detection wandb dashboards continue to work.

3. **`SwinTrainerClassifier(Trainer)` overrides only the hooks it needs.** Unblocks the classifier pretraining path. Detection paths untouched.

4. **Future task trainers pick their parent.** Segmentation-only → `SegmentationTrainer(Trainer)`. Distillation → `DistillationTrainer(Trainer)`. Each isolates its field reads to its own hook overrides.

## Optional Cleanups Later

- **Remove `model_outputs` type coupling.** The scaffold uses `model_outputs` opaquely — subclass hooks receive it and know what it is. No `PanopticModelOutputs` imports needed in the base file.
- **Move all print / log I/O into hooks.** Scaffold has zero I/O of its own. Easier to unit-test.
- **Consider a `TrainMetricsAccumulator` composition object.** Instead of overriding hooks, subclass passes an accumulator instance to base. Slightly cleaner for very complex tasks, overkill for the current two.

## Non-Goals

- Not changing the `train()` outer loop.
- Not changing checkpoint save/load — works for any `nn.Module`.
- Not changing `_apply_warmup` — operates on optimizer param groups, task-agnostic.
- Not changing the callback hook signatures.
- Not changing `TrainingArgument` shape.

## Reference — Where Task-Specific Reads Live Today (Base)

For grep-ability during the refactor. Move each of these into the corresponding hook on `DetectionTrainer`:

- `trainer.py:183-206` — per-iter detection/drivable/lane loss reads
- `trainer.py:217-262` — 10% window aggregation + wandb metrics
- `trainer.py:_eval_one_epoch` — same reads on the eval side

These are the exact lines to lift. Nothing else in `_train_one_epoch` / `_eval_one_epoch` should need to move.
