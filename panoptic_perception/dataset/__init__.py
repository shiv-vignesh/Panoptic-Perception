from typing import Dict, Optional

from torch.utils.data import DataLoader

from panoptic_perception.dataset.bdd100k_dataset import (
    BDD100KDataset, FoggyBDD100KDataset, BDDPreprocessor, DatasetMode
)
from panoptic_perception.utils.logger import Logger

class DataLoaderBuilder:

    def __init__(self, dataset_kwargs: dict, logger:Logger=None):
        self._kw = dataset_kwargs
        self._dataset_class = dataset_kwargs.get("dataset_class", "default")
        self._depth_estimator = None

        if self._dataset_class == "foggy":
            self._init_depth_estimator(logger)

    def _init_depth_estimator(self, logger:Logger):
        adverse_params = self._kw.get("adverse_params", {})
        depth_backend = adverse_params.get("depth_backend", "heuristic")
        depth_device = adverse_params.get("depth_device", "cuda")

        if depth_backend == "onnx":
            from panoptic_perception.dataset.adverse_weather.depth_estimators import ONNXDepthEstimator
            self._depth_estimator = ONNXDepthEstimator(
                onnx_path=adverse_params["onnx_backend_path"],
                device=depth_device, input_size=518, normalization_epsilon=1e-8,
            )
        elif depth_backend == "depth_anything":
            from panoptic_perception.dataset.adverse_weather.depth_estimators import DepthAnythingEstimator
            self._depth_estimator = DepthAnythingEstimator(
                model_name="LiheYoung/depth-anything-small-hf",
                device=depth_device, normalization_epsilon=1e-8,
            )
        elif depth_backend == "torch_compile":
            from panoptic_perception.dataset.adverse_weather.depth_estimators import TorchCompiledDepthEstimator
            self._depth_estimator = TorchCompiledDepthEstimator(
                model_name="LiheYoung/depth-anything-small-hf",
                device=depth_device, normalization_epsilon=1e-8,
            )

        if depth_backend != "heuristic":
            if logger:
                logger.log_message(
                    f"[Dataloader] depth_backend={depth_backend} requires CUDA "
                    f"-> overriding num_workers to 0"
                )
            self._kw["train_num_workers"] = 0
            self._kw["val_num_workers"] = 0

    def _base_kwargs(self, preprocessor_kwargs: dict) -> dict:
        return {
            "images_dir": self._kw["images_dir"],
            "detection_annotations_dir": self._kw["detection_annotations_dir"],
            "segmentation_annotations_dir": self._kw["segmentation_annotations_dir"],
            "drivable_annotations_dir": self._kw["drivable_annotations_dir"],
            "preprocessor_kwargs": preprocessor_kwargs,
        }

    def _add_foggy_kwargs(self, kwargs: dict) -> dict:
        kwargs["depth_map_dir"] = self._kw.get("depth_map_dir", None)
        kwargs["adverse_params"] = self._kw.get("adverse_params", {})
        return kwargs

    def _make_loader(self, dataset, batch_size, shuffle, num_workers) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=BDDPreprocessor.collate_fn,
            pin_memory=True,
        )

    def build_train(self) -> DataLoader:
        kwargs = self._base_kwargs(self._kw.get("train_preprocessor_kwargs", {}))
        perform_aug = kwargs["preprocessor_kwargs"].get("perform_augmentation", False)

        if self._dataset_class == "foggy":
            self._add_foggy_kwargs(kwargs)
            dataset = FoggyBDD100KDataset(
                kwargs, dataset_type="train",
                perform_augmentation=perform_aug,
                mode=DatasetMode.TRAIN,
                strict_map=self._kw.get("strict_map", True),
                apply_fog_prob=self._kw.get("apply_fog_prob", 0.67),
                depth_estimator=self._depth_estimator,
            )
        else:
            dataset = BDD100KDataset(
                kwargs, dataset_type="train",
                perform_augmentation=perform_aug,
                mode=DatasetMode.TRAIN,
            )

        return self._make_loader(
            dataset,
            batch_size=self._kw["train_batch_size"],
            shuffle=self._kw.get("train_shuffle", True),
            num_workers=self._kw.get("train_num_workers", 4),
        )

    def build_val(self) -> Dict[str, DataLoader]:
        kwargs = self._base_kwargs(self._kw.get("val_preprocessor_kwargs", {}))
        batch_size = self._kw["val_batch_size"]
        num_workers = self._kw.get("val_num_workers", 4)

        clean_dataset = BDD100KDataset(
            kwargs, dataset_type="val",
            perform_augmentation=False,
            mode=DatasetMode.EVAL,
        )

        if self._dataset_class != "foggy":
            return {"val": self._make_loader(clean_dataset, batch_size, False, num_workers)}

        foggy_kwargs = self._add_foggy_kwargs({**kwargs})
        foggy_dataset = FoggyBDD100KDataset(
            foggy_kwargs, dataset_type="val",
            perform_augmentation=False,
            mode=DatasetMode.EVAL,
            strict_map=self._kw.get("strict_map", True),
            apply_fog_prob=1.0,
            depth_estimator=self._depth_estimator,
        )

        return {
            "val_clean": self._make_loader(clean_dataset, batch_size, False, num_workers),
            "val_foggy": self._make_loader(foggy_dataset, batch_size, False, num_workers),
        }
