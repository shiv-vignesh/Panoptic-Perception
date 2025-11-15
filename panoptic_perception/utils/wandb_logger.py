"""
WandB Logger for tracking experiments.
"""

import wandb
from typing import Dict, Any, Optional
import torch


class WandBLogger:
    """Weights & Biases logger for experiment tracking."""

    def __init__(self, project_name: str, run_name: str, config: Dict[str, Any],
                 entity: Optional[str] = None, tags: Optional[list] = None,
                 enabled: bool = True):
        """
        Initialize WandB logger.

        Args:
            project_name: Name of the WandB project
            run_name: Name of this specific run
            config: Configuration dictionary to log
            entity: WandB entity (username or team name)
            tags: List of tags for this run
            enabled: Whether to actually log to WandB (useful for debugging)
        """
        self.enabled = enabled

        if self.enabled:
            try:
                self.run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config,
                    entity=entity,
                    tags=tags,
                    reinit=True
                )
                print(f" WandB initialized: {project_name}/{run_name}")
            except Exception as e:
                print(f"� WandB initialization failed: {e}")
                print(f"  Continuing without WandB logging")
                self.enabled = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Global step number
            commit: Whether to commit the log immediately
        """
        if not self.enabled:
            return

        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            print(f"� WandB logging failed: {e}")

    def log_image(self, key: str, image, caption: Optional[str] = None, step: Optional[int] = None):
        """
        Log an image to WandB.

        Args:
            key: Name/key for the image
            image: Image as numpy array, PIL Image, or file path
            caption: Optional caption for the image
            step: Global step number
        """
        if not self.enabled:
            return

        try:
            wandb.log({key: wandb.Image(image, caption=caption)}, step=step)
        except Exception as e:
            print(f"� WandB image logging failed: {e}")

    def log_table(self, key: str, columns: list, data: list, step: Optional[int] = None):
        """
        Log a table to WandB.

        Args:
            key: Name for the table
            columns: List of column names
            data: List of rows (each row is a list of values)
            step: Global step number
        """
        if not self.enabled:
            return

        try:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({key: table}, step=step)
        except Exception as e:
            print(f"� WandB table logging failed: {e}")

    def log_histogram(self, key: str, values, step: Optional[int] = None):
        """
        Log a histogram to WandB.

        Args:
            key: Name for the histogram
            values: Values as numpy array or torch tensor
            step: Global step number
        """
        if not self.enabled:
            return

        try:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            wandb.log({key: wandb.Histogram(values)}, step=step)
        except Exception as e:
            print(f"� WandB histogram logging failed: {e}")

    def watch_model(self, model, log_freq: int = 100, log: str = "gradients"):
        """
        Watch model gradients and parameters.

        Args:
            model: PyTorch model to watch
            log_freq: Frequency of logging
            log: What to log - "gradients", "parameters", or "all"
        """
        if not self.enabled:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            print(f"WandB watch failed: {e}")

    def log_artifact(self, artifact_path: str, artifact_type: str, name: str):
        """
        Log an artifact (model checkpoint, dataset, etc.).

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            name: Name for the artifact
        """
        if not self.enabled:
            return

        try:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"� WandB artifact logging failed: {e}")

    def finish(self):
        """Finish the WandB run."""
        if not self.enabled:
            return

        try:
            wandb.finish()
        except Exception as e:
            print(f"� WandB finish failed: {e}")
