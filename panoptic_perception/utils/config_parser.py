import json
import os

from panoptic_perception.trainer.trainer_args import TrainingArgument

def load_json(config_fn:str):
    if not config_fn or not os.path.exists(config_fn):
        raise ValueError(f"Config File path is empty or does not exist: {config_fn}")

    return json.load(open(config_fn))

def parse_config(config: dict) -> dict:                                                                                           
    """Returns flattened dict for TrainingArgument.from_config()"""                                                               
                                                                                                                                
    result = {}                                                                                                                   

    # trainer_kwargs — direct mapping (bulk of the fields)
    if "trainer_kwargs" in config:
        result.update(config["trainer_kwargs"])

    # optimizer_kwargs — overlapping field names
    if "optimizer_kwargs" in config:
        optim = config["optimizer_kwargs"]
        for key in ("initial_lr", "weight_decay", "momentum",
                    "warmup_bias_lr", "warmup_momentum", "warmup_epochs"):
            if key in optim:
                result[key] = optim[key]

    # lr_scheduler_kwargs — needs key remapping
    if "lr_scheduler_kwargs" in config:
        sched = config["lr_scheduler_kwargs"]
        if "_type" in sched:
            result["lr_type"] = sched["_type"]
        if "linear_lr_kwargs" in sched:
            result["linear_lr_start"] = sched["linear_lr_kwargs"].get("start_factor", 1.0)
            result["linear_lr_end"] = sched["linear_lr_kwargs"].get("end_factor", 0.01)
        if "cosine_annealing_lr_kwargs" in sched:
            result["cosine_annealing_eta_min"] = sched["cosine_annealing_lr_kwargs"].get("eta_min", 1e-6)

    return result