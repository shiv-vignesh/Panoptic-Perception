
from typing import Dict, Any

class LossFactory:

    _loss_functions : dict[str, type] = {}

    @classmethod
    def register_loss_function(cls, name:str):
        def decorator(model_cls):
            cls._loss_functions[name] = model_cls
            return model_cls
        
        return decorator
    
    @classmethod
    def build(cls, cfg:Dict[str, Any]):

        cfg_copy = cfg.copy()
        loss_type = cfg_copy.pop("_type")
        if loss_type not in cls._loss_functions:
            raise KeyError(f"Loss Type: {loss_type} is not registered!")
        
        return cls._loss_functions[loss_type](**cfg_copy.get("kwargs", {}))