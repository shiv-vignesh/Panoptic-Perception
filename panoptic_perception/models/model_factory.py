import os

class ModelFactory:
    _task_models : dict[str, type] = {}
    _enhancements : dict[str, type] = {}

    @classmethod
    def register_task_model(cls, name:str):
        def decorator(model_cls):
            cls._task_models[name] = model_cls
            return model_cls

        return decorator
    
    @classmethod
    def register_enhancement(cls, name:str):
        def decorator(model_cls):
            cls._enhancements[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def from_config(cls, model_kwargs:dict, loss_weights:dict = None):
        model_type = model_kwargs.get("model_type", "")
        cfg_path = model_kwargs.get("cfg_path", "")
                
        assert os.path.exists(cfg_path), f'{cfg_path} does not exists'        
        if model_type not in cls._task_models:
            raise KeyError(f'Unknown Model Class: {model_type}')
        
        task_model_cls = cls._task_models[model_type]
        task_kwargs = {"cfg": cfg_path, "loss_weights": loss_weights}
        if "transmission_kwargs" in model_kwargs:
            task_kwargs["transmission_kwargs"] = model_kwargs["transmission_kwargs"]
        try:
            task_model = task_model_cls(**task_kwargs)
        except TypeError:
            # Models that don't accept transmission_kwargs (e.g. YOLOv8P) fall back.
            task_kwargs.pop("transmission_kwargs", None)
            task_model = task_model_cls(**task_kwargs)
        
        enhancement = model_kwargs.get("enhancement")
        if enhancement is None:
            return task_model
        
        if enhancement not in cls._enhancements:
            raise KeyError(f"Unknown enhancement: {enhancement}")
        
        enh_cls = cls._enhancements[enhancement]
        return enh_cls.from_config(task_model, **model_kwargs)