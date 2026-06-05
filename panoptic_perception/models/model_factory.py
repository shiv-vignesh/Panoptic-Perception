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
    def from_config(cls, model_kwargs:dict):
        model_type = model_kwargs.get("model_type", "")
        cfg_path = model_kwargs.get("cfg_path", "")
                
        assert os.path.exists(cfg_path), f'{cfg_path} does not exists'        
        if model_type not in cls._task_models:
            raise KeyError(f'Unknown Model Class: {model_type}')
        
        task_model_cls = cls._task_models[model_type]
        task_model = task_model_cls(cfg_path)
        
        enhancement = model_kwargs.get("enhancement")
        if enhancement is None:
            return task_model
        
        if enhancement not in cls._enhancements:
            raise KeyError(f"Unknown enhancement: {enhancement}")
        
        enh_cls = cls._enhancements[enhancement]
        return enh_cls.from_config(task_model, **model_kwargs)