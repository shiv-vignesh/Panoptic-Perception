
from panoptic_perception.scripts.train.train_v2 import create_model
from panoptic_perception.models.teacher_model import TeacherFusion
from panoptic_perception.models.model_factory import ModelFactory

def create_teacher_model(model_kwargs:dict, loss_kwargs:dict):

    cfg_file = model_kwargs["cfg"]
    
