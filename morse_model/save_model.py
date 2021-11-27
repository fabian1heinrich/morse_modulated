import random
import string
import torch
from datetime import datetime


def save_model(model):

    timestamp = datetime.now().strftime("%Y%m%d_%I_%M_%S_")
    random_string = ''.join(random.choices(string.ascii_uppercase, k=7))

    path = "saved_models/"
    name = timestamp+random_string+".json"

    torch.save({"model_state_dict": model.state_dict()},
               "{path}{name}".format(path=path, name=name))
