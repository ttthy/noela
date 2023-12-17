import sys

import torch.nn as nn

import numpy as np


class BaseModel(nn.Module):
    def __init__(self, *args, **kawrgs):
        super(BaseModel, self).__init__()

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        params_memory = sum([sys.getsizeof(p) for p in model_parameters])
        print("""\n---------------- Summary ----------------
            \nNamed modules: {}
            \nTrainable parameters: {} params, {:.5f} MB, {}
            \nModel: \n{}
        """.format(
            [k for k, v in self.named_parameters() if v.requires_grad],
            params, params*32*1.25e-7, params_memory,
            self if not self.encoder_name.startswith("bert") else ""
        ))
