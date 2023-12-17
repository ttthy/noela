import torch
import torch.nn as nn
from ure.nn.base import BaseModel
import math


class LinkPredictor(BaseModel):

    def __init__(self, config):
        super(LinkPredictor, self).__init__()
        self.ent_emb = nn.Embedding(config['n_ents'], config['ent_embdim'])
        self.ent_argument_bias = nn.Embedding(config['n_ents'], 1)
        self.ent_embdim = config['ent_embdim']
        self.n_rels = config['n_rels']

        self.rescal = nn.Bilinear(
            config['ent_embdim'], config['ent_embdim'], config['n_rels'], bias=False)

        self.sel_pre = nn.Linear(2*config['ent_embdim'], config['n_rels'], bias=False)
        self.init()

    def init(self):
        self.ent_emb.weight.data.uniform_(-0.01, 0.01)
        self.ent_argument_bias.weight.data.fill_(0.0)
        self.rescal.weight.data.normal_(0, math.sqrt(0.1))
        self.sel_pre.weight.data.normal_(0, math.sqrt(0.1))

    def forward(self, inputs, arg_bias):
        # [2Bk] -> [2Bk, D]
        head_emb = self.ent_emb(inputs['head_ent'])
        tail_emb = self.ent_emb(inputs['tail_ent'])

        # [2Bk, D] bilinear [2Bk, D] -> [2Bk, n_rels]
        rescal = self.rescal(head_emb, tail_emb)
        # [2Bk, 2*D] -> [2Bk, n_rels]
        selectional_preferences = self.sel_pre(torch.cat([head_emb, tail_emb], dim=1))

        # [2Bk, n_rels]
        psi = rescal + selectional_preferences
        if arg_bias == 'h':
            head_bias = self.ent_argument_bias(inputs['head_ent'])
            psi += head_bias
        elif arg_bias == 't':
            tail_bias = self.ent_argument_bias(inputs['tail_ent'])
            psi += tail_bias
        elif arg_bias == 'ht':
            head_bias = self.ent_argument_bias(inputs['head_ent'])
            tail_bias = self.ent_argument_bias(inputs['tail_ent'])
            psi = torch.cat([psi+head_bias, psi+tail_bias], dim=0)   

        return psi
