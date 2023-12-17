import sys
import torch
import torch.nn as nn

import numpy as np
from ure.nn.base import BaseModel
from ure.nn.encoder import *
from ure.nn.link_predictor import LinkPredictor


############################ Model ###################
class Scorer(BaseModel):
    def __init__(self, config):
        super(Scorer, self).__init__()
        ENCODER = {
            "bow": EmbedBagEncoder,
            "lstm": LSTMEncoder,
            "pcnn": PCNNEncoder,
        }
        self.encoder_name = config["encoder_name"]
        self.encoder = ENCODER[self.encoder_name](config)
        self.scorer = nn.Sequential(
            nn.Dropout(p=config["dropout"]),
            nn.Linear(2*config["fc_dim"], 1))
        # TODO n_cands for inference
        self.n_pos, self.n_neg = config["n_pos"], config["n_neg"]

    def score2relation(self, inputs, scores):
        # TODO: scores of all templates
        # [B, N_CANDs]: scores
        # get the highest score indices
        # [B]
        indices = torch.argmax(scores, dim=1, keepdim=False)
        # get template corresponding to indices
        rel_pred = inputs["candidate2relation"][torch.arange(indices.shape[0]), indices]
        return rel_pred

    def score2relationdist(self, inputs, scores):
        # TODO: scores of all templates
        raise NotImplementedError("score2relationdist")

    def predict_relation(self, inputs):
        scores = self.forward(inputs)
        relations = self.score2relation(inputs, scores)
        return relations

    def forward(self, inputs):
        batch_size = inputs["batch_size"]
        if self.training:
            n_cands = self.n_pos + self.n_neg
        else:
            # TODO n_cands for inference
            n_cands = self.n_pos
        reprs = self.encoder(inputs)

        # [B, N_CANDs, DIM]
        sentences = reprs[:batch_size].unsqueeze(dim=1).repeat(1, n_cands, 1)
        candidates = reprs[batch_size:].view(batch_size, n_cands, -1)        
        scores = self.scorer(torch.cat([sentences, candidates], dim=2)).squeeze(-1)
        return scores


class UREBase(Scorer):
    def __init__(self, config):
        super(UREBase, self).__init__(config)
        self.margin = config["margin"]

    def compute_margin_loss(self, scores):
        # Max-margin loss
        pos_scores = scores[:, :self.n_pos].max(dim=1)[0]
        neg_scores = scores[:, self.n_pos:].max(dim=1)[0]
        diff = neg_scores + self.margin - pos_scores

        margin_loss = torch.where(diff > 0, diff, torch.zeros(diff.shape).cuda()).mean()
        return margin_loss
        
    def compute_loss(self, inputs, scores=None):
        if scores is None:
            scores = self.forward(inputs)

        # Max-margin loss
        margin_loss = self.compute_margin_loss(scores)

        loss = margin_loss
        return loss, {"margin": margin_loss}


class URELP(UREBase):
    def __init__(self, config):
        super(URELP, self).__init__(config)
        self.loss_coef = config["loss_coef"]
        self.link_predictor = LinkPredictor(config)

    def regularizer(self, pred_relations):
        B = pred_relations.shape[0]
        n_rels = pred_relations.shape[1]

        # skewness
        # [B, n_rels]->[B]->1
        loss_s = -(pred_relations * torch.log(pred_relations + 1e-5)).sum(1).mean()

        # dispersion
        # [B, n_rels] -> [n_rels]
        avg = pred_relations.mean(0)
        loss_d = (avg * torch.log(avg + 1e-5)).sum()
        return loss_s, loss_d
        
    def compute_lp_loss(self, inputs, pred_relations):
        k = inputs["head_mention_samples"].shape[1]
        (B, n_rels) = pred_relations.shape

        positive_psi = self.link_predictor({
            "head_mention": inputs["head_mention"],
            "tail_mention": inputs["tail_mention"]
        })

        # [B*k, n_rels]
        negative_psi_head = self.link_predictor({
            # [B*k]
            "head_mention": inputs["head_mention_samples"].flatten(),
            # B -> [B*k]
            "tail_mention": inputs["tail_mention"].unsqueeze(1).repeat(1, k).flatten()
        }, arg_bias="h")

        # [B*k, n_rels]
        negative_psi_tail = self.link_predictor({
            "head_mention": inputs["head_mention"].unsqueeze(1).repeat(1, k).flatten(),
            "tail_mention": inputs["tail_mention_samples"].flatten()
        }, arg_bias="t")

        # [2B, n_rels] -> [2B]
        positive_psi = (pred_relations.repeat(2, 1) * positive_psi).sum(dim=-1)
        # [2B]
        positive_psi = nn.functional.logsigmoid(positive_psi)
        # [2Bk, n_rels] -> [2Bk]
        negative_psi_head = (
            pred_relations.unsqueeze(1)
            * negative_psi_head.view(B, k, n_rels)).sum(dim=-1).flatten()
        negative_psi_head = nn.functional.logsigmoid(-negative_psi_head)
        negative_psi_tail = (
            pred_relations.unsqueeze(1)
            * negative_psi_tail.view(B, k, n_rels)).sum(dim=-1).flatten()
        negative_psi_tail = nn.functional.logsigmoid(-negative_psi_tail)

        loss_lp = -torch.cat(
            [positive_psi, negative_psi_head, negative_psi_tail], dim=0).mean()
        

        return loss_lp

    
    def compute_loss(self, inputs):
        scores = self.forward(inputs)
        margin_loss = self.compute_margin_loss(scores)
        pred_relations = self.score2relationdist(inputs, scores)
        linkpred_loss = self.compute_lp_loss(inputs, pred_relations)
        loss_skewness, loss_dispersion = self.regularizers(pred_relations)

        loss = margin_loss + self.loss_coef["linkpred"] * linkpred_loss \
            + self.loss_coef["skewness"] * loss_skewness \
            + self.loss_coef["dispersion"] * loss_dispersion

        return loss
