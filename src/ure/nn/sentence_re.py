import sys
import torch
import torch.nn as nn

import numpy as np
from ure.nn.base import BaseModel
from ure.nn.encoder import *
from ure.nn.link_predictor import LinkPredictor
from ure.utils.nn_utils import to_cuda, to_long_tensor, log_sum_exp
from ure.utils.tiktok import bcolors


############################ Model ###################
class Classifier(BaseModel):
    def __init__(self, config):
        super(Classifier, self).__init__()
        ENCODER = {
            "bow": EmbedBagEncoder,
            "lstm": LSTMEncoder,
            "pcnn": PCNNEncoder,
            "bertmp": BertMentionPooling,
            "bertmpet": BertMentionPoolingEType,
            "bertes": BertEntityStart,
            "berteset": BertEntityStartEType,
            "bertec": BertEntityConcat,
            "bertecet": BertEntityConcatEType,
            "etype": EType,
        }
        self.encoder_name = config["encoder_name"]
        self.encoder = ENCODER[self.encoder_name](config)
        self.n_class = config["n_class"]
        # TODO change config['fc_dim'] to <encoder.output_size>
        self.scorer = nn.Sequential(
            nn.Dropout(p=config["dropout"]),
            # nn.Linear(config["fc_dim"], self.n_class))
            nn.Linear(self.encoder.hidden_size, self.n_class))
        self.use_softmax = config["use_softmax"]

    def get_cached_embeddings(self, inputs):
        self.encoder.get_cached_embeddings(inputs)

    def convert_cached_embeddings_to_tensor(self, data_partition):
        self.encoder.cached_mention_embeddings[data_partition] = torch.cat(self.encoder.cached_mention_embeddings[data_partition], dim=0)

    def remove_cached_embeddings(self, data_partition):
        self.encoder.cached_mention_embeddings.remove(data_partition)
        
    def predict_relation(self, inputs):
        logits = self.forward(inputs)
        # if self.use_softmax:
        logits = nn.functional.softmax(logits, -1)
        scores, predictions = logits.max(-1)
        return predictions
    
    def forward(self, inputs):
        reprs = self.encoder(inputs)
        logits = self.scorer(reprs)
        return logits


class ProbBase(Classifier):
    def __init__(self, config):
        super(ProbBase, self).__init__(config)
        self.n_cand = (config["n_pos"], config["n_neg"])
        self.check_print = False

    def compute_loss(self, inputs, scores):
        if scores is None:
            scores = self.forward(inputs)

        if self.use_softmax:
            scores = torch.softmax(scores, dim=1) + 1e-7
        else:
            scores = torch.log_softmax(scores, dim=1) #+ 1e-7

        # Loglikelihood loss
        batch_size = inputs["batch_size"]
        pos_scores = scores[torch.arange(batch_size).unsqueeze(1).repeat(1, self.n_cand[0]), inputs["pos"]]
        pos_masks = inputs["pos_masks"]
        if self.use_softmax:
            if self.n_cand[0] > 1:
                if inputs["epoch"] < 6:
                    if not self.check_print:
                        print("{}Learn from @1{}".format(bcolors.FAIL, bcolors.ENDC))
                    pos_scores = pos_scores[:, 0].unsqueeze(1)
                elif inputs["epoch"] < 11:
                    pos_scores = pos_scores[:, :2]
                    pos_scores = pos_scores*pos_masks[:, :2]
                else:
                    pos_scores = pos_scores
                    pos_scores = pos_scores*pos_masks
            loss = -torch.log(pos_scores.sum(dim=1)).mean()
        else:
            if not self.check_print:
                print("Log sum exp")
                self.check_print = True
            if self.n_cand[0] > 1:
                if inputs["epoch"] < 5:
                    pos_scores = pos_scores[:, 0].unsqueeze(1)
                elif inputs["epoch"] < 10:
                    pos_scores = pos_scores[:, :2]
                    pos_scores = pos_scores*pos_masks[:, :2] - (1 - pos_masks[:, :2]) * 1e7
                else:
                    pos_scores = pos_scores
                    pos_scores = pos_scores*pos_masks - (1 - pos_masks) * 1e7
            loss = - log_sum_exp(pos_scores, -1).mean()
            if loss < 0:
                import pdb; pdb.set_trace()
        # loss = -torch.log(torch.exp(pos_scores).sum(dim=1)).mean()
        return loss, {"loss": loss}


class MILBase(Classifier):

    def __init__(self, config):
        super(MILBase, self).__init__(config)

        self.margin = config["margin"]
        self.n_cand = (config["n_pos"], config["n_neg"])

    def compute_margin_loss(self, inputs, probs):
        batch_size = inputs["batch_size"]
        
        pos_scores = probs[torch.arange(batch_size).unsqueeze(1).repeat(1, self.n_cand[0]), inputs["pos"]]
        if self.n_cand[0] > 1:
            if inputs["epoch"] < 21:
                pos_scores = pos_scores[:, 0]
            elif inputs["epoch"] < 31:
                pos_scores = pos_scores[:, :2].max(dim=1)[0]
            else:
                pos_scores = pos_scores.max(dim=1)[0]
                
        # pos_scores = pos_scores.max(dim=1)[0]
        neg_scores = probs[torch.arange(batch_size).unsqueeze(1).repeat(1, self.n_cand[1]), inputs["neg"]].max(dim=1)[0]
        
        diff = neg_scores + self.margin - pos_scores
        margin_loss = torch.where(diff > 0, diff, torch.zeros(diff.shape).cuda()).mean()
        
        return margin_loss

    def compute_loss(self, inputs, scores=None):
        if scores is None:
            scores = self.forward(inputs)

        # if self.use_softmax:
        scores = nn.functional.softmax(scores, dim=1) + 1e-7

        # Max-margin loss
        margin_loss = self.compute_margin_loss(inputs, scores)
        loss = margin_loss
        return loss, {"margin": margin_loss}


class MILRegularizer(MILBase):

    def __init__(self, config):
        super(MILRegularizer, self).__init__(config)
    
        self.loss_coef = config["loss_coef"]

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
        
    def compute_loss(self, inputs, scores=None):
        if scores is None:
            scores = self.forward(inputs)

        probs = nn.functional.softmax(scores, dim=1) + 1e-7

        # Max-margin loss
        margin_loss = self.compute_margin_loss(inputs, scores)

        loss_skewness, loss_dispersion = self.regularizer(probs)

        loss = margin_loss \
            + self.loss_coef["skewness"] * loss_skewness \
            + self.loss_coef["dispersion"] * loss_dispersion
        return loss, {"margin": margin_loss, "skewness": loss_skewness, "dispersion": loss_dispersion}


class SentenceRENonNA(Classifier):
    def __init__(self, config):
        super(SentenceRENonNA, self).__init__(config)
        self.batch_kl_input = config["batch_kl_input"]
        self.loss_coef = config["loss_coef"]
        # relation_prior = torch.Tensor(self.n_class).fill_(1. /self.n_class) # uniform dist
        # self.register_buffer('relation_prior', to_cuda(relation_prior))

    def compute_loss(self, inputs, scores):
        batch_size = inputs["batch_size"]
        logits = self.forward(inputs)
        logits = nn.functional.softmax(logits, dim=1) + 1e-7

        # From pre-trained LMs
        relation_dist = nn.functional.softmax(-inputs["relation_dist"], -1) + 1e-7
        
        inst_kl_loss = (logits * torch.log(logits / relation_dist)).sum(1).mean()
        skewness = - (logits * torch.log(logits)).sum(1).mean()

        if self.batch_kl_input == 'encoder':
            mnb_logits = logits.mean(0)
        elif self.batch_kl_input == 'LM':
            mnb_logits = relation_dist.mean(0)
        else:
            raise TypeError("Select `batch_kl_loss` between `encoder` & `LM`")
        batch_kl_loss = (mnb_logits * torch.log(mnb_logits + 1e-7)).sum()

        loss = self.loss_coef["inst_kl"] * inst_kl_loss \
            + self.loss_coef["batch_kl"] * batch_kl_loss \
            + self.loss_coef["skewness"] * skewness

        return loss, {"inst_kl": inst_kl_loss, "batch_kl": batch_kl_loss}


class SentenceRE(Classifier):
    def __init__(self, config):
        super(SentenceRE, self).__init__(config)
        self.batch_kl_input = config["batch_kl_input"]
        self.loss_coef = config["loss_coef"]

        self.na_score = nn.Parameter(torch.Tensor(1))
        relation_prior = torch.Tensor(self.n_class).fill_(
            (1 - config["na_prior"]) / (self.n_class - 1))  # uniform dist
        relation_prior[0] = config["na_prior"]
        self.register_buffer('relation_prior', to_cuda(relation_prior))
        self.reset_parameters()

    def reset_parameters(self):
        self.na_score.data.fill_(1e-5)

    def compute_loss(self, inputs, logits=None):
        batch_size = inputs["batch_size"]
        if logits is None:
            logits = self.forward(inputs)
        logits = nn.functional.softmax(logits, dim=1) + 1e-7

        # From pre-trained LMs
        relation_dist_wo_na = -inputs["relation_dist"]
        # TODO sentence score math.exp
        relation_dist = torch.cat(
            [torch.sigmoid(self.na_score).unsqueeze(0).repeat(batch_size, 1), torch.sigmoid(relation_dist_wo_na)], dim=1)
        relation_dist = nn.functional.softmax(relation_dist, -1) + 1e-7

        inst_kl_loss = (logits * torch.log(logits / relation_dist)).sum(-1).mean()

        if self.batch_kl_input == 'encoder':
            mnb_logits = logits.mean(0)
        elif self.batch_kl_input == 'LM':
            mnb_logits = relation_dist.mean(0)
        else:
            raise TypeError("Select `batch_kl_loss` between `encoder` & `LM`")
        batch_kl_loss = (mnb_logits * torch.log(mnb_logits / self.relation_prior)).sum()

        loss = self.loss_coef["inst_kl"] * inst_kl_loss \
            + self.loss_coef["batch_kl"] * batch_kl_loss

        return loss, {"inst_kl": inst_kl_loss, "batch_kl": batch_kl_loss}


class SupSentenceRE(Classifier):
    def __init__(self, config):
        super(SupSentenceRE, self).__init__(config)

    def compute_loss(self, inputs, logits=None):
        if logits is None:
            logits = self.forward(inputs)
        loss = nn.functional.cross_entropy(logits, to_long_tensor(inputs["relation_id"]))
        return loss, {"CE": loss}
