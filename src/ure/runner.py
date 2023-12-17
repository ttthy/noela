import logging
import os
import random
import signal
import sys
import pdb
import ure.utils.metrics as module_metrics
from collections import Counter
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ure.loader.optim_loader import get_optimizer
from ure.utils.tiktok import get_time, tik, tok, print_time_n_remove, print_time
from ure.utils.store_model import ModelSaver
from ure.loader.vocabulary import load_pretrained_embeddings, load_vocab


class Runner(object):

    def __init__(self, model, config, data_loader=None):
        super(Runner, self).__init__()
        tik("Build_runner")
        self.model = model
        self.encoder_name = config["model"]["encoder_name"]
        self.na_id = config["na_id"]
        self.data_loader = data_loader
        self.training = False
        self.best_eval_metrics = {}
        self.patience = config["training"]["patience"]
        self.metrics = config["metrics"]
        # Keep track on saving model
        self.saver = ModelSaver(config["output_dir"], config["training"]["max_checkpoints"])
        tok("Build_runner")
    
    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def evaluate(self, dataset=None, data_partition="test", 
                check_is_best=True):
        if dataset is None:
            dataset = self.data_loader["test"]

        target, predictions = [], []

        self.model.eval()
        with torch.no_grad():
            for _, batch_data in enumerate(dataset):
                # TODO check if correct len
                batch_data["data_partition"] = data_partition
                pred = self.model.predict_relation(batch_data).detach().cpu().data.tolist()
                predictions.extend(pred)
                target.extend(batch_data["relation_id"])

        self.check_predictions(predictions, target)
        
        print("------------ Evaluation")
        if check_is_best:
            is_best = False
            for m in self.metrics:
                score = getattr(module_metrics, m)(target, predictions, self.na_id)
                if self.training:
                    if m not in self.best_eval_metrics:
                        score['Epoch'] = self.count_epoch
                        self.best_eval_metrics[m] = score
                    elif module_metrics.compare_metrics(
                        m, score, self.best_eval_metrics[m]):
                        score['Epoch'] = self.count_epoch
                        self.best_eval_metrics[m] = score
                        is_best = True
                print(m, "Best score:" if is_best else "", "\t".join(
                    ["{}={:.2f}".format(k, v*100) for k, v in score.items() if k != 'Epoch']))
        return is_best

    def check_predictions(self, predictions, gold):
        pred_set = Counter(predictions)
        gold_set = Counter(gold)
        print("Diversity [{}]: {}".format(len(pred_set), pred_set))
        # print("Diversity [{}]: {}".format(len(gold_set), gold_set))


class Trainer(Runner):
    def __init__(self, model, config, data_loader):
        super(Trainer, self).__init__(model, config, data_loader)
        tik("Build_trainer")
        self.config = config["training"]
        self.train_size = data_loader["train"].size
        self.test_size = data_loader["test"].size
        self.training = True

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        # Keep this to handle exit signal
        self.count_epoch, self.count_batch, self.count_sample = 0, 0, 0

        self.n_epochs = config["training"]["n_epochs"]
        self.early_stopping = config["training"]["n_epochs"]

        # Optimizer used for training
        optim_fn, optim_params = get_optimizer(config["optimizer"])
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optim_fn(trainable_params, **optim_params)
        tok("Build_trainer")

    def reset_loop(self):
        self.count_batch, self.count_sample = 0, 0

    def handle_signal(self, sig, frame):
        print(
            "\nSIG {} detected at epoch[{}] batch[{}], saving last checkpoint... ".format(
                sig, self.count_epoch+1, self.count_batch+1))
        self.save_checkpoint()
        print("Current log path: ", self.saver.checkpoint_dir)
        tok("Training")
        print_time()
        sys.exit(0)

    def save_checkpoint(self, *args, **kwargs):
        is_best = kwargs.get("is_best", False)

        self.saver.save({
            "epoch": self.count_epoch,
            "global_steps": self.count_batch,
            "encoder_name": self.encoder_name,  # Model name
            "state_dict": self.model.state_dict(),
            "optim_dict": self.optimizer.state_dict(),
        }, is_best)

    def train_one_batch(self, batch_data):
        # compute model output and loss
        scores = self.model(batch_data)
        loss, loss_details = self.model.compute_loss(batch_data, scores)
        # pdb.set_trace()

        # In pytorch, grad will be accumulated
        # so we need to clear previous grads
        self.optimizer.zero_grad()
        # then compute grads of all variables
        loss.backward()
        # performs updates using calculated grads
        self.optimizer.step()

        return loss.data.cpu().item(), loss_details

    def get_cached_embeddings(self, data_partition):
        print("Get cache embeddings from LMs")
        data_loader = self.data_loader[data_partition]
        batch_size = data_loader.batch_size
        data_size = data_loader.size
        self.model.eval()
        with torch.no_grad():
            for start_id in range(0, data_size, batch_size):
                bids = list(range(start_id, min(data_size, start_id+batch_size)))
                batch_data = data_loader.get_batch(bids)
                batch_data["data_partition"] = data_partition
                self.model.get_cached_embeddings(batch_data)
                print("Getting embeddings from {:d}".format(start_id),
                      end="\r" if random.random() < 0.995 else "\n")
            self.model.convert_cached_embeddings_to_tensor(data_partition)

    def train(self):
        tik("Training")
        print("*"*25, "Start training")
        train_data = self.data_loader["train"]
        if self.model.encoder_name.startswith("bert"):
            self.get_cached_embeddings("train")
            self.get_cached_embeddings("test")
        is_best = self.evaluate(dataset=self.data_loader["test"])
        n_epochs = self.config["n_epochs"]
        not_increase_epoch = 0
        while self.count_epoch < n_epochs:
            # Run one epoch
            self.count_epoch += 1
            self.reset_loop()
            cur_epoch = "epoch_{}".format(self.count_epoch)
            tik(cur_epoch)
            print("*****************************  Epoch {}/{} *****************************".format(self.count_epoch, n_epochs))

            # generate data iterator
            self.model.train()
            for _, batch_data in enumerate(train_data):
                # TODO check if correct
                batch_data["epoch"] = self.count_epoch
                cur_bsize = batch_data["batch_size"]
                batch_data["data_partition"] = "train"
                self.count_batch += 1
                self.count_sample += cur_bsize
                loss, loss_details = self.train_one_batch(batch_data)
                s_loss_details = " ".join(["{}={:.5f}".format(k, v.data.cpu().item()) 
                                          for k, v in loss_details.items()])
                # Evaluate
                print("{:d}\tL={:.6f}  {:s}".format(
                      self.count_sample, loss, s_loss_details),
                      end="\r" if random.random() < 0.995 else "\n")
                # pdb.set_trace()
            is_best = self.evaluate(dataset=self.data_loader["test"])
            self.save_checkpoint(is_best=is_best)
            tok(cur_epoch)
            print_time_n_remove(cur_epoch)
            print("-"*40)
            if is_best:
                not_increase_epoch = 0
            else:
                not_increase_epoch += 1
            if not_increase_epoch == self.patience:
                print("Stop after {} epochs!".format(not_increase_epoch))
                break

        print("******************* Best dev scores")
        for m, score in self.best_eval_metrics.items():
            print(m, "\t".join(
                    ["{}={:.2f}".format(k, v*100) 
                     if k != 'Epoch' 
                     else "Epoch {}".format(v) 
                     for k, v in score.items()]))
        tok("Training")
