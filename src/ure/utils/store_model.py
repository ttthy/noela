import json
import logging
import os
import glob
import shutil
import time

import torch

logger = logging.getLogger(__name__)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn"t accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


class ModelSaver(object):

    def __init__(self, checkpoint_dir, max_to_keep=1):
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            logging.info("Checkpoint Directory does not exist! Creating directory {}".format(checkpoint_dir))
            os.mkdir(checkpoint_dir)

        if max_to_keep > 1:
            self.max_to_keep = max_to_keep
        else:
            self.max_to_keep = 1
        self.last_checkpoints = list()
        self.last_best_checkpoints = list()

    def save(self, state, is_best):
        """Saves model and training parameters at checkpoint_dir + "last.pth". If is_best==True, also saves
        checkpoint_dir + "best.pth"

        Args:
            state: (dict) contains model"s state_dict, may contain other keys such as epoch, optimizer state_dict
            is_best: (bool) True if it is the best model seen till now
            checkpoint_dir: (string) folder where parameters are to be saved
            max_to_keep: (int) maximum number of recent checkpoint files to keep.
        """
        checkpoint_dir = self.checkpoint_dir
        # TODO implement max to keep        
        basename = "checkpoint_e{}_s{}.pth".format(state["epoch"], state["global_steps"])
        filepath = os.path.join(checkpoint_dir, basename)
        torch.save(state, filepath)
        self.last_checkpoints.append(filepath)
        if len(self.last_checkpoints) > self.max_to_keep:
            os.remove(self.last_checkpoints.pop(0))

        if is_best:
            bestfp = os.path.join(checkpoint_dir, "best_{}".format(basename))
            self.last_best_checkpoints.append(bestfp)
            # Save best checkpoint
            torch.save(state, bestfp)
            if len(self.last_best_checkpoints) > self.max_to_keep:
                os.remove(self.last_best_checkpoints.pop(0))

    def load(self, model, checkpoint=None, optimizer=None):
        """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
        optimizer assuming it is present in checkpoint.

        Args:
            checkpoint: (string) filename which needs to be loaded
            model: (torch.nn.Module) model for which the parameters are loaded
            optimizer: (torch.optim) optional: resume optimizer from checkpoint
        """
        if checkpoint is not None:
            checkpoint = checkpoint
        else:
            checklist = glob.iglob(os.path.join(self.checkpoint_dir, 'best_checkpoint_*.pth'))
            checkpoint = max(checklist, key=os.path.getctime)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
        logger.info("Load model from checkpoint <{}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        #, map_location='cpu') 
        #, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optim_dict"])
        return model
