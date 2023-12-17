import argparse
import itertools
import logging
import os
import re
import string
import sys
import time
import unicodedata
from collections import Counter, OrderedDict
import pprint

import yaml

from ure.utils.nn_utils import set_seed_everywhere

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u"tag:yaml.org,2002:float",
    re.compile(u"""^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""", re.X),
    list(u"-+0123456789."))

##################  Ordered YAML Loading ##################
# Taken from 
# https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


######################## To files ##############################

class Tee(object):
    """ From http://mail.python.org/pipermail/python-list/2007-May/438106.html """

    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


def save_config(config, config_path):
    with open(config_path, "w") as outfile:
        ordered_dump(config, outfile, default_flow_style=False)


def restore_from_file(config, vocas):
    # Change later
    from ure.loader.vocabulary import Vocabulary
    from ure.utils.tiktok import tik, tok
    # ------------- Build vocabulary
    tik("load_vocab")
    print("Loading word vocab")
    word_vocab = Vocabulary.load(
        os.path.join(config["output_dir"], "word.tsv"),
        lower=config["model"].lowercase,
        digit_0=config["model"].digit_0,
        add_pad_unk=True)
    print("Loading entity set")
    entity_vocab = Vocabulary.load(
        os.path.join(config["output_dir"], "entity.tsv"),
        lower=False, digit_0=False)
    print("Loading tag set")
    tag_vocab = Vocabulary.load(
        os.path.join(config["output_dir"], "tag.tsv"),
        lower=False, digit_0=False)

    # TODO load embeddings
    config["word_vocab"] = word_vocab
    config["tag_vocab"] = tag_vocab
    config["model"].vocab_size = word_vocab.size()
    config["model"].n_labels = tag_vocab.size() + 2  # begin/end
    tok("load_vocab")
    return config


########################## Config loader ############################
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="config file path (default: None)")
    parser.add_argument("-r", "--resume", default=None, type=str,
                        help="path to latest checkpoint (default: None)")
    parser.add_argument("-m", "--mode", default="train", type=str,
                        help="train/test")

    cmd_args = parser.parse_args()
    assert os.path.isfile(cmd_args.config), "Config path is invalid!!!"
    return load_config(cmd_args, is_training=cmd_args.mode == "train")

def load_config(cmd_args, is_training=True):
    """
    Read configuration from config file and returns a dictionary.
    """
    print("Loading configuration from file {}".format(cmd_args.config))
    with open(cmd_args.config, "r") as stream:
        try:
            config = ordered_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # for given parameters overwrite config
    for arg in cmd_args.__dict__:
        if config.get(arg, None) is not None and getattr(cmd_args, arg) is not None:
            config[arg] = getattr(cmd_args, arg)

    assert os.path.isfile(config["path_train"])
    assert os.path.isfile(config["path_test"])
    if "path_pretrained_word" in config and config["path_pretrained_word"] is None:
        assert config["model"]["word_dim"] > 0

    # Check whether to use CUDA and set random seed
    if config["random_seed"] != -1:
        set_seed_everywhere(config["random_seed"])

    STARTED_DATESTRING = time.strftime("-%m-%dT%H-%M-%S", time.localtime())
    config["mode"] = cmd_args.mode
    if is_training:
        if not os.path.exists(config["output_dir"]):
            os.makedirs(config["output_dir"])

        encoder_name = config["model"]["encoder_name"]
        folder_name = "{}-{}-{}".format(
            config["model_class"],
            encoder_name,
            config["model"]["mask_entity"] if config["model"]["mask_entity"] else "womask"
        )
        if config["model_class"].startswith("MIL"):
            folder_name = folder_name + "-{}vs{}".format(
                config["model"]["n_pos"], config["model"]["n_neg"])
        if encoder_name.startswith("bert"):
            if config["model"]["freeze_lm"]:
                folder_name = folder_name + "-freezelm"
        elif encoder_name not in ['etype']:
            folder_name = folder_name + "-{}{}{}".format(
            "pretr" if "path_pretrained_word" in config and config["path_pretrained_word"] else "rand",
            config["model"]["word_dim"],
            "freeze" if config["model"]["freeze_word_emb"] else "")

        folder_name = folder_name + STARTED_DATESTRING
        folder_path = os.path.join(
            config["output_dir"], 
            folder_name
        )
        try:
            os.mkdir(folder_path)
        except FileExistsError:
            pass
        config["output_dir"] = folder_path

        # Save config to output directory for later use
        save_config_path = os.path.join(folder_path, "config.yaml")
        save_config(config, save_config_path)
    else:
        folder_path = config["output_dir"]

    log_file = os.path.join(folder_path, "{}.log".format(cmd_args.mode))
    sys.stdout = Tee(log_file, "w")
    print("Log path: {}".format(folder_path))
    return config


def print_config(config):
    pprint.pprint(config)
    
