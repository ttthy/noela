import argparse
import os
from ure.runner import Trainer, Runner

from ure.loader.config_loader import parse_args, print_config
from ure.loader.data_loader import TACRED
from ure.loader.vocabulary import load_vocabs
from ure.nn.sentence_re import *
from ure.utils.data_utils import get_from_json
from ure.utils.nn_utils import set_seed_everywhere, to_cuda
from ure.utils.store_model import ModelSaver
import ure.utils.tiktok as tiktok
import pickle


def run_test(config, vocas, model, runner):
    print("************************* Testing *************************")
    # Load data
    tiktok.tik("Load test data")
    test_data = TACRED(
        config["path_test"],
        vocas=vocas,
        max_len=config["max_len"],
        batch_size=config["training"]["batch_size"],
        max_position=config["model"]["max_position"],
        is_training=False,
        mask_entity=config["model"]["mask_entity"],
        parse_func="parse_line_w_position"
    )
    tiktok.tok("Load test data")

    tiktok.tik("Testing")
    runner.evaluate(test_data)
    tiktok.tok("Testing")


def test(config):
    # TODO
    tiktok.tik("Load vocas")
    with open(os.path.join(config["output_dir"], "vocas.pkl"), "rb") as f:
        vocas = pickle.load(f)
    tiktok.tok("Load vocas")
    print("\n".join("{}: {}".format(k, v) for k, v in vocas.items()))

    config["model"]["vocab_size"] = vocas["word"].size()
    config["model"]["n_class"] = vocas["relation"].size()

    tiktok.tik("Load model")
    model = eval(config["model_class"])(config["model"])
    to_cuda(model)
    saver = ModelSaver(config["output_dir"], config["training"]["max_checkpoints"])
    model = saver.load(model)
    model.summary()
    tiktok.tok("Load model")

    config["na_id"] = vocas["relation"].get_id(config["na_label"])
    print("NA ID: {}".format(config["na_id"]))


    tiktok.tik("Load_runner")
    runner = Runner(model,
                    config=config)
    tiktok.tok("Load_runner")
    run_test(config, vocas, model, runner)


def train(config):
    # Vocabs
    vocas = load_vocabs(config)
    with open(os.path.join(config["output_dir"], "vocas.pkl"), "wb") as f:
        pickle.dump(vocas, f)

    print("\n----------------------- Vocabulary\n",
          "\n".join("{}: {}".format(k, v) for k, v in vocas.items()),
          "\n------------------------\n\n")

    # Data loading
    tiktok.tik("Load data")
    train_data = TACRED(
        config["path_train"],
        vocas=vocas,
        max_len=config["max_len"],
        batch_size=config["training"]["batch_size"],
        max_position=config["model"]["max_position"],
        is_training=True,
        n_cand=(config["model"]["n_pos"], config["model"]["n_neg"]),
        mask_entity=config["model"]["mask_entity"],
        parse_func=config["parse_funct"])

    valid_data = TACRED(
        config["path_dev"],
        vocas=vocas,
        max_len=config["max_len"],
        batch_size=config["training"]["batch_size"],
        max_position=config["model"]["max_position"],
        is_training=False,
        mask_entity=config["model"]["mask_entity"],
        parse_func="parse_line_w_position")
    tiktok.tok("Load data")

    # Create model graph, print summary
    tiktok.tik("Create model")
    config["model"]["n_class"] = vocas["relation"].size()
    model = eval(config["model_class"])(config["model"])
    to_cuda(model)
    model.summary()
    tiktok.tok("Create model")

    config["na_id"] = vocas["relation"].get_id(config["na_label"])
    print("NA ID: {}".format(config["na_id"]))
    
    trainer = Trainer(model,
                      data_loader={
                          "train": train_data,
                          "test": valid_data
                      },
                      config=config)

    trainer.train()
    print('last_best_checkpoints', trainer.saver.last_best_checkpoints)
    model = trainer.saver.load(
        model, 
        checkpoint=trainer.saver.last_best_checkpoints[-1])
    run_test(config, vocas, model, trainer)


if __name__ == "__main__":
    tiktok.time_reset()

    config = parse_args()
    print_config(config)

    if config["mode"] == "train":
        train(config)
    elif config['mode'] == "test":
        test(config)
    tiktok.tok()
    tiktok.print_time()
