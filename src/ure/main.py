import argparse
import os
from ure.runner import Trainer, Runner

from ure.loader.config_loader import load_config
from ure.loader.data_loader import NYT
from ure.loader.vocabulary import load_vocabs
from ure.nn.model import UREBase
from ure.utils.data_utils import get_from_json
from ure.utils.nn_utils import set_seed_everywhere, to_cuda
from ure.utils.store_model import ModelSaver
import ure.utils.tiktok as tiktok
import pickle


def run_test(config, vocas, model, runner):
    print("\n", "*"*25, "Run test")

    # Load data
    tiktok.tik("Load test data")
    test_data = NYT(
        config["path_test"],
        vocas=vocas,
        max_len=config["max_len"],
        batch_size=config["training"]["batch_size"],
        max_position=config["model"]["max_position"],
        is_training=False,
        mask_entity=config["model"]["mask_entity"],
        parse_func="parse_line"
    )
    tiktok.tok("Load test data")

    tiktok.tik("Testing")
    runner.evaluate(test_data)
    tiktok.tok("Testing")


def run_test_tacred(config, vocas, model, runner):
    # TODO test on TACRED
    # Load data
    tiktok.tik("Load test data")
    test_data = NYT(
        config["path_test"],
        vocas=vocas,
        max_len=config["max_len"],
        batch_size=config["training"]["batch_size"],
        max_position=config["model"]["max_position"],
        is_training=False,
        mask_entity=config["model"]["mask_entity"],
        parse_func="parse_line"
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

    tiktok.tik("Load model")
    model = UREBase(config["model"])
    to_cuda(model)
    saver = ModelSaver(config["output_dir"], config["training"]["max_checkpoints"])
    model = saver.load(model)
    model.summary()
    tiktok.tok("Load model")

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
    train_data = NYT(
        config["path_train"],
        vocas=vocas,
        max_len=config["max_len"],
        batch_size=config["training"]["batch_size"],
        max_position=config["model"]["max_position"], 
        is_training=True,
        mask_entity=config["model"]["mask_entity"], 
        parse_func="parse_line")

    valid_data = NYT(
        config["path_dev"],
        vocas=vocas,
        max_len=config["max_len"],
        batch_size=config["training"]["batch_size"],
        max_position=config["model"]["max_position"],
        is_training=False,
        mask_entity=config["model"]["mask_entity"],
        parse_func="parse_line")
    tiktok.tok("Load data")

    # Create model graph, print summary
    tiktok.tik("Create model")
    model = UREBase(config["model"])
    to_cuda(model)
    model.summary()
    tiktok.tok("Create model")

    trainer = Trainer(model, 
                      data_loader={
                          "train": train_data,
                          "test": valid_data
                      },
                      config=config)

    trainer.train()

    run_test(config, vocas, model, trainer)


if __name__ == "__main__":
    tiktok.time_reset()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="config file path (default: None)")
    parser.add_argument("-r", "--resume", default=None, type=str,
                        help="path to latest checkpoint (default: None)")
    parser.add_argument("-m", "--mode", default="train", type=str,
                        help="train/test")

    cmd_args = parser.parse_args()
    assert os.path.isfile(cmd_args.config), "Config path is invalid!!!"

    config = load_config(cmd_args, is_training=cmd_args.mode == "train")
    if cmd_args.mode == "train":
        train(config)
    elif cmd_args.mode == "test":
        test(config)
    tiktok.tok()
    tiktok.print_time()
