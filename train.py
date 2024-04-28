import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import yaml

import tensorflow as tf
import keras


tf.get_logger().setLevel("ERROR")

from keras.applications import EfficientNetV2B3
from data.preprocessor import Preprocessor
from models.rnn_model import RNNImageCaptioner
from modules.callbacks import GenerateText
from modules.feature_extractor import FeatureExtractor
from modules.losses import masked_loss
from modules.tokenizer import Tokenizer
from modules.trainer import Trainer
from utils.utils import load_image
from data.data_loader import Flicker8K


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image captioning training")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Name of the model to train, it will be used to save the model weights and logs",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/lstm_config.yaml",
        help="Path to the configuration file, default to 'configs/lstm_config.yaml'",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="number of epochs to train the model, if not provided, config value will be used",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        help="Verbosity level (0, 1, 2), if not provided, config value will be used",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        help="Directory to save the model weights, if not provided, config value will be used",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_dataset_splits():
    print("Loading dataset...")
    data_loader = Flicker8K(
        "data/flicker8k", tokenizer=tokenizer, feature_extractor=feature_extractor
    )
    train_raw, valid_raw, test_raw = data_loader.get_raw_splits()
    print("Preprocessing dataset...")
    train_ds = data_loader.preprocess_ds(train_raw)
    test_ds = data_loader.preprocess_ds(test_raw)
    valid_ds = data_loader.preprocess_ds(valid_raw)
    return train_ds, valid_ds, test_ds


def create_model():
    print("Creating model...")
    if config["model"]["type"] == "rnn":
        model = RNNImageCaptioner(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            **config["model"]["params"],
        )
        model.build(
            input_shape=((None, *config["model"]["img_features_shape"]), (None, None))
        )

    elif config["model"]["type"] == "transformer":
        raise NotImplementedError("Transformer model not implemented")
    return model


def get_learning_rate():
    if config["train"]["lr_schedule_InverseTimeDecay"]:
        lr = keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=config["train"]["learning_rate"],
            decay_steps=config["train"]["decay_steps_multiple"]
            * config["train"]["steps_per_epoch"],
            decay_rate=config["train"]["decay_rate"],
            staircase=config["train"]["staircase"],
        )
    else:
        lr = config["train"]["learning_rate"]
    return lr


def get_callbacks():
    callbacks = []
    if config["train"]["early_stopping"]["allow"]:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=config["train"]["early_stopping"]["monitor"],
            patience=config["train"]["early_stopping"]["patience"],
            restore_best_weights=config["train"]["early_stopping"][
                "restore_best_weights"
            ],
            verbose=config["train"]["early_stopping"]["verbose"],
        )
        callbacks.append(early_stopping)

    if config["train"]["checkpoint"]["allow"]:
        ckp_name = args.name + "-ep:{epoch:02d}-loss:{loss:.2f}.weights.h5"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, ckp_name),
            verbose=config["train"]["checkpoint"]["verbose"],
            save_freq=config["train"]["steps_per_epoch"]
            * config["train"]["checkpoint"]["save_freq_multiple"],
            save_weights_only=config["train"]["checkpoint"]["save_weights_only"],
        )
        callbacks.append(checkpoint)

    if config["train"]["generate_text"]["allow"]:
        image = load_image(config["train"]["generate_text"]["image_path"])
        generate_text = GenerateText(
            image=image, freq=config["train"]["generate_text"]["freq"]
        )
        callbacks.append(generate_text)
    return callbacks


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    epochs = args.epochs if args.epochs else config["train"]["epochs"]
    verbose = args.verbose if args.verbose else config["train"]["verbose"]
    save_dir = args.save_dir if args.save_dir else config["train"]["save_dir"]

    pooling = None
    if config["model"]["params"]["pooling"] == False:
        pooling = "avg"
    effnet = EfficientNetV2B3(include_top=False, pooling=pooling)
    effnet.trainable = False

    feature_extractor = FeatureExtractor(
        imgs_path="./data/flicker8k/Flicker8k_Dataset",
        feature_extractor=effnet,
    )

    tokenizer = Tokenizer.from_vocabulary(
        path="./weights/tokenizer_vocab.pkl",
        standardize=Preprocessor(),
        ragged=True,
    )

    train_ds, valid_ds, test_ds = get_dataset_splits()

    model = create_model()

    lr = get_learning_rate()
    optimizer = keras.optimizers.Adam(lr)
    callbacks = get_callbacks()

    trainer = Trainer(
        model=model,
        name=args.name,
        optimizer=optimizer,
        loss_fn=masked_loss,
        train_ds=train_ds,
        valid_ds=valid_ds,
        epochs=epochs,
        steps_per_epoch=config["train"]["steps_per_epoch"],
        validation_steps=config["train"]["validation_steps"],
        callbacks=callbacks,
        adapt_bias=config["train"]["adapt_bias"],
        verbose=verbose,
        save_dir=save_dir,
    )
    print("Starting training...")
    trainer.train()
    print("Training finished!", end=" ")
    trainer.save()
