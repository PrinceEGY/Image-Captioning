import os
import urllib
import urllib.request

import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.utils import load_image
from data.preprocessor import Preprocessor
from modules.feature_extractor import FeatureExtractor
from modules.tokenizer import Tokenizer

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
import argparse
from models.rnn_model import RNNImageCaptioner
from keras.applications import EfficientNetV2B3


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image captioning inference")
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        required=True,
        help="Path to the image file or cloud image URI",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/lstm_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-w",
        "--weights_path",
        type=str,
        default="weights/lstm-emb256-rnn1.512-ep_150-loss_1.66.weights.h5",
        help="Path to the model weights",
    )
    parser.add_argument(
        "-m",
        "--gen_method",
        choices=["greedy", "beam"],
        default=["greedy"],
        nargs="+",
        help="Search strategy for caption generation",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generating captions (greedy search only)",
    )
    parser.add_argument(
        "-k",
        "--kbeams",
        type=int,
        default=4,
        help="Number of beams for beam search",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model(weights_path):
    if config["model"]["type"] == "rnn":
        model = RNNImageCaptioner(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            **config["model"]["params"],
        )
        model.build(
            input_shape=((None, *config["model"]["img_features_shape"]), (None, None))
        )
        model.load_weights(weights_path)

    elif config["model"]["type"] == "transformer":
        raise NotImplementedError("Transformer model not implemented")
    return model


def _is_url(path):
    return urllib.parse.urlparse(path).scheme in ("http", "https")


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)

    if _is_url(args.image_path):
        image_path, _ = urllib.request.urlretrieve(args.image_path, "_temp.jpg")
    else:
        image_path = args.image_path

    pooling = None
    if config["model"]["params"]["pooling"] == False:
        pooling = "avg"
    effnet = EfficientNetV2B3(include_top=False, pooling=pooling)
    effnet.trainable = False

    feature_extractor = FeatureExtractor(
        feature_extractor=effnet, features_shape=config["model"]["img_features_shape"]
    )

    tokenizer = Tokenizer.from_vocabulary(
        path="./weights/tokenizer_vocab.pkl",
        standardize=Preprocessor(),
        ragged=True,
    )

    image = load_image(image_path)
    model = load_model(args.weights_path)

    for method in args.gen_method:
        if method == "greedy":
            caption = model.greedy_gen(image, temperature=args.temperature)
        elif method == "beam":
            caption = model.beam_search_gen(image, Kbeams=args.kbeams)
        else:
            raise ValueError(f"Invalid generation method: {method}")
        print(f"Caption generated using '{method}' search strategy:\n{caption[0]}\n")

    if _is_url(args.image_path):
        os.remove(image_path)
