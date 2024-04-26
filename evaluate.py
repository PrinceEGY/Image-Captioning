import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import yaml

import tensorflow as tf
import keras

tf.get_logger().setLevel("ERROR")

from keras.applications import EfficientNetV2B3
from data.data_loader import Flicker8K
from data.preprocessor import Preprocessor
from models.rnn_model import RNNImageCaptioner
from modules.feature_extractor import FeatureExtractor
from modules.tokenizer import Tokenizer
from modules.evaluator import Evaluator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image captioning evaluation")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=r"configs/lstm_config.yaml",
        help="Path to the configuration file, default to 'configs/lstm_config.yaml'",
    )
    parser.add_argument(
        "-w",
        "--weights_path",
        type=str,
        default=r"weights\lstm-emb256-rnn1.512-ep_150-loss_1.66.weights.h5",
        help="Path to the model weights",
    )
    parser.add_argument(
        "-e",
        "--eval_metrics",
        default=["bleu"],
        nargs="+",
        help="Evaluation metrics to use",
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
        "--temperatures",
        default=[0.0],
        nargs="+",
        help="Temperature for generating captions (greedy search only)",
    )
    parser.add_argument(
        "-k",
        "--kbeams",
        type=int,
        default=4,
        help="Number of beams for beam search",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="./",
        help="Directory to save the model weights, if not provided, cwd will be used",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_test_raw():
    print("Loading dataset...")
    data_loader = Flicker8K(
        r"data\flicker8k", tokenizer=tokenizer, feature_extractor=feature_extractor
    )
    _, _, test_raw = data_loader.get_raw_splits()
    return test_raw


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


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    save_dir = args.save_dir if args.save_dir else config["train"]["save_dir"]

    pooling = None
    if config["model"]["params"]["pooling"] == False:
        pooling = "avg"
    effnet = EfficientNetV2B3(include_top=False, pooling=pooling)
    effnet.trainable = False

    feature_extractor = FeatureExtractor(feature_extractor=effnet)

    if config["model"]["params"]["pooling"] == False:
        feature_extractor.load(r"weights\features-pool.cache.pkl")
    else:
        feature_extractor.load(r"weights\features.cache.pkl")

    tokenizer = Tokenizer.from_vocabulary(
        path=r"weights\tokenizer_vocab.pkl",
        standardize=Preprocessor(),
        ragged=True,
    )

    test_raw = get_test_raw()
    model = load_model(args.weights_path)
    evaluator = Evaluator(model, test_raw.take(5), feature_extractor)
    for metric in args.eval_metrics:
        if metric == "bleu":
            for method in args.gen_method:
                if method == "greedy":
                    evaluator.evaluate_bleu_greedy(
                        temperatures=args.temperatures,
                        output_file="results/bleu_results_greedy.txt",
                    )
                elif method == "beam":
                    evaluator.evaluate_bleu_beam(
                        kbeams=args.kbeams,
                        output_file="results/bleu_results_beam.txt",
                    )
        else:
            raise ValueError(f"Unknown evaluation method: {metric}")
