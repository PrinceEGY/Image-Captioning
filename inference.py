import os
import urllib
import urllib.request

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

parser = argparse.ArgumentParser(description="Image captioning inference")
parser.add_argument(
    "-i",
    "--image_path",
    type=str,
    required=True,
    help="Path to the image file for caption generation, it can be either local image path or cloud image URI",
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    choices=["rnn", "transformer"],
    default="rnn",
    help="Choose the model you want to make inferences with (RNN model or transformer model)",
)

parser.add_argument(
    "-w",
    "--weights_ckp_path",
    type=str,
    required=True,
    help="Path to the weights of the Keras model (.weight.h5 file)",
)

parser.add_argument(
    "-g",
    "--gen_method",
    choices=["greedy", "beam"],
    default="greedy",
    help="Search strategy (greedy or beam)",
)

parser.add_argument(
    "-t",
    "--temp",
    type=float,
    default=0.0,
    help="Temperature for generating captions (greedy search only)",
)

parser.add_argument(
    "-k",
    "--kbeams",
    type=int,
    default=3,
    help="Number of beams for beam search",
)

args = parser.parse_args()
effnet = EfficientNetV2B3(include_top=False)
effnet.trainable = False

feature_extractor = FeatureExtractor(
    r"data\flicker8k\Flicker8k_Dataset", feature_extractor=effnet
)
feature_extractor.load(r"weights\features.cache.pkl")

preprocess = Preprocessor()
tokenizer = Tokenizer.from_vocabulary(
    path=r"weights\tokenizer_vocab.pkl",
    standardize=preprocess,
    ragged=True,
)


def load_model(weights_path):
    if args.model == "rnn":
        model = RNNImageCaptioner(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            embedding_dim=256,
            rnn_layers=1,
            rnn_units=512,
        )
        model.build(input_shape=((None, 7, 7, 1536), (None, None)))
        model.load_weights(weights_path)
    elif args.model == "transformer":
        raise NotImplementedError("Transformer model not implemented")
    return model


def _is_url(path):
    return urllib.parse.urlparse(path).scheme in ("http", "https")


if __name__ == "__main__":
    if _is_url(args.image_path):
        image_path, _ = urllib.request.urlretrieve(args.image_path, "temp.jpg")
    else:
        image_path = args.image_path

    image = load_image(image_path)
    model = load_model(args.weights_ckp_path)

    if args.gen_method == "greedy":
        caption = model.greedy_gen(image, temperature=args.temp)
    elif args.gen_method == "beam":
        caption = model.beam_search_gen(image, Kbeams=args.kbeams)

    print(caption[0])
