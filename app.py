import os
import streamlit as st
import tensorflow as tf
import yaml
import gdown

from data.preprocessor import Preprocessor
from modules.feature_extractor import FeatureExtractor
from modules.tokenizer import Tokenizer


from models.rnn_model import RNNImageCaptioner
from keras.applications import EfficientNetV2B3
from utils.utils import load_image

WEIGHTS_PATH = "./weights/lstm-emb256-rnn1.512-ep_150-loss_1.66.weights.h5"


def app_render():
    st.title("Image Captioning App")
    st.header("Upload Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.header("Caption Generation Options")
    method = st.radio(
        "Select Generation Method",
        horizontal=True,
        options=["Greedy", "Beam Search"],
        help="""Two algorithms supported for generation\n
1- Greedy Decoding with Temperature Argument: In this approach, the model generates captions by greedily selecting the most probable word at each time step, with the softmax output adjusted by a temperature parameter. This allows for controlling the randomness of word selection during generation. A low temperature (e.g., 0.1) generates more focused and deterministic text, while a high temperature (e.g., 1.0) produces more random and diverse outputs.\n
2- Beam Search with Number of Beams Argument: Beam search is a search algorithm that explores multiple possible sequences simultaneously. The beam width parameter determines the number of sequences the model considers at each step. A higher beam width can lead to more diverse captions but increases computational complexity.
        """,
    )
    if method == "Greedy":
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0)
    elif method == "Beam Search":
        kbeams = st.number_input(
            "Number of Beams (k)", min_value=1, max_value=10, value=3
        )

    if uploaded_image is not None:
        _, center, _ = st.columns(3)
        with center:
            st.image(uploaded_image, width=300)
        image = load_image(uploaded_image.read())
        if st.button("Generate Captions", use_container_width=True):
            info = st.info("Generating Captions...")
            if method == "Greedy":
                caption = model.greedy_gen(image, temperature=temperature)
            elif method == "Beam Search":
                caption = model.beam_search_gen(image, Kbeams=kbeams)

            if caption:
                info.empty()
                st.write("Caption generated: ")
                st.success(caption[0])
            else:
                info.error("Error generating caption")


def download_model():
    if os.path.exists(WEIGHTS_PATH):
        return
    else:
        print("Downloading model weights...")
        gdown.download(
            "https://drive.google.com/file/d/1iZR7CLzlV0H8hMRcSkkJ-t9zl6IWKzHY/view?usp=drive_link",
            WEIGHTS_PATH,
            fuzzy=True,
        )


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


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    download_model()
    config = load_config("./configs/lstm_config.yaml")

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
    model = load_model(WEIGHTS_PATH)
    app_render()
