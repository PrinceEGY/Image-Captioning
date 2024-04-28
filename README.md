# Image Captioning

Image captioning is a process in which a machine-learning model generates textual descriptions or captions for images. It combines computer vision techniques, which allow the model to understand the content of the image, with natural language processing (NLP) techniques, which enable the model to generate coherent and descriptive text.

## Dataset info
[Flicker8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k): consists of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. … The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situation

## Model Info
- An Encoder-Decoder model was used as the architecture for this project.
- The Encoder network is a feature extractor using a backbone pretrained image model (EffecientNetV2B3 was used but any would work).
- The Decoder network consists of one or more RNN layers (GRU or LSTM).
- Finally, output of both are added together to produce probabilities of the next token
<img alt="model arch" src='https://github.com/PrinceEGY/Image-Captioning/blob/main/assets/arch.png?raw=true' height='400'/>

## Project Setup
1- Clone this repository:
```bash
git clone https://github.com/PrinceEGY/Image-Captioning.git
cd Image-Captioning
```
2- Set up environment:
```bash
pip install -r requirements.txt
```
3- download pretrained checkpoints:
- LSTM Model checkpoint (name: lstm-emb256-rnn1.512-ep_150-loss_1.66.weights.h5)
```bash
gdown --fuzzy https://drive.google.com/file/d/1iZR7CLzlV0H8hMRcSkkJ-t9zl6IWKzHY/view?usp=drive_link -O .\weights\
```
- GRU Model checkpoint (name: gru-emb256-rnn1.512-ep_150-loss_1.54.weights.h5)
```bash
gdown --fuzzy https://drive.google.com/file/d/12slgnMETzzhDjsVxCegw0YMYshFxV-t6/view?usp=drive_link -O .\weights\
```
## Making inference
Two algorithms supported for generation

1- **Greedy Decoding** with **Temperature** Argument: In this approach, the model generates captions by greedily selecting the most probable word at each time step, with the softmax output adjusted by a temperature parameter. This allows for controlling the randomness of word selection during generation.
A low temperature (e.g., 0.1) generates more focused and deterministic text, while a high temperature (e.g., 1.0) produces more random and diverse outputs.

2- **Beam Search** with **Number of Beams** Argument: Beam search is a search algorithm that explores multiple possible sequences simultaneously. The beam width parameter determines the number of sequences the model considers at each step. A higher beam width can lead to more diverse captions but increases computational complexity.

### Usage
Using CLI, **supports images locally or hosted on the web**.
```bash
python inference.py [-h] -i IMAGE_PATH [-c CONFIG] [-w WEIGHTS_PATH] [-m {greedy,beam} [{greedy,beam} ...]] [-t TEMPERATURE] [-k KBEAMS]
```

Usage example:
```bash
python inference.py --image "examples/dog.jpg" --gen_method beam --kbeams 4
python inference.py -i "examples/three-children.jpg" -c "configs/gru_config.yaml" -w "weights/gru-emb256-rnn1.512-ep_150-loss_1.54.weights.h5" -m greedy beam -t 0.5 -k 5
```

## Training the Model
Setup the model config at `.\configs` and do the follwing.
### Usage
Using CLI, **all optional arguemnts are implicitly inherited from the config file.**
```bash
python train.py [-h] -n NAME -c CONFIG [-e EPOCHS] [-v VERBOSE] [-s SAVE_DIR]
```
Usage example:
```bash
python train.py --name "LSTM-2layers-512units" --config "configs/lstm_config.yaml" --epochs 50
python train.py -n "GRU-1layer-256units" -c "configs/gru_config.yaml" -e 100 -v 0 -s "weights/"
```
Note that the model weights will be automatically saved after training at the `save-dir` location with name=`name`.

## Evaluating the Model
Evaluating the model after training using known metrics. **(NOTE: only BLEU is implemented for now)**
Setup the model config at `.\configs` and do the follwing.
### Usage
Using CLI, **all optional arguemnts are implicitly inherited from the config file.**
```bash
python evaluate.py [-h] -c CONFIG -w WEIGHTS_PATH [-e EVAL_METRICS [EVAL_METRICS ...]] [-m {greedy,beam} [{greedy,beam} ...]] [-t TEMPERATURES [TEMPERATURES ...]] [-k KBEAMS]  [-s SAVE_DIR]
```
Usage example:
```bash
python evaluate.py --config "configs/lstm_config.yaml" --weights_path "weights/lstm-emb256-rnn1.512-ep_150-loss_1.66.weights.h5"
python evaluate.py -c "configs/gru_config.yaml" -w "weights/gru-emb256-rnn1.512-ep_150-loss_1.54.weights.h5" -m greedy beam -t 0 0.5 1 -k 4 -s "results/"
```
Note that all the evaluation results will be automatically saved after evaluation at the `save-dir` location.

### Some examples
<img alt="examples" src='https://github.com/PrinceEGY/Image-Captioning/blob/main/assets/examples.png?raw=true' height='400'/>
