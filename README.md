# Image Captioning

Image captioning is a process in which a machine learning model generates textual descriptions or captions for images. It combines computer vision techniques, which allow the model to understand the content of the image, with natural language processing (NLP) techniques, which enable the model to generate coherent and descriptive text.

## Dataset info
[Flicker8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k): consists of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. … The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situation

## Model Info
- An Encoder-Decoder model was used as the archtecture for this project.
- The Encoder network is an feature extractor using a backbone pretraind image model (EffecientNetV2B3 was used but any would work).
- The Decoder network consists of one or more RNN layers (GRU or LSTM).
- Finally output of both are added together to produce probabiltes of the next token
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
## Making inference
Two algorithms supported for generation
1- **Greedy Decoding** with **Temperature** Argument: In this approach, the model generates captions by greedily selecting the most probable word at each time step, with the softmax output adjusted by a temperature parameter. This allows for controlling the randomness of word selection during generation.
A low temperature (e.g., 0.1) generates more focused and deterministic text, while a high temperature (e.g., 1.0) produces more random and diverse outputs.

2- **Beam Search** with **Number of Beams** Argument: Beam search is a search algorithm that explores multiple possible sequences simultaneously. The beam width parameter determines the number of sequences the model considers at each step. A higher beam width can lead to more diverse captions but increases computational complexity.

### Usage
Using cli, **supports images locally or hosted on the web**.
```bash
python inference.py -i (image_path) -m [model arch] -w [checkpoint-path] -g [greedy or beam geneartion] -t [greedy gen temperature]  -k [number of beams]
```

Usage example:
```bash
python inference.py -i "examples\dog.jpg" -g beam -k 4
```
### Some examples
<img alt="examples" src='https://github.com/PrinceEGY/Image-Captioning/blob/main/assets/examples.png?raw=true' height='400'/>
