import keras
import pickle


class Tokenizer(keras.layers.TextVectorization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_vocabulary(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_vocabulary(), f)

    @classmethod
    def from_vocabulary(cls, path, **kwargs):
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        return cls(vocabulary=vocab, **kwargs)
