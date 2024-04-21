import collections
import tensorflow as tf
import numpy as np
import keras
from tqdm import tqdm


class BaseImageCaptioner(keras.Model):
    def __init__(self, tokenizer, feature_extractor, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.word_to_index = keras.layers.StringLookup(
            vocabulary=tokenizer.get_vocabulary(), mask_token=""
        )
        self.index_to_word = keras.layers.StringLookup(
            vocabulary=tokenizer.get_vocabulary(), mask_token="", invert=True
        )

    def greedy_gen(self, images, max_len=30, temperature=1):
        raise NotImplementedError()

    def beam_search_gen(self, images, K_beam=3, max_len=30):
        raise NotImplementedError()

    def _decode_tokens(self, tokens):
        words = self.index_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=" ")
        decoded_result = result.numpy().decode()
        return decoded_result

    def _one_step_gen(self, inputs, **kwargs):
        preds, *rem = self(
            inputs,
            return_state=True,
            training=False,
            **kwargs,
        )  # (batch, sequence, vocab)
        preds = preds.numpy()[:, -1, :]  # (batch, vocab)
        return preds, *rem


class SmartOutput(keras.Layer):
    def __init__(self, tokenizer, banned_tokens=("", "[UNK]", "<s>"), **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(units=tokenizer.vocabulary_size(), **kwargs)
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens
        self.marignal_bias = self.add_weight(
            shape=(tokenizer.vocabulary_size(),),
            initializer="zeros",
            trainable=False,
        )

    def adapt(self, ds):
        """
        Adapts the initial bias of the dense layer to the provided token counts,
        leading to a more efficient training process.

        This method calculates the optimal initial bias for each output neuron based
        on the marginal entropy of the token distribution. This reduces the initial
        loss from the entropy of the uniform distribution (log(vocabulary_size))
        to the marginal entropy (-p * log(p)), where p is the probability of
        each token.
        """
        counts = collections.Counter()
        vocab_dict = {
            name: id for id, name in enumerate(self.tokenizer.get_vocabulary())
        }

        for tokens in tqdm(ds, desc="Adapting bias... "):
            counts.update(tokens.numpy().flatten())

        counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))
        counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(
            counts.values()
        )

        counts_arr = counts_arr[:]
        for token in self.banned_tokens:
            counts_arr[vocab_dict[token]] = 0

        total = counts_arr.sum()
        p = counts_arr / total
        p[counts_arr == 0] = 1.0
        log_p = np.log(p)  # log(1) == 0

        entropy = -(log_p * p).sum()

        print()
        print(f"Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}")
        print(f"Marginal entropy: {entropy:0.2f}")

        bias = log_p
        bias[counts_arr == 0] = -1e9

        self.marignal_bias.assign(bias)

    def call(self, inputs):
        x = self.dense(inputs)
        return x + self.marignal_bias

    def build(self, input_shape):
        self.dense.build(input_shape)
