import tensorflow as tf
import numpy as np
import keras
from .base_model import BaseImageCaptioner, SmartOutput


class RNNImageCaptioner(BaseImageCaptioner):
    def __init__(
        self,
        tokenizer,
        feature_extractor,
        embedding_dim,
        rnn_layers,
        rnn_units,
        dropout_rate=0.4,
        **kwargs
    ):
        super().__init__(tokenizer, feature_extractor, **kwargs)
        self.embedding_dim = embedding_dim
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate

        # Image feaures layers
        self.GAP_layer = keras.layers.GlobalAveragePooling2D()
        self.img_dropout_layer1 = keras.layers.Dropout(0.4)
        self.img_dense_layer1 = keras.layers.Dense(
            rnn_units, activation="relu"
        )  # match the rnn units to add them together later

        # Captions layers
        self.embedding = keras.layers.Embedding(
            input_dim=len(self.tokenizer.get_vocabulary()),
            output_dim=embedding_dim,
            mask_zero=True,
        )
        self.rnn = [
            keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                return_state=True,
                dropout=self.dropout_rate,
            )
            for _ in range(rnn_layers)
        ]

        # Combining image and captions
        self.add_layer = keras.layers.Add()
        self.output_layer = SmartOutput(self.tokenizer)

    def call(self, inputs, initial_state=None, return_state=False, training=False):
        features, caps = inputs

        if self.pooling:
            x1 = self.GAP_layer(x1)
        x1 = self.img_dropout_layer1(x1)
        x1 = self.img_dense_layer1(x1)

        x2 = self.embedding(x2, training=training)

        for rnn_layer in self.rnn:
            if initial_state is None:
                x2, state = rnn_layer(
                    x2,
                    initial_state=x1,
                    training=training,
                )
            else:
                x2, state = rnn_layer(
                    x2,
                    initial_state=initial_state,
                    training=training,
                )

        x3 = self.add_layer([x1, x2])
        x3 = self.output_layer(x3)

        if return_state:
            return x3, state
        else:
            return x3

    def build(self, input_shape):
        out_shape1, out_shape2 = input_shape  # img_features, captions
        if self.pooling:
            self.GAP_layer.build(out_shape1)
        out_shape1 = self.GAP_layer.compute_output_shape(out_shape1)
        self.img_dropout_layer1.build(out_shape1)
        out_shape1 = self.img_dropout_layer1.compute_output_shape(out_shape1)
        self.img_dense_layer1.build(out_shape1)
        out_shape1 = self.img_dense_layer1.compute_output_shape(out_shape1)

        self.embedding.build(caps)
        out_shape2 = self.embedding.compute_output_shape(caps)
        for rnn_layer in self.rnn:
            rnn_layer.build(out_shape2)
            out_shape2 = rnn_layer.compute_output_shape(out_shape2)

        self.add_layer.build(out_shape2)
        out_shape3 = self.add_layer.compute_output_shape(out_shape2)
        self.output_layer.build(out_shape3)

    def greedy_gen(self, images, max_len=30, temperature=0.0):
        if images.ndim == 3:  # set as batch shape
            images = images[tf.newaxis, ...]

        if images.shape[-1] == 3:  # extract the features if the input is an image
            img_features = self.feature_extractor.feature_extractor(images)
        else:
            img_features = images

        batch_size = img_features.shape[0]
        next_token = tf.repeat(self.word_to_index([["<s>"]]), batch_size, 0)
        final_tokens = next_token
        state = None

        for i in range(max_len):
            preds, state = self._one_step_gen(
                (img_features, next_token), initial_state=state
            )
            if temperature == 0:
                next_token = tf.argmax(preds, axis=-1)[:, tf.newaxis]  # (batch, 1)
            else:
                next_token = tf.random.categorical(
                    preds / temperature, num_samples=1
                )  # (batch, 1)

            final_tokens = tf.concat(
                [final_tokens, next_token], axis=1
            )  # (batch, sequence)
            # TODO: early stop if all batch reached EOF

        parsed_sentences = []
        for sen in final_tokens:
            decoded_result = self._decode_tokens(sen[1:-1])
            try:
                eof_idx = decoded_result.split().index("<e>")
                trunced = decoded_result.split()[:eof_idx]
                parsed_sentences.append(" ".join(trunced))
            except ValueError:  # no EOF exist
                parsed_sentences.append(decoded_result)

        return parsed_sentences

    def beam_search_gen(self, images, Kbeams=1, max_len=30):
        if images.ndim == 3:  # set as batch shape
            images = images[tf.newaxis, ...]
        assert (
            images.shape[0] == 1
        ), "Current implementaion only support generation with batch of size 1 only"

        if images.shape[-1] == 3:  # extract the features if the input is an image
            img_features = self.feature_extractor.feature_extractor(images)
        else:
            img_features = images

        initial = self.word_to_index([["<s>"]])
        sequences = [(initial, 0, None)]  # index, prob, state
        completed_sequences = []
        # ONLY SUPPORT BATCH OF SIZE 1
        # TODO: Reimplement to support batched beam
        for i in range(max_len):
            candidate_sequences = []
            for seq, prob, state in sequences:
                preds, state = self._one_step_gen(
                    (img_features, seq[:, -1:]), initial_state=state
                )
                logits = tf.nn.softmax(preds).numpy()
                sorted_logits = np.argsort(logits)
                topk_cands = sorted_logits[:, -Kbeams:]

                for cand in topk_cands[0]:  # cand = index of candidate
                    cand_seq = tf.concat(
                        [seq, np.expand_dims(cand, (0, 1))], axis=1
                    ).numpy()
                    new_prob = prob + np.log(logits[0][cand])
                    candidate_sequences.append((cand_seq, new_prob, state))

            candidate_sequences = sorted(
                candidate_sequences, reverse=True, key=lambda k: k[1]
            )
            # Remove completed sequences from candidates
            completed_inds = [
                idx
                for (idx, seq) in enumerate(candidate_sequences)
                if seq[0][0][-1] == self.word_to_index("<e>")
            ]
            completed_sequences += [candidate_sequences[i] for i in completed_inds]
            candidate_sequences = [
                seq
                for (idx, seq) in enumerate(candidate_sequences)
                if idx not in completed_inds
            ]

            sequences = candidate_sequences[:Kbeams]

        completed_sequences += (
            sequences  # add sequences from last iteration even if they are not complete
        )
        normalized_seqs = [
            (tokens, logits / len(tokens[0]))
            for (tokens, logits, _) in completed_sequences
        ]  # normalize by the length to ensure inferences with differenet sizes treated samely
        topk_completed = sorted(normalized_seqs, key=lambda k: k[1], reverse=True)[
            :Kbeams
        ]

        parsed_sentences = []
        for sen, _ in topk_completed:
            decoded_result = self._decode_tokens(sen[0, 1:-1])
            parsed_sentences.append(decoded_result)

        return parsed_sentences
