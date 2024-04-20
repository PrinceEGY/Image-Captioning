import string
import tensorflow as tf


class Preprocessor:
    def __init__(self):
        pass

    def preprocess(self, txt):
        txt = self.remove_punctuation(txt)
        txt = self.remove_hangings(txt)
        txt = self.to_lower(txt)
        txt = self.affix_tokens(txt)
        txt = self.remove_consecutive_spaces(txt)
        return txt

    def __call__(self, txt):
        return self.preprocess(txt)

    def remove_hangings(self, txt):
        splitted = tf.strings.split(txt)
        replaced = tf.where(
            tf.strings.length(splitted) > 1, splitted, ""
        )  # leave any word with length more than 1 intact, while replacing the rest to white space
        return tf.strings.reduce_join(replaced, separator=" ", axis=-1)

    def remove_punctuation(self, txt):
        return tf.strings.regex_replace(txt, f"[{string.punctuation}]", "")

    def to_lower(self, txt):
        return tf.strings.lower(txt)

    def affix_tokens(self, txt):
        return tf.strings.join(["<s>", txt, "<e>"], separator=" ")

    def remove_consecutive_spaces(self, txt):
        return tf.strings.regex_replace(txt, r"\s+", " ")
