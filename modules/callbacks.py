import keras


class GenerateText(keras.callbacks.Callback):
    def __init__(self, image, freq=5):
        self.image = image
        self.freq = freq

    def on_epoch_end(self, epochs=None, logs=None):
        if epochs % self.freq == 0:
            print()
            print()
            for t in (0.0, 0.5, 1.0):
                result = self.model.greedy_gen(self.image, temperature=t)
                print(result)
            print()
