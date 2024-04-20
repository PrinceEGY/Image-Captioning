import keras


class GenerateText(keras.callbacks.Callback):
    def __init__(self, image):
        self.image = image

    def on_epoch_end(self, epochs=None, logs=None):
        if epochs % 5 == 0:
            print()
            print()
            for t in (0.0, 0.5, 1.0):
                result = self.model.greedy_gen(self.image, temperature=t)
                print(result)
            print()
