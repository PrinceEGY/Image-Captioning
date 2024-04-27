import matplotlib.pyplot as plt
import keras
import os

keras.utils.set_random_seed(2024)


class Trainer:
    def __init__(
        self,
        model,
        name,
        optimizer,
        loss_fn,
        train_ds,
        valid_ds,
        epochs,
        steps_per_epoch,
        validation_steps,
        callbacks,
        adapt_bias=True,
        verbose=1,
        save_dir="weights",
    ):
        self.model = model
        self.name = name
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.callbacks = callbacks
        self.adapt_bias = adapt_bias
        self.verbose = verbose
        self.save_dir = save_dir

    def train(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        if self.adapt_bias:
            self.model.output_layer.adapt(
                self.train_ds.map(lambda images, labels: labels)
            )
        history = self.model.fit(
            self.train_ds,
            validation_data=self.valid_ds,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )
        loss_path = os.path.join("results", self.name)
        self._save_loss_plot(history, save_path=loss_path)

    def save(self, save_weights_only=True):
        path = os.path.join(
            self.save_dir, self._parse_name(self.name, save_weights_only)
        )
        if save_weights_only:
            self.model.save_weights(path)
        else:
            self.model.save(path)
        print(f"Model saved at {path}")

    def _save_loss_plot(self, history, save_path="results"):
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel("Epoch #")
        plt.ylabel("Cross Entropy/token")
        plt.legend()
        plt.savefig(save_path + "-loss_plot.png")

    def _parse_name(self, save_dir: str, save_weights_only: bool) -> str:
        if save_weights_only:
            if save_dir.endswith(".weights.h5"):
                return save_dir
            elif save_dir.endswith(".h5"):
                return save_dir[:-3] + ".weights.h5"
            else:
                return save_dir + ".weights.h5"
        else:
            if save_dir.endswith(".keras"):
                return save_dir
            else:
                return save_dir + ".keras"
