import pickle
import tensorflow as tf
import tqdm
from utils.utils import load_image


class FeatureExtractor(tf.keras.Layer):
    def __init__(
        self, imgs_path, feature_extractor=None, batch_size=64, img_shape=(224, 224)
    ):
        super().__init__()
        self.imgs_path = imgs_path
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.features_dict = {}
        self.img_shape = img_shape
        self._features_shape = None

    def adapt(self, ds):
        ds = ds.map(
            lambda img_path, _: (
                self._get_img_id(img_path),
                load_image(img_path, self.img_shape),
            ),
            tf.data.AUTOTUNE,
        ).batch(self.batch_size)

        for imgs_paths, imgs in tqdm(ds, desc="Extracting features..."):
            features = self.feature_extractor.predict(imgs, verbose=0)
            self.features_dict.update(dict(zip(imgs_paths.numpy(), features)))

        self._update_features_shape()

    def call(self, img_path):
        # TODO: support images with no batch shape
        @tf.py_function(Tout=tf.float32)
        def extract_features(img_path):
            img_id = self._get_img_id(img_path).numpy()
            try:
                features = self.features_dict[img_id]
            except KeyError:
                img = load_image(img_path, self.img_shape)
                features = self.feature_extractor.predict(
                    img[tf.newaxis, ...], verbose=0
                )
                features = features[0, ...]  # Remove batch dimension
                self.features_dict[img_id] = features
            return features

        features = extract_features(img_path)
        features.set_shape(self._features_shape)

        return features

    def _get_img_id(self, img_path):
        @tf.py_function(Tout=tf.string)
        def extract_id(img_path):
            return img_path.numpy().decode("utf-8").split("/")[-1]

        img_id = extract_id(img_path)
        img_id.set_shape(img_path.shape)
        return img_id

    def save(self, out_path):
        assert (
            len(self) != 0
        ), "No features exists, Please extract features first using 'adapt' method"
        with open(out_path, "wb") as file:
            pickle.dump(dict(self.features_dict), file)

    def load(self, file_path):
        with open(file_path, "rb") as file:
            self.features_dict = pickle.load(file)
        self._update_features_shape()

    def _update_features_shape(self):
        assert (
            len(self) != 0
        ), "No features exists, Please extract features first using 'adapt' method or 'load' from file"
        sample = list(self.features_dict.values())[0]
        self._features_shape = sample.shape

    def clear(self):
        self.features_dict.clear()

    def __len__(self):
        return len(self.features_dict)
