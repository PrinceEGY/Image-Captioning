import collections
import tensorflow as tf
import pathlib


class DataLoader:
    def __init__(
        self,
        data_path,
        tokenizer,
        feature_extractor,
        img_size,
        batch_size,
        shuffle_buffer,
        max_tokens,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.max_tokens = max_tokens

    def get_raw_splits(self) -> tf.data.Dataset:
        raise NotImplementedError()

    def preprocess_ds(self, ds):
        ds = (
            ds.map(lambda img_path, caps: (self.feature_extractor(img_path), caps))
            .batch(self.batch_size)
            .map(self.repeat_images, tf.data.AUTOTUNE)
            .unbatch()
            .shuffle(self.shuffle_buffer)
            .batch(self.batch_size)
            .map(self.prepare_entries, tf.data.AUTOTUNE)
        )
        return ds

    def repeat_images(self, image_path, captions):
        """Repeat the image path 5 times to match the captions"""
        return (tf.repeat(image_path, 5, axis=0), tf.reshape(captions, (-1,)))

    def prepare_entries(self, img, cap):
        def pad_seq(
            seq,
        ):  # To be used later if needed, sets all sequences over all batches to the same length
            return tf.pad(
                seq,
                paddings=tf.constant([[0, 0], [0, self.max_tokens]]),
                constant_values=0,
            )

        cap = self.tokenizer(cap)

        input_cap = cap[..., :-1].to_tensor()
        output_cap = cap[..., 1:].to_tensor()
        return (img, input_cap), output_cap


class Flicker8K(DataLoader):
    def __init__(
        self,
        data_path,
        tokenizer,
        feature_extractor,
        img_size=(224, 224),
        batch_size=32,
        shuffle_buffer=2000,
        max_tokens=25,
    ):
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            img_size=img_size,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            max_tokens=max_tokens,
        )

    def get_raw_splits(self):
        path = pathlib.Path(self.data_path)

        if len(list(path.rglob("*"))) < 16197:
            self.download_ds_from_uri()

        captions_path = path / "Flickr8k.token.txt"
        captions = captions_path.read_text().splitlines()
        captions = [line.split("\t") for line in captions]
        captions = ((fname.split("#")[0], caption) for (fname, caption) in captions)

        img_to_caps = collections.defaultdict(list)
        for img_id, cap in captions:
            img_to_caps[img_id].append(cap)

        train_files = (path / "Flickr_8k.trainImages.txt").read_text().splitlines()
        train_captions = [
            (str(path / "Flicker8k_Dataset" / img_id), img_to_caps[img_id])
            for img_id in train_files
        ]

        valid_files = (path / "Flickr_8k.devImages.txt").read_text().splitlines()
        valid_captions = [
            (str(path / "Flicker8k_Dataset" / img_id), img_to_caps[img_id])
            for img_id in valid_files
        ]

        test_files = (path / "Flickr_8k.testImages.txt").read_text().splitlines()
        test_captions = [
            (str(path / "Flicker8k_Dataset" / img_id), img_to_caps[img_id])
            for img_id in test_files
        ]

        train_raw = tf.data.experimental.from_list(train_captions)
        valid_raw = tf.data.experimental.from_list(valid_captions)
        test_raw = tf.data.experimental.from_list(test_captions)
        return (train_raw, valid_raw, test_raw)

    def download_ds_from_uri(self):
        tf.keras.utils.get_file(
            origin="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
            cache_dir=".",
            cache_subdir=self.data_path,
            extract=True,
        )
        tf.keras.utils.get_file(
            origin="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
            cache_dir=".",
            cache_subdir=self.data_path,
            extract=True,
        )
