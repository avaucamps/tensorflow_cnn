import tensorflow as tf 
from pathlib import Path
import numpy as np

class DatasetGenerator:
    
    def __init__(self, data_path, is_data_augmentation_enabled, image_size, image_channels, batch_size):
        self.data_path = data_path
        self.is_data_augmentation_enabled = is_data_augmentation_enabled
        self.image_size = image_size
        self.image_channels = image_channels
        self.batch_size = batch_size
        self.all_image_paths = []
        self.all_image_labels = []
        self.image_count = 0


    def get_data(self):
        self.load_images_filenames()
        self.load_labels()

        image_label_dataset = tf.data.Dataset.from_tensor_slices((self.all_image_paths, self.all_image_labels))

        image_label_dataset = image_label_dataset.shuffle(buffer_size=self.image_count)
        image_label_dataset = image_label_dataset.repeat()

        image_label_dataset = image_label_dataset.map(self.parse_image, num_parallel_calls=4)

        if self.is_data_augmentation_enabled:
            image_label_dataset = image_label_dataset.map(self.perform_data_augmentation, num_parallel_calls=4)

        image_label_dataset = image_label_dataset.batch(self.batch_size)
        image_label_dataset = image_label_dataset.prefetch(1)

        return image_label_dataset


    def parse_image(self, filename, label):
        image_string = tf.read_file(filename)

        image = tf.image.decode_jpeg(image_string, channels=self.image_channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, self.image_size)
        
        return image, label

    
    def perform_data_augmentation(self, image, label):
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label


    def load_images_filenames(self):
        self.all_image_paths = list(Path(self.data_path).glob('*/*'))
        self.all_image_paths = [str(path) for path in self.all_image_paths]
        self.image_count = len(self.all_image_paths)


    def load_labels(self):
        label_names = sorted(item.name for item in Path(self.data_path).glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        self.all_image_labels = np.zeros((self.image_count, 12))
        for i in range(self.image_count):
            label = Path(self.all_image_paths[i]).parent.name
            label_index = label_to_index[label]
            label_array = np.zeros((12))
            label_array[label_index] = 1
            self.all_image_labels[i] = label_array