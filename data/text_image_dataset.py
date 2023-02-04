import os
import json
import random
import numpy as np
import tensorflow as tf
from vaik_text_generator import text_generator

from data import ops


class TextImageDataset:
    classes = None
    text_classes = None
    random_text_ratio = None
    text_classes_max_size = None
    a_text_generator = None
    input_size = None
    aug_ratio = None
    feature_divide_num = None
    output_signature = None

    def __new__(cls, input_size=(96, None),
                font_dir_path=os.path.join(os.path.dirname(text_generator.__file__), 'fonts'),
                char_json_path=os.path.join(os.path.dirname(__file__), 'jpn_character.json'),
                classes_json_path=os.path.join(os.path.dirname(__file__), 'number_plate_address.json'),
                random_text_ratio=0.5, aug_ratio=0.5, feature_divide_num=16, font_size_ratio=(64, 128)):
        cls.classes = cls.char_json_read(char_json_path)
        cls.a_text_generator = text_generator.TextGenerator(font_dir_path, font_size_ratio)
        with open(classes_json_path, 'r') as f:
            cls.text_classes = json.load(f)['classes']
        cls.text_classes_max_size = max([len(text_class) for text_class in cls.text_classes])
        cls.random_text_ratio = random_text_ratio
        cls.input_size = input_size
        cls.aug_ratio = aug_ratio
        cls.feature_divide_num = feature_divide_num
        cls.output_signature = (
            (
                tf.TensorSpec(name=f'image', shape=(input_size[0], input_size[1], 3), dtype=tf.uint8),
                tf.TensorSpec(name=f'labels', shape=(None,), dtype=tf.int32),
                tf.TensorSpec(name=f'label_length', shape=(), dtype=tf.int32),
                tf.TensorSpec(name=f'logit_length', shape=(), dtype=tf.int32),
            ),
        )

        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=cls.output_signature
        )

        return dataset

    @classmethod
    def _generator(cls):
        while True:
            if random.uniform(0.0, 1.0) < cls.random_text_ratio:
                text = "".join(
                    [random.choice(cls.classes[:cls.classes.index("弌")]) for _ in range(random.randint(1, cls.text_classes_max_size))])
            else:
                text = random.choice(cls.text_classes)
            text_image = cls.a_text_generator.write(text)
            text_image = 255 - text_image
            class_label_indexes = [cls.classes.index(a_char) for a_char in text]
            if random.uniform(0.0, 1.0) < cls.aug_ratio:
                text_image = cls._data_aug(text_image)
            text_image = ops.random_padding(text_image, random_height_ratio=(0.0, 0.25), random_width_ratio=(0.0, 0.75))
            text_image = ops.data_valid(text_image)
            text_image = ops.resize_upper_height_limit(text_image, cls.input_size[0])
            canvas_image = tf.image.pad_to_bounding_box(text_image, 0, 0, max(1, cls.input_size[0]),
                                                        max(1, text_image.shape[1] + (
                                                                    cls.feature_divide_num - text_image.shape[
                                                                1] % cls.feature_divide_num))).numpy().astype(np.uint8)
            yield (
                (
                    tf.convert_to_tensor(canvas_image, dtype=tf.uint8),
                    tf.convert_to_tensor(class_label_indexes, dtype=tf.int32),
                    tf.convert_to_tensor(len(class_label_indexes), dtype=tf.int32),
                    tf.convert_to_tensor(canvas_image.shape[1]//cls.feature_divide_num, dtype=tf.int32),
                ),
            )

    @classmethod
    def _data_aug(cls, np_image: np.array, random_r_ratio=0.25):
        np_image = ops.random_scale(np_image)
        np_image = ops.random_resize(np_image)
        np_image = ops.random_padding(np_image)
        np_image = ops.random_hsv(np_image, random_ratio=random_r_ratio)
        return np_image

    @classmethod
    def get_all_data(cls, dataset, max_sample_num=100):
        dataset = dataset.padded_batch(batch_size=max_sample_num, padding_values=((tf.constant(0, dtype=tf.uint8),
                                                              tf.constant(0, dtype=tf.int32),
                                                              tf.constant(0, dtype=tf.int32),
                                                              tf.constant(0,
                                                                          dtype=tf.int32)),))
        dataset = iter(dataset)
        data_list = []
        sample_num = 0
        for data in dataset:
            sample_num += 1
            if sample_num > max_sample_num:
                break
            data_list.append(data)
        all_data_list = [None for _ in range(len(data_list[0]))]
        for index in range(len(data_list[0])):
            all_data_list[index] = [data[index] for data in data_list]
        return all_data_list

    @classmethod
    def get_classes(cls):
        return cls.classes

    @classmethod
    def char_json_read(cls, char_json_path):
        with open(char_json_path, 'r') as f:
            json_dict = json.load(f)
        classes = []
        for character_dict in json_dict['character']:
            classes.extend(character_dict['classes'])
        return classes
