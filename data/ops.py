import tensorflow as tf
import random


def resize_upper_height_limit(np_image, resize_height):
    np_image = data_valid(np_image)
    if (np_image.shape[0] > resize_height):
        resize_width = max(1, int((np_image.shape[0] / resize_height) * np_image.shape[1]))
        np_image = tf.image.resize(np_image, (resize_height, resize_width)).numpy()
    return np_image


def resize_upper_width_limit(np_image, resize_width):
    np_image = data_valid(np_image)
    if (np_image.shape[1] > resize_width):
        resize_height = max(1, int((np_image.shape[1] / resize_width) * np_image.shape[0]))
        np_image = tf.image.resize(np_image, (resize_height, resize_width)).numpy()
    return np_image


def resize_upper_height_width_limit(np_image, resize_height, resize_width):
    np_image = data_valid(np_image)
    aspect = np_image.shape[1] / np_image.shape[0]
    if resize_width / resize_height >= aspect:
        height = resize_height
        width = round(height * aspect)
    else:
        width = resize_width
        height = round(width / aspect)
    np_image = tf.image.resize(np_image, (height, width)).numpy()
    return np_image


def random_scale(np_image, random_ratio=(0.8, 1.2)):
    np_image = data_valid(np_image)
    max_np_image_size = max(np_image.shape)
    max_np_image_size = max(1, int(max_np_image_size * random.uniform(random_ratio[0], random_ratio[1])))
    resize_image = tf.image.resize(np_image, (max_np_image_size, max_np_image_size), preserve_aspect_ratio=True).numpy()
    return resize_image


def random_resize(np_image, random_trials=2):
    org_np_image_size = np_image.shape
    for trial in range(random_trials):
        if random.uniform(0.0, 1.0) < 0.5:
            if min(np_image.shape[0], np_image.shape[1]) > 2:
                np_image = tf.image.resize(np_image, (int(np_image.shape[0] / 2), int(np_image.shape[1] / 2)))
    np_image = tf.image.resize(np_image, (org_np_image_size[0], org_np_image_size[1])).numpy()
    return np_image


def random_flip(np_image):
    if random.uniform(0.0, 1.0) < 0.5:
        np_image = tf.image.flip_left_right(np_image).numpy()
    return np_image


def random_padding(np_image, random_height_ratio=(0.0, 0.05), random_width_ratio=(0.0, 0.05)):
    left = int(np_image.shape[1] * random.uniform(random_width_ratio[0], random_width_ratio[1]))
    right = int(np_image.shape[1] * random.uniform(random_width_ratio[0], random_width_ratio[1]))
    top = int(np_image.shape[0] * random.uniform(random_height_ratio[0], random_height_ratio[1]))
    bottom = int(np_image.shape[0] * random.uniform(random_height_ratio[0], random_height_ratio[1]))
    np_image = tf.pad(np_image, tf.stack([[top, bottom], [left, right], [0, 0]])).numpy()
    return np_image


def random_hsv(image, max_delta=0.1, lower=2, upper=5, random_ratio=0.25):
    if random.uniform(0.0, 1.0) < random_ratio:
        image = tf.image.random_hue(image, max_delta)
        image = tf.image.random_saturation(image, lower, upper)
        image = image.numpy()
    return image


def data_valid(np_image):
    if 0 in np_image.shape:
        np_image = tf.zeros((1, 1, np_image.shape[-1]), dtype=np_image.dtype)
    return np_image
