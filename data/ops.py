import tensorflow as tf
import random

def resize_upper_height_limit(np_image, resize_height):
    if (np_image.shape[0] > resize_height):
        resize_width = int((np_image.shape[0]/resize_height) * np_image.shape[1])
        np_image = tf.image.resize(np_image, (resize_height, resize_width)).numpy()
    return np_image

def random_scale(np_image, random_ratio=(0.8, 1.2)):
    max_np_image_size = max(np_image.shape)
    max_np_image_size = max(1, int(max_np_image_size * random.uniform(random_ratio[0], random_ratio[1])))
    resize_image = tf.image.resize(np_image, (max_np_image_size, max_np_image_size), preserve_aspect_ratio=True).numpy()
    return resize_image

def random_resize(np_image, random_trials=2):
    org_np_image_size = np_image.shape
    for trial in range(random_trials):
        if random.uniform(0.0, 1.0) < 0.5:
            if min(np_image.shape[0], np_image.shape[1]) > 2:
                np_image = tf.image.resize(np_image, (int(np_image.shape[0]/2), int(np_image.shape[1]/2)))
    np_image = tf.image.resize(np_image, (org_np_image_size[0], org_np_image_size[1])).numpy()
    return np_image

def random_flip(np_image):
    if random.uniform(0.0, 1.0) < 0.5:
        np_image = tf.image.flip_left_right(np_image).numpy()
    return np_image

def random_padding(np_image, random_ratio=(0.0, 0.05)):
    left = int(np_image.shape[1] * random.uniform(random_ratio[0], random_ratio[1]))
    right = int(np_image.shape[1] * random.uniform(random_ratio[0], random_ratio[1]))
    top = int(np_image.shape[0] * random.uniform(random_ratio[0], random_ratio[1]))
    bottom = int(np_image.shape[0] * random.uniform(random_ratio[0], random_ratio[1]))
    np_image = tf.pad(np_image, tf.stack([[top, bottom], [left, right], [0, 0]])).numpy()
    return np_image

def random_hsv(image, max_delta=0.1, lower=2, upper=5, random_ratio=0.25):
    if random.uniform(0.0, 1.0) < random_ratio:
        image = tf.image.random_hue(image, max_delta)
        image = tf.image.random_saturation(image, lower, upper)
        image = image.numpy()
    return image