import os
import argparse
from PIL import Image
import tensorflow as tf

from data import text_image_dataset

def decode2labels(decode, classes, blank_index):
    labels = ""
    for label_index in decode:
        if label_index == blank_index:
            continue
        labels += classes[label_index]
    return labels

def dump(train_font_dir_path, char_json_path, classes_json_path, image_height, image_width, sample_num, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    # train
    TrainDataset = type(f'TrainDataset', (text_image_dataset.TextImageDataset,), dict())
    train_dataset = TrainDataset((image_height, image_width), train_font_dir_path, char_json_path, classes_json_path,
                                 random_text_ratio=0.2, aug_ratio=0.2)

    classes = TrainDataset.get_classes()
    train_dataset = train_dataset.padded_batch(batch_size=1, padding_values=((tf.constant(0, dtype=tf.uint8),
                                                                                       tf.constant(len(classes)-1, dtype=tf.int32),
                                                                                       tf.constant(0, dtype=tf.int32),
                                                                                       tf.constant(0,
                                                                                                   dtype=tf.int32)),))
    train_dataset = iter(train_dataset)
    for index in range(sample_num):
        image, label, _, _ = train_dataset.get_next()[0]
        image = image.numpy()[0]
        label = label.numpy().tolist()[0]
        text = decode2labels(label, classes, len(classes)-1)
        output_image_path = os.path.join(output_dir_path, f'{text}_{index:05d}.jpg')
        Image.fromarray(image).save(output_image_path, quality=100, subsampling=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dump dataset')
    parser.add_argument('--train_font_dir_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data/train_font/font'))
    parser.add_argument('--char_json_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data/jpn_character.json'))
    parser.add_argument('--classes_json_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data/number_plate_address.json'))
    parser.add_argument('--image_height', type=int, default=96)
    parser.add_argument('--image_width', type=int, default=576)
    parser.add_argument('--sample_num', type=int, default=2500)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik_text_recognition_pb_trainer/dump_dataset')
    args = parser.parse_args()

    args.train_font_dir_path = os.path.expanduser(args.train_font_dir_path)
    args.char_json_path = os.path.expanduser(args.char_json_path)
    args.classes_json_path = os.path.expanduser(args.classes_json_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    dump(args.train_font_dir_path, args.char_json_path, args.classes_json_path, args.image_height, args.image_width, args.sample_num, args.output_dir_path)
