import os
import argparse
import logging
from datetime import datetime
import pytz
import tensorflow as tf

from model import simple_conv_model, simple_tcn_model

tf.get_logger().setLevel('ERROR')
tf.debugging.disable_traceback_filtering()

from data import text_image_dataset

from callbacks import save_callback

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

model_dict = {
    'simple_conv_model': simple_conv_model.prepare,
    'simple_tcn_model': simple_tcn_model.prepare,
}


def export(input_weight_path, char_json_path, model_type, image_height, image_width, output_dir_path):
    classes = text_image_dataset.TextImageDataset.char_json_read(char_json_path)

    # prepare model
    model, saved_model = model_dict[model_type](len(classes), (image_height, image_width, 3))
    model.compile(optimizer=tf.keras.optimizers.SGD())
    saved_model.load_weights(input_weight_path)

    saved_model.save(output_dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--input_weight_path', type=str,
                        default='~/.vaik_text_recognition_pb_trainer/output_model/2023-02-21-23-35-54/step-5000_batch-16_epoch-35_loss_0.3708_val_loss_0.1772/step-5000_batch-16_epoch-35_loss_0.3708_val_loss_0.1772')
    parser.add_argument('--char_json_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data/jpn_character.json'))
    parser.add_argument('--model_type', type=str, default='simple_tcn_model')
    parser.add_argument('--image_height', type=int, default=96)
    parser.add_argument('--image_width', type=int, default=576)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik_text_recognition_pb_trainer/output_fixed_model')
    args = parser.parse_args()

    args.input_weight_path = os.path.expanduser(args.input_weight_path)
    args.char_json_path = os.path.expanduser(args.char_json_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(args.output_dir_path, exist_ok=True)

    export(args.input_weight_path, args.char_json_path, args.model_type, args.image_height, args.image_width,
           args.output_dir_path)
