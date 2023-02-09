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
from vaik_text_generator import text_generator

from callbacks import save_callback

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

model_dict = {
    'simple_conv_model': simple_conv_model.prepare,
    'simple_tcn_model': simple_tcn_model.prepare
}


def train(train_font_dir_path, valid_font_dir_path, char_json_path, classes_json_path,
          model_type, epochs, step_size, batch_size, test_max_sample, image_height, output_dir_path):
    # train
    TrainDataset = type(f'TrainDataset', (text_image_dataset.TextImageDataset,), dict())
    train_dataset = TrainDataset((image_height, None), train_font_dir_path, char_json_path, classes_json_path,
                                 random_text_ratio=0.2, aug_ratio=0.2)
    classes = TrainDataset.get_classes()
    train_dataset = train_dataset.padded_batch(batch_size=batch_size, padding_values=((tf.constant(0, dtype=tf.uint8),
                                                                                       tf.constant(len(classes), dtype=tf.int32),
                                                                                       tf.constant(0, dtype=tf.int32),
                                                                                       tf.constant(0,
                                                                                                   dtype=tf.int32)),))
    # valid
    ValidDataset = type(f'ValidDataset', (text_image_dataset.TextImageDataset,), dict())
    valid_dataset = ValidDataset((image_height, None), valid_font_dir_path, char_json_path, classes_json_path,
                                 random_text_ratio=0.1, aug_ratio=0.1)
    valid_dataset = valid_dataset.padded_batch(batch_size=test_max_sample, padding_values=((tf.constant(0, dtype=tf.uint8),
                                                                                            tf.constant(0, dtype=tf.int32),
                                                                                            tf.constant(0, dtype=tf.int32),
                                                                                            tf.constant(0,
                                                                                                        dtype=tf.int32)),))
    valid_data = next(iter(valid_dataset))

    # prepare model
    model, saved_model = model_dict[model_type](len(classes), (image_height, None, 3))
    model.compile(optimizer=tf.keras.optimizers.SGD())

    # prepare callback
    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')
    prefix = f'step-{step_size}_batch-{batch_size}'
    callback = save_callback.SaveCallback(saved_model=saved_model, save_model_dir_path=save_model_dir_path, prefix=prefix, valid_data=valid_data)

    model.fit_generator(train_dataset, steps_per_epoch=step_size,
                        epochs=epochs,
                        validation_data=valid_data,
                        callbacks=[callback]
                        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--train_font_dir_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data/train_font/font'))
    parser.add_argument('--valid_font_dir_path', type=str,
                        default=os.path.join(os.path.dirname(text_generator.__file__), 'fonts'))
    parser.add_argument('--char_json_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data/jpn_character.json'))
    parser.add_argument('--classes_json_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data/number_plate_address.json'))
    parser.add_argument('--model_type', type=str, default='simple_conv_model')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--step_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_max_sample', type=int, default=100)
    parser.add_argument('--image_height', type=int, default=96)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik_text_recognition_pb_trainer/output_model')
    args = parser.parse_args()

    args.train_font_dir_path = os.path.expanduser(args.train_font_dir_path)
    args.valid_font_dir_path = os.path.expanduser(args.valid_font_dir_path)
    args.char_json_path = os.path.expanduser(args.char_json_path)
    args.classes_json_path = os.path.expanduser(args.classes_json_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(args.output_dir_path, exist_ok=True)

    train(args.train_font_dir_path, args.valid_font_dir_path, args.char_json_path, args.classes_json_path,
          args.model_type, args.epochs, args.step_size, args.batch_size, args.test_max_sample,
          args.image_height, args.output_dir_path)
