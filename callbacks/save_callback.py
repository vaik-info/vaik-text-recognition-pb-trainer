import os
import tensorflow as tf

class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, saved_model, save_model_dir_path, prefix, valid_data):
        super(SaveCallback, self).__init__()
        os.makedirs(save_model_dir_path, exist_ok=True)

        self.saved_model = saved_model
        self.save_model_dir_path = save_model_dir_path
        self.prefix = prefix
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        loss_string = "_".join([f'{k}_{v:.4f}' for k, v in logs.items()])
        save_model_name = f'{self.prefix}_epoch-{epoch}_{loss_string}'
        output_model_dir_path = os.path.join(self.save_model_dir_path, save_model_name)
        os.makedirs(output_model_dir_path, exist_ok=True)
        self.saved_model.save(output_model_dir_path)