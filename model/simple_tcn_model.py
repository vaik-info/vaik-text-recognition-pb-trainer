import tensorflow as tf


def prepare(class_num, image_size=(96, 256, 3), last_conv_filter=256, bottle_neck=128, tcn_unit=256):
    input_image = tf.keras.layers.Input(shape=image_size, name="image")

    x = conv_block(input_image, 32, True)
    x = conv_block(x, 64, True)
    x = conv_block(x, 128, False)
    x = conv_block(x, 128, True)
    x = conv_block(x, last_conv_filter, True)

    x = tf.keras.layers.Permute((2, 1, 3))(x)
    x_shape = x.get_shape().as_list()
    x = tf.keras.layers.Reshape((-1, x_shape[2] * x_shape[3]))(x)

    x = dense_block(x, bottle_neck)

    x = tcn_convolution_block(x, tcn_unit)

    x = tf.keras.layers.Dense(class_num)(x)

    labels = tf.keras.layers.Input((None,), dtype=tf.int32, name="labels")
    label_length = tf.keras.layers.Input((), dtype=tf.int32, name="label_length")
    logit_length = tf.keras.layers.Input((), dtype=tf.int32, name="logit_length")
    ctc_output = CtcLossLayer()(inputs=[labels, x, label_length, logit_length])

    model = tf.keras.Model(inputs=[input_image, labels, label_length, logit_length], outputs=ctc_output)
    saved_model = tf.keras.Model(inputs=input_image, outputs=x)
    return model, saved_model

def conv_block(x, filter, is_pool):
    x = tf.keras.layers.Conv2D(filters=filter, kernel_size=3, padding='same')(x)
    if is_pool:
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def dense_block(x, units):
    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def conv1d_block(input, num_filters, dilate):
    x = tf.keras.layers.Conv1D(num_filters, 3, dilation_rate=dilate, padding="same")(input)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(num_filters, 3, dilation_rate=dilate, padding="same")(x)
    input = tf.keras.layers.Conv1D(num_filters, 1, padding="same")(input)
    x = tf.keras.layers.Add()([input, x])
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def tcn_convolution_block(input, num_filters):
    x = conv1d_block(input, num_filters, 1)
    x = conv1d_block(x, num_filters, 2)
    x = conv1d_block(x, num_filters, 4)
    x = conv1d_block(x, num_filters, 8)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


class CtcLossLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CtcLossLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        labels, logits, label_length, logit_length = inputs
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, label_length, logit_length, blank_index=-1,
                                             logits_time_major=False))
        self.add_loss(loss)

        return logits
