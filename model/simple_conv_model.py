import tensorflow as tf


def prepare(class_num, blank_index=0, image_size=(96, None, 3), last_conv_filter=256, bottle_neck=128, gru_unit=256):
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

    x = gru_block(x, gru_unit)
    x = gru_block(x, gru_unit)

    x = tf.keras.layers.Dense(class_num)(x)

    labels = tf.keras.layers.Input((None,), dtype=tf.int32, name="labels")
    label_length = tf.keras.layers.Input((), dtype=tf.int32, name="label_length")
    logit_length = tf.keras.layers.Input((), dtype=tf.int32, name="logit_length")
    x = CtcLossLayer(blank_index=blank_index, class_num=class_num)(inputs=[labels, x, label_length, logit_length])

    model = tf.keras.Model(inputs=[input_image, labels, label_length, logit_length], outputs=x)
    return model


def conv_block(x, filter, is_pool):
    x = tf.keras.layers.Conv2D(filters=filter, kernel_size=3, padding='same')(x)
    if is_pool:
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    return x


def dense_block(x, units):
    x = tf.keras.layers.Dense(units=units)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    return x


def gru_block(x, units):
    x1 = tf.keras.layers.GRU(units, return_sequences=True)
    x2 = tf.keras.layers.GRU(units, return_sequences=True, go_backwards=True)
    x = tf.keras.layers.Bidirectional(x1, backward_layer=x2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


class CtcLossLayer(tf.keras.layers.Layer):
    def __init__(self, blank_index=0, class_num=10):
        self.blank_index = blank_index
        self.class_num = class_num
        super(CtcLossLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        labels, logits, label_length, logit_length = inputs
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, label_length, logit_length, blank_index=self.blank_index,
                                             logits_time_major=False))
        self.add_loss(loss)

        decode, prob = tf.nn.ctc_greedy_decoder(tf.transpose(logits, (1, 0, 2)), logit_length,
                                                blank_index=self.blank_index)
        metric = tf.reduce_mean(
            tf.edit_distance(tf.cast(decode[0], tf.int32), tf.cast(tf.sparse.from_dense(labels), tf.int32)))
        self.add_metric(metric, name='edit_distance')

        return logits
