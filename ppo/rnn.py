import tensorflow as tf


class RNN(tf.keras.Model):
    def __init__(self, num_units):
        super(RNN, self).__init__()
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)

    def call(self, inputs, training=False):
        state = self.cell.zero_state(inputs.shape[0], tf.float32)

        outputs = []
        inputs = tf.unstack(inputs, num=inputs.shape[1], axis=1)
        for inp in inputs:
            output, state = self.cell(inp, state)
            outputs.append(output)

        outputs = tf.stack(outputs, axis=1)
        return outputs
