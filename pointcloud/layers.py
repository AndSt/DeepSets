import tensorflow as tf


class EquiMaxLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim):
        super(EquiMaxLayer, self).__init__()

        self.out_dim = out_dim
        self.gamma = tf.keras.layers.Dense(out_dim, use_bias=True)
        self.lamda = tf.keras.layers.Dense(out_dim, use_bias=False)

    def call(self, x):
        x_max = tf.reduce_max(x, axis=1, keepdims=True)
        x_max = self.lamda(x_max)
        x = self.gamma(x)
        x = x - x_max
        return x


class DeepSets(tf.keras.Model):
    def __init__(self, dim=3):
        super(DeepSets, self).__init__()

        self.phi = tf.keras.Sequential([
            EquiMaxLayer(out_dim=dim),
            tf.keras.layers.Activation('tanh'),
            EquiMaxLayer(out_dim=dim),
            tf.keras.layers.Activation('tanh'),
            EquiMaxLayer(out_dim=dim),
            tf.keras.layers.Activation('tanh'),
        ])

        self.rho = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(dim, activation='tanh'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(dim, activation=None),
        ])

    def call(self, input, training=False):
        phi = self.phi(input)
        phi_max = tf.reduce_max(phi, axis=1, keepdims=True)

        rho = self.rho(phi_max)
        return rho
