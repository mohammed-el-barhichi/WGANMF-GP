import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(real, fake, autoencoder):
    alpha = tf.random_uniform(shape=[real.shape[0], 1], minval=0., maxval=1.)
    differences = fake - real
    interpolates = real + (alpha * differences)
    gradients = tf.gradients(autoencoder(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    return gradient_penalty


def loss_function(real, fake, autoencoder):
    L=10
    return wasserstein_loss(real, fake) + L * gradient_penalty(real, fake, autoencoder)


