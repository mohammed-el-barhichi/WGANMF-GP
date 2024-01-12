import tensorflow as tf

def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

def gradient_penalty(real, fake, autoencoder, L=10):
    # Randomly sample points between real and fake samples
    alpha = tf.random.uniform(shape=[tf.shape(real)[0], 1, 1], minval=0., maxval=1.)
    interpolates = alpha * real + (1 - alpha) * fake
    # Compute the gradients of the critic with respect to the interpolated samples
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        disc_interpolates, _, _ = autoencoder(interpolates)
    gradients = tape.gradient(disc_interpolates, interpolates)
    # Compute the 2-norm of the gradients
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    # Compute the gradient penalty term
    penalty = L * tf.reduce_mean(tf.square(gradients_norm - 1))
    return penalty

def loss_function(real, fake, autoencoder, L=10):
    wasserstein_loss_value = wasserstein_loss(real, fake)
    penalty = gradient_penalty(real, fake, autoencoder, L)
    return wasserstein_loss_value + penalty
