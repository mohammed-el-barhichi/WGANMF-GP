import tensorflow as tf

#on implemente le loss function WGAN
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)




def loss_function(y_true, y_pred):
    return wasserstein_loss(y_true, y_pred)
