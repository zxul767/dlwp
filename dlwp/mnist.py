import numpy as np
import numpy.typing as npt
from tensorflow import keras

MNIST_DIGIT_SIZE = 28  # each image is 28x28 pixels


def prepare_unlabeled_mnist() -> npt.NDArray[np.float32]:
    """
    Returns all images from the MNIST dataset (training and test samples) for
    self-supervised training (i.e., no labels are included)
    """
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    # since this is self-supervised, we train on all MNIST digits
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    # we add the channel dimension because the encoder expects a 3D tensor
    # we also normalize it to the [0, 1] range for training stability and performance
    mnist_digits = np.expand_dims(mnist_digits, -1).astype(np.float32) / np.float32(255)
    return mnist_digits
