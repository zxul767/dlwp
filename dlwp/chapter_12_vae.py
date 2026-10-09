from itertools import product
from typing import TYPE_CHECKING, Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tqdm
from tensorflow import keras
from tensorflow.keras import layers

EncoderShape: TypeAlias = tuple[int | None, int | None, int | None]
if TYPE_CHECKING:
    KerasModel: TypeAlias = keras.Model[Any, Any]
    KerasLayer: TypeAlias = layers.Layer[tuple[tf.Tensor, tf.Tensor], tf.Tensor]
else:
    KerasModel = keras.Model
    KerasLayer = layers.Layer

MNIST_DIGIT_SIZE = 28  # each image is 28x28 pixels
DEFAULT_COLOR_MAP = "Grays_r"


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


def build_encoder(
    *, image_size: tuple[int, int, int], n_latent_dims: int
) -> tuple[KerasModel, EncoderShape]:
    """
    Returns a VAE encoder and the final feature map shape (to be able to create
    the corresponding VAE decoder).

    The VAE encoder takes an input image and outputs the parameters for a normal
    distribution (mu, logvar), which we will use to sample latent vectors.
    """
    # ----------------------------------------------------------------------------------
    # Image features extraction
    # ----------------------------------------------------------------------------------
    encoder_inputs = keras.Input(shape=image_size)
    # `strides=2` downsamples feature maps to half the input size
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    shape = tuple(x.shape)
    # 1) we need this in order to build the corresponding decoder
    # 2) discard the batch dimension (it is always `None`: its value is known until runtime)
    last_feature_map_shape: EncoderShape = (shape[1], shape[2], shape[3])

    x = layers.Flatten()(x)

    # give the encoder additional nonlinear modeling capacity and act as an intermediate
    # bottleneck which encourages the model to retain the most useful, high-level
    # information rather than memorizing every pixel.
    x = layers.Dense(16, activation="relu")(x)

    # ----------------------------------------------------------------------------------
    # Latent output heads
    # ----------------------------------------------------------------------------------
    z_mean = layers.Dense(n_latent_dims, name="z_mean")(x)
    z_log_var = layers.Dense(n_latent_dims, name="z_log_var")(x)

    return (
        keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder"),
        last_feature_map_shape,
    )


def build_decoder(
    *, last_feature_map_shape: EncoderShape, n_latent_dims: int
) -> KerasModel:
    """
    Returns a VAE decoder.

    The VAE decoder takes a latent vector and produces a single-channel image.
    """
    height, width, channels = last_feature_map_shape
    if height is None or width is None or channels is None:
        raise ValueError("Encoder shape must have known spatial dimensions")

    # ----------------------------------------------------------------------------------
    # Final Feature Map Recovery
    # ----------------------------------------------------------------------------------
    n_units = height * width * channels
    latent_inputs = keras.Input(shape=(n_latent_dims,))
    x = layers.Dense(n_units, activation="relu")(latent_inputs)
    x = layers.Reshape((height, width, channels))(x)
    # ----------------------------------------------------------------------------------
    # Upsampling to Recover Original Image
    # (replicate the two convolutional layers in the encoder)
    # ----------------------------------------------------------------------------------
    x = layers.Conv2DTranspose(
        channels, kernel_size=3, activation="relu", strides=2, padding="same"
    )(x)
    x = layers.Conv2DTranspose(
        channels // 2, kernel_size=3, activation="relu", strides=2, padding="same"
    )(x)
    # recover a single-channel image
    decoder_outputs = layers.Conv2D(
        1, kernel_size=3, activation="sigmoid", padding="same"
    )(x)

    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


class VAESampler(KerasLayer):
    """
    Implements the VAE’s reparameterization trick: it samples a latent vector z
    from (mu, logvar), while keeping the operation differentiable.
    """

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch_size, n_latent_dims = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, n_latent_dims))
        # `*` is elementwise, so every latent dimension gets its own learned
        # standard deviation (= `exp(log_variance / 2)`)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(KerasModel):
    """A Variational AutoEncoder."""

    def __init__(self, encoder: KerasModel, decoder: KerasModel, **kwargs: Any):
        super().__init__(**kwargs)
        self.encoder: KerasModel = encoder
        self.decoder: KerasModel = decoder
        self.sampler: VAESampler = VAESampler()
        self.total_loss_tracker: keras.metrics.Mean = keras.metrics.Mean(
            name="total_loss"
        )
        self.reconstruction_loss_tracker: keras.metrics.Mean = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker: keras.metrics.Mean = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        # in each training step, `data` is a batch of images:
        with tf.GradientTape() as tape:
            # 1) encode each image (i.e., map each into a mean and a variance)
            z_mean, z_log_var = self.encoder(data)
            # 2) sample latent vectors from the gaussian distributions
            z = self.sampler((z_mean, z_log_var))
            # 3) decode back the latent vectors into images
            reconstruction = self.decoder(z)
            # 4) compute the losses:
            #    we sum the reconstruction loss over the spatial dimensions and take
            #    its mean over the batch dimension
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            #    add the regularization term (Kullback-Leibler divergence)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)

            # 5) we update the weights using the gradient and do some bookkeeping to
            # track the losses of interest
            grads = tape.gradient(total_loss, self.trainable_weights)
            assert self.optimizer is not None, (
                "an optimizer needs to be set before training!"
            )
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }


def plot_latent_mnist_digit(vae: KerasModel, n_latent_dims: int) -> None:
    """
    Plots a single latent MNIST digit (i.e., a digit image from the VAE's latent space)
    """
    # sample a latent image
    z_sample = tf.random.normal(shape=(1, n_latent_dims), stddev=2.0)
    reconstructed_image = vae.decoder.predict(z_sample, verbose=0)

    # Reshape the image for display
    digit_size = MNIST_DIGIT_SIZE
    reconstructed_image = reconstructed_image[0].reshape(digit_size, digit_size)

    # Display the reconstructed image
    _ = plt.figure(figsize=(5, 5))
    _ = plt.imshow(reconstructed_image, cmap=DEFAULT_COLOR_MAP)
    _ = plt.title(f"Sampled and Reconstructed Image: {z_sample.numpy()}")
    _ = plt.axis("off")
    plt.show()


def plot_latent_mnist_grid(vae: KerasModel, grid_size: int = 30) -> None:
    """
    Plots a grid of latent MNIST digits (i.e., digit images from the VAE's latent space).

    Assumes that `n_latent_dims == 2` because otherwise the visualization would become
    more complicated.
    """
    digit_size = MNIST_DIGIT_SIZE
    figure = np.zeros((digit_size * grid_size, digit_size * grid_size))

    # sample points linearly on an 2d grid
    grid_x = np.linspace(-1, 1, grid_size)
    grid_y = np.linspace(-1, 1, grid_size)[::-1]
    for (i, yi), (j, xj) in tqdm.tqdm(
        product(enumerate(grid_y), enumerate(grid_x)), total=grid_size * grid_size
    ):
        z_sample = np.array([[xj, yi]])
        x_decoded = vae.decoder.predict(z_sample, verbose=0)
        # the decoder always returns a tensor with a batch dimension (of size 1 here)
        # x_decoded.shape == (1, 28, 28, 1)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[
            i * digit_size : (i + 1) * digit_size,
            j * digit_size : (j + 1) * digit_size,
        ] = digit

    _ = plt.figure(figsize=(15, 15))

    # draw axes ticks
    start_range = digit_size // 2
    end_range = start_range + grid_size * digit_size
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    _ = plt.xticks(pixel_range, list(map(str, sample_range_x)))
    _ = plt.yticks(pixel_range, list(map(str, sample_range_y)))

    # draw axes labels
    _ = plt.xlabel("z[0]")
    _ = plt.ylabel("z[1]")
    _ = plt.axis("off")

    # show figure
    _ = plt.imshow(figure, cmap=DEFAULT_COLOR_MAP)
