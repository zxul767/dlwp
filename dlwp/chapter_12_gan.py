from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if TYPE_CHECKING:
    from tensorflow._aliases import TensorCompatible

    KerasModel: TypeAlias = keras.Model[Any, Any]
else:
    TensorCompatible = Any
    KerasModel = keras.Model

DEFAULT_BATCH_SIZE = 128
DEFAULT_N_LATENT_DIMS = 128


def prepare_gan_dataset(
    input_image_size: tuple[int, int], *, batch_size: int = DEFAULT_BATCH_SIZE
) -> tf.data.Dataset[tf.Tensor]:
    dataset = keras.utils.image_dataset_from_directory(
        "celeba_gan",
        label_mode=None,  # only images will be returned, no labels
        image_size=input_image_size,
        batch_size=batch_size,
        # resize using cropping and resizing to preserve aspect ratio
        crop_to_aspect_ratio=True,
    )
    # normalize each image tensor to [0, 1] range
    dataset = dataset.map(normalize_image)
    return dataset


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Normalize an image tensor to the [0, 1] range."""
    return image / 255.0


def make_discriminator_model(
    input_image_size: tuple[int, int],
) -> KerasModel:
    return keras.Sequential(
        [
            keras.Input(shape=(*input_image_size, 3)),
            layers.Conv2D(
                input_image_size[0], kernel_size=4, strides=2, padding="same"
            ),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(
                2 * input_image_size[0], kernel_size=4, strides=2, padding="same"
            ),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(
                2 * input_image_size[0], kernel_size=4, strides=2, padding="same"
            ),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )


def make_generator_model(
    input_image_size: tuple[int, int], n_latent_dims: int = DEFAULT_N_LATENT_DIMS
) -> KerasModel:
    # each convolutional layers reduces the size in half, and there are 3 such layers
    # in the discriminator
    final_image_size = int(input_image_size[0] / 2**3)
    final_layer_size = 2 * input_image_size[0]

    return keras.Sequential(
        [
            keras.Input(shape=(n_latent_dims,)),
            layers.Dense(final_image_size**2 * final_layer_size),
            layers.Reshape((final_image_size, final_image_size, final_layer_size)),
            layers.Conv2DTranspose(
                final_layer_size, kernel_size=4, strides=2, padding="same"
            ),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(
                2 * final_layer_size, kernel_size=4, strides=2, padding="same"
            ),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(
                4 * final_layer_size, kernel_size=4, strides=2, padding="same"
            ),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )


class GAN(KerasModel):
    def __init__(
        self,
        discriminator: KerasModel,
        generator: KerasModel,
        n_latent_dims: int,
    ) -> None:
        super().__init__()
        self.discriminator: KerasModel = discriminator
        self.generator: KerasModel = generator
        self.n_latent_dims: int = n_latent_dims
        self.d_loss_metric: keras.metrics.Mean = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric: keras.metrics.Mean = keras.metrics.Mean(name="g_loss")
        self.d_optimizer: keras.optimizers.Optimizer | None = None
        self.g_optimizer: keras.optimizers.Optimizer | None = None
        self.loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None = None

    def compile(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        d_optimizer: keras.optimizers.Optimizer,
        g_optimizer: keras.optimizers.Optimizer,
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        **kwargs: Any,
    ) -> None:
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        return [self.d_loss_metric, self.g_loss_metric]

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
        mask: TensorCompatible | None = None,
    ) -> tf.Tensor:
        # This method is primarily used for model building and tracing.
        # When `fit` is called with a dataset yielding `real_images`,
        # Keras will invoke `gan.call(real_images)` to build the graph.
        # We need to ensure both sub-models are built during this tracing.

        # Build the discriminator by calling it with the real images.
        _ = self.discriminator(inputs)

        # Build the generator by calling it with dummy latent vectors.
        batch_size = inputs.shape[0] if inputs.shape[0] is not None else 32
        dummy_latent_vectors = tf.zeros(shape=(batch_size, self.n_latent_dims))
        _ = self.generator(dummy_latent_vectors)

        # A GAN model itself does not typically produce a direct output from `real_images`.
        # Returning a dummy tensor satisfies Keras's tracing requirements.
        return tf.constant(0.0)

    def train_step(self, data: TensorCompatible) -> dict[str, tf.Tensor]:
        real_images = cast(tf.Tensor, data)
        assert self.d_optimizer is not None
        assert self.g_optimizer is not None
        assert self.loss_fn is not None
        # Use static batch size to avoid placeholder errors in tf.distribute graph-mode tracing
        batch_size = real_images.shape[0]
        if batch_size is None:
            batch_size = tf.shape(real_images)[0]

        # TRAINING THE DISCRIMINATOR
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.n_latent_dims))
        fake_images = self.generator(random_latent_vectors)
        # combine them with real images
        combined_images = tf.concat([fake_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # adding random noise to the labels is an important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # TRAINING THE GENERATOR
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.n_latent_dims))
        # these labels claim: "these are all real images!" but it's a lie
        misleading_labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, n_images: int = 3, n_latent_dims: int = 128) -> None:
        self.n_images: int = n_images
        self.n_latent_dims: int = n_latent_dims

    def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        random_latent_vectors = tf.random.normal(
            shape=(self.n_images, self.n_latent_dims)
        )
        generated = self.model.generator(random_latent_vectors)
        generated *= 255
        generated.numpy()
        for i in range(self.n_images):
            img = keras.utils.array_to_img(generated[i])
            img.save(f"generated_img_{epoch:03d}_{i}.png")
