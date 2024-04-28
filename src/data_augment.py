import numpy as np
import tensorflow as tf 
# --------------------------- < Data Augmentation >-----------------------------


def _random_crop(key, img, cropped_shape):
    """Random crop an image."""
    _, width, height = cropped_shape[:-1]
    x = np.random.randint(minval=0, maxval=img.shape[1] - width,shape=())
    y = np.random.randint(minval=0, maxval=img.shape[2] - height,shape=())
    return img[:, x:x + width, y:y + height]


# @functools.partial(jax.jit, static_argnums=(3,))
# @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def _crop_with_indices(img, x, y, cropped_shape):
    start_indices = tf.concat([[x, y, 0], tf.zeros(tf.rank(cropped_shape) - 1, dtype=tf.int32)], axis=0)
    #   cropped_image = jax.lax.dynamic_slice(img, [x, y, 0], cropped_shape[1:])
    cropped_image = tf.slice(img, begin=start_indices, size=cropped_shape[1:])
    return cropped_image


def _per_image_random_crop(key, img, cropped_shape):
    """Random crop an image."""
    batch_size, width, height = cropped_shape[:-1]
    #   key_x, key_y = np.random.split(key, 2)
    x = np.random.randint(minval=0, maxval=img.shape[1] - width,shape=(batch_size,))
    y = np.random.randint(minval=0, maxval=img.shape[2] - height,shape=(batch_size,))
    return _crop_with_indices(img, x, y, cropped_shape)


def _intensity_aug(key, x, scale=0.05):
    """Follows the code in Schwarzer et al. (2020) for intensity augmentation."""
    r = np.random.normal(key, shape=(x.shape[0], 1, 1, 1))
    noise = 1.0 + (scale * tf.clip(r, -2.0, 2.0))
    return x * noise


# @functools.partial(jax.jit)
def drq_image_aug(key, obs, img_pad=4):
    """Padding and cropping for DrQ."""
    flat_obs = obs.reshape(-1, *obs.shape[-3:])
    paddings = [(0, 0), (img_pad, img_pad), (img_pad, img_pad), (0, 0)]
    cropped_shape = flat_obs.shape
    # The reference uses ReplicationPad2d in pytorch, but it is not available
    # in Jax. Use 'edge' instead.
    flat_obs = tf.pad(flat_obs, paddings, 'edge')
    key1, key2 = np.random.split(key, num=2)
    cropped_obs = _per_image_random_crop(key2, flat_obs, cropped_shape)
    # cropped_obs = _random_crop(key2, flat_obs, cropped_shape)
    aug_obs = _intensity_aug(key1, cropped_obs)
    return aug_obs.reshape(*obs.shape)

