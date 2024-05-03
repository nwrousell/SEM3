import numpy as np
import cv2
import tensorflow as tf


def central_crop(image, crop_height, crop_width):
    # fix this logic maybe its mad wack 
    height, width = image.shape[:2]
    start_x = max(width // 2 - (crop_width // 2), 0)
    start_y = max(height // 2 - (crop_height // 2), 0)
    end_x = min(start_x + crop_width, width)
    end_y = min(start_y + crop_height, height)

    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

def _crop_with_indices(img, x, y, cropped_shape):
    cropped_image = tf.image.crop_to_bounding_box(img, y, x, cropped_shape[-3], cropped_shape[-2])
    return cropped_image

def _per_image_random_crop(img, cropped_shape):
    """Random crop an image."""
    batch_size, width, height = cropped_shape[:-1]
    # size has to be a scalar because tf.image.crop_to_bounding_box doesn't allow for different starting_x, starting_y across a batch
    x = np.random.randint(0, img.shape[1]-width, size=())
    y = np.random.randint(0, img.shape[2]-height, size=())
    return _crop_with_indices(img, x, y, cropped_shape)

def _intensity_aug(x, scale=0.05):
    """Follows the code in Schwarzer et al. (2020) for intensity augmentation."""
    r = np.random.normal(size=(x.shape[0], 1, 1, 1))
    noise = 1.0 + (scale * np.clip(r, -2.0, 2.0))
    return x * noise

def drq_image_aug(obs, img_pad=4):
    """Padding and cropping for DrQ."""
    flat_obs = tf.reshape(obs, (-1, *obs.shape[-3:]))

    paddings = [(0, 0), (img_pad, img_pad), (img_pad, img_pad), (0, 0)]
    cropped_shape = flat_obs.shape
    
    flat_obs = np.pad(flat_obs, paddings, 'edge')
    cropped_obs = _per_image_random_crop(flat_obs, cropped_shape)
    aug_obs = _intensity_aug(cropped_obs)
    return tf.reshape(aug_obs, obs.shape)

def process_inputs(obs, _: bool, scale_type: str): 
    crop_height, crop_width = 84, 84  # Desired crop size 
    # original_height, original_width = 210, 160  # Original image size 

    image, audio = obs
   
    image = image.astype(np.float32) / 255.0 

    # central_fraction = min(crop_height/original_height, crop_width/original_width) 
    if scale_type == 'linear': 
        # linearly scale it 
        image = cv2.resize(image, 
                              (crop_height, crop_width), 
                              interpolation=cv2.INTER_LINEAR) 
    elif scale_type == 'crop': 
        image = central_crop(image, crop_height, crop_width)
    
    audio = audio.astype(np.float32) / 255.0
    

    return image, audio

    
        
  
