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

def process_inputs(image, scale_type: str, augmentation=False): 
    crop_height, crop_width = 84, 84  # Desired crop size 
    # original_height, original_width = 210, 160  # Original image size 

    image = image.astype(np.float32) / 255.0 

    # central_fraction = min(crop_height/original_height, crop_width/original_width) 
    if scale_type == 'linear': 
        # linearly scale it 
        return cv2.resize(image, 
                              (crop_height, crop_width), 
                              interpolation=cv2.INTER_LINEAR) 
    elif scale_type == 'crop': 
        # fix this part of it 
        # image = tf.expand_dims(image, axis=-1) 
        return central_crop(image, crop_height, crop_width)
    else:
        return image;

# self.uses_augmentation = False
#         for aug in augmentation:
#             if aug == "affine":
#                 transformation = random_affine(5, (.14, .14), (.9, 1.1), (-5, 5))
#                 # eval_transformation = nn.Identity()
#                 self.uses_augmentation = True
#             elif aug == "crop":
#                 transformation = random_crop((84, 84))
#                 # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
#                 eval_transformation = central_crop((84, 84))
#                 self.uses_augmentation = True
#                 imagesize = 84
#             elif aug == "rrc":
#                 transformation = random_resized_crop((100, 100), (0.8, 1))
#                 # eval_transformation = nn.Identity()
#                 self.uses_augmentation = True
#             elif aug == "blur":
#                 transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
#                 # eval_transformation = nn.Identity()
#                 self.uses_augmentation = True
#             elif aug == "shift":
#                 image_padded = replication_pad_2d(image, [padding_size, padding_size, padding_size, padding_size])
#                 # Apply RandomCrop
#                 output_image = random_crop(image_padded, [crop_size[0], crop_size[1], tf.shape(image)[2]])  
#                 # transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
#                 # eval_transformation = nn.Identity()
#             elif aug == "intensity":
#                 transformation = Intensity(scale=0.05)
#                 # eval_transformation = nn.Identity()
#             elif aug == "none":
#                 # transformation = eval_transformation = nn.Identity()
#             else:
#                 raise NotImplementedError()
#             self.transforms.append(transformation)
#             self.eval_transforms.append(eval_transformation)    

class Intensity(tf.keras.layers.Layer):
    def __init__(self, scale):
        super(Intensity, self).__init__()
        self.scale = scale

    def call(self, x):
        batch_size = tf.shape(x)[0]
        noise_shape = [batch_size, 1, 1, 1]
        r = tf.random.normal(noise_shape)
        noise = 1.0 + (self.scale * tf.clip_by_value(r, -2.0, 2.0))
        return x * noise       
          
class GaussianBlur2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(5, 5), sigma=(1.5, 1.5)):
        super(GaussianBlur2d, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def call(self, x):
        # Convert input tensor to float32
        x_float32 = tf.cast(x, tf.float32)
        
        # Convert image tensor to NHWC format (batch_size, height, width, channels)
        x_nhwc = tf.expand_dims(x_float32, axis=0)
        
        # Define Gaussian blur filter
        blur_filter = tf.ones((*self.kernel_size, 1, 1), dtype=tf.float32)
        
        # Compute Gaussian kernel
        kernel_h, kernel_w = self.kernel_size
        sigma_h, sigma_w = self.sigma
        grid_x = tf.range(-kernel_w // 2, kernel_w // 2 + 1, dtype=tf.float32)
        grid_y = tf.range(-kernel_h // 2, kernel_h // 2 + 1, dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        kernel = tf.exp(-(grid_x ** 2 / (2 * sigma_w ** 2) + grid_y ** 2 / (2 * sigma_h ** 2)))
        kernel /= tf.reduce_sum(kernel)
        kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
        
        # Apply Gaussian blur filter
        blurred_image = tf.nn.depthwise_conv2d(x_nhwc, kernel, strides=[1, 1, 1, 1], padding="SAME")
        
        # Convert NHWC format back to original shape and type
        blurred_image = tf.squeeze(blurred_image, axis=0)
        blurred_image = tf.cast(blurred_image, x.dtype)

        return blurred_image   

def random_resized_crop(image, output_size):
    # Get random crop parameters
    bbox = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.zeros((0, 0, 4)),  # For whole image
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3),
        area_range=(0.08, 1.0),
        max_attempts=100,
        use_image_if_no_bounding_boxes=True
    )
    # Crop and resize the image
    bbox_begin, bbox_size, _ = bbox
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = output_size
    cropped_image = tf.image.crop_to_bounding_box(
        image,
        offset_y, offset_x,
        bbox_size[0], bbox_size[1]
    )
    resized_image = tf.image.resize(cropped_image, [target_height, target_width])
    return resized_image
        
def random_affine(image, degrees=(-30, 30), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)):
    # Generate random affine transformations
    degrees = tf.random.uniform([], degrees[0], degrees[1])
    translations = tf.random.uniform([2], translate[0], translate[1])
    scales = tf.random.uniform([2], scale[0], scale[1])
    shear = tf.random.uniform([], shear[0], shear[1])
    # Apply affine transformations
    affine_matrix = tf.keras.preprocessing.image.affine_transform_matrix(
        translate=translations,
        shear=shear,
        scale=scales,
        rotation=degrees
    )
    affine_image = tf.keras.preprocessing.image.apply_affine_transform(image, affine_matrix)
    return affine_image
        
def replication_pad_2d(image, padding):
    pad_left, pad_right, pad_top, pad_bottom = padding
    image_padded = tf.pad(image, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='SYMMETRIC')
    return image_padded

def random_crop(image, crop_size):
    cropped_image = tf.image.random_crop(image, size=crop_size)
    return cropped_image
# if the gym environment auto does frame skip 


# ale = ALEInterface()
# ale.act(action)
# rgb_image = ale.getScreenRGB()

# downscales and then stacks the frames (210,160)
# resize the appropriate image
    # def resize_image(self, image):
    #     """ Appropriately resize a single image """

    #     if self.resize_method == 'crop':
    #         # resize keeping aspect ratio
    #         resize_height = int(round(
    #             float(self.height) * self.resized_width / self.width))

    #         resized = cv2.resize(image,
    #                              (self.resized_width, resize_height),
    #                              interpolation=cv2.INTER_LINEAR)

    #         # Crop the part we want
    #         crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
    #         cropped = resized[crop_y_cutoff:
    #                           crop_y_cutoff + self.resized_height, :]

    #         return cropped
    #     elif self.resize_method == 'scale':
    #         return cv2.resize(image,
    #                           (self.resized_width, self.resized_height),
    #                           interpolation=cv2.INTER_LINEAR)
    #     else:
    #         raise ValueError('Unrecognized image resize method.')


# this makes the images gray 

# import sys
# import matplotlib.pyplot as plt
# import cPickle
# import lasagne.layers

# net_file = open(sys.argv[1], 'r')
# network = cPickle.load(net_file)
# print network
# q_layers = lasagne.layers.get_all_layers(network.l_out)
# w = q_layers[1].W.get_value()
# count = 1
# for f in range(w.shape[0]): # filters
#     for c in range(w.shape[1]): # channels/time-steps
#         plt.subplot(w.shape[0], w.shape[1], count)
#         img = w[f, c, :, :]
#         plt.imshow(img, vmin=img.min(), vmax=img.max(),
#                    interpolation='none', cmap='gray')
#         plt.xticks(())
#         plt.yticks(())
#         count += 1