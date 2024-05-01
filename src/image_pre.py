import numpy as np
import cv2
import tensorflow as tf


# def resize_image(self,image):
    
# Github Paper: https://github.com/spragunr/deep_q_rl/tree/master

# either linearly scales or center crops + the data + the augmentation
# linear: bool -> if False, center crop if True, linear scale

def process_inputs(image, scale_type: str, augmentation = False):
    crop_height, crop_width = 84, 84  # Desired crop size
    original_height, original_width = 210, 160  # Original image size

    image = image.astype(np.float32) / 255.0

    central_fraction = min(crop_height/original_height, crop_width/original_width)
    if scale_type == "linear":
        # linearly scale it 
        return cv2.resize(image,
                              (crop_height, crop_width),
                              interpolation=cv2.INTER_LINEAR)
    elif scale_type == "crop":
        image = tf.expand_dims(image, axis=-1)
        return tf.image.central_crop(image, central_fraction) 
    elif scale_type == "none":
        return image


        # linearly scale it down 
# self.uses_augmentation = False
#         for aug in augmentation:
#             if aug == "affine":
#                 transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
#                 eval_transformation = nn.Identity()
#                 self.uses_augmentation = True
#             elif aug == "crop":
#                 transformation = RandomCrop((84, 84))
#                 # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
#                 eval_transformation = CenterCrop((84, 84))
#                 self.uses_augmentation = True
#                 imagesize = 84
#             elif aug == "rrc":
#                 transformation = RandomResizedCrop((100, 100), (0.8, 1))
#                 eval_transformation = nn.Identity()
#                 self.uses_augmentation = True
#             elif aug == "blur":
#                 transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
#                 eval_transformation = nn.Identity()
#                 self.uses_augmentation = True
#             elif aug == "shift":
#                 transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
#                 eval_transformation = nn.Identity()
#             elif aug == "intensity":
#                 transformation = Intensity(scale=0.05)
#                 eval_transformation = nn.Identity()
#             elif aug == "none":
#                 transformation = eval_transformation = nn.Identity()
#             else:
#                 raise NotImplementedError()
#             self.transforms.append(transformation)
#             self.eval_transforms.append(eval_transformation)        
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