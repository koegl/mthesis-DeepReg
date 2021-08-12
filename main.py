import os
# We import some utility modules.
# import nibabel
import tensorflow as tf
import deepreg
# import deepreg.model.layer as layer
# import deepreg.model.loss.image as image_loss
# from deepreg.model.loss.image import ssd
# import deepreg.model.loss.deform as deform_loss
# import deepreg.model.layer_util as layer_util
import matplotlib.pyplot as plt
import h5py
# import numpy as np

FILE_PATH = "C:\\Users\\fryde\\Documents\\University a\\Master\\Master's thesis\\code\\deepreg\\" \
            "MICCAI_2020_reg_tutorial\\demos\\classical_ct_headandneck_affine\\dataset\\demo.h5"

plt.rcParams["figure.figsize"] = (100, 100)

# open the h5 file
fid = h5py.File(FILE_PATH, "r")
image_fixed = tf.cast(tf.expand_dims(fid["image"], axis=0), dtype=tf.float32)
image_fixed_0 = image_fixed[0, :, :, 0]

label_fixed = tf.cast(tf.expand_dims(fid["label"], axis=0), dtype=tf.float32)
label_fixed_01 = label_fixed[0, :, :, 0, 1]

# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(label_fixed_01)
# axs[1].imshow(image_fixed_0)
# axs[2].imshow(image_fixed_0, cmap='gray')
# axs[2].imshow(label_fixed_01, cmap='jet', alpha=0.5)
# axs[0].set_title("Fixed label")
# axs[1].set_title("Fixed image")
# axs[2].set_title("Overlay")
# plt.show()


# create a random affine transformation
transform_random = layer_util.random_transform_generator(batch_size=1, scale=0.02, seed=4)

rot = tf.convert_to_tensor([[1, 0.1, 0], [0, 1, 0.10], [0, 0, 1], [0, 0, 0]], dtype=tf.float32)
rot = tf.expand_dims(rot, axis=0)

# create a reference grid of image size
grid_ref = layer_util.get_reference_grid(grid_size=label_fixed.shape[1:4])

# warp reference grid with random transform
grid_random = layer_util.warp_grid(grid_ref, rot)#transform_random)

# resample/distort
image_moving = layer_util.resample(vol=image_fixed, loc=grid_random)
image_moving_0 = image_moving[0, :, :, 0]

fig, axs = plt.subplots(2, 3)
axs[0,0].imshow(image_fixed_0, cmap='Greys')
axs[0,1].imshow(image_moving_0, cmap='Greys')
axs[0,2].imshow(image_fixed_0 - image_moving_0, cmap='Greys')
axs[0,0].set_title("Fixed image")
axs[0,1].set_title("Moving image")
axs[0,2].set_title("Difference")
plt.show()

ssd_loss = ssd(np.expand_dims(image_moving, axis=-1), np.expand_dims(image_fixed, axis=-1))
print(ssd_loss)