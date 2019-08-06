# -*- coding: utf-8 -*-
import os
import sys
import skimage.io
import skimage.transform
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from datetime import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find local version of the library
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize

# ● Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs_")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs_", "shapes20190805T0239", "mask_rcnn_shapes_0030.h5")

# the path of test image
IMAGE_DIR = "/Users/gsl/Desktop/receipt"

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # ● Number of classes (including background)
    NUM_CLASSES = 1 + 1

    # ● the same with training
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10


config = ShapesConfig()
config.display()

class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'signature']
# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]

# ● the image to test
image = skimage.io.imread(os.path.join(IMAGE_DIR, "test_image_2.jpg"))
# image = skimage.transform.rescale(image, 0.3)
# sakai_receipt kikuno_receipt kurosawa test_image_1.jpg

# Run detection
a=datetime.now()
results = model.detect([image], verbose=1)
b=datetime.now()

# Visualize results
print("Time:", (b-a).seconds)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])