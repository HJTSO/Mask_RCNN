# -*- coding: UTF-8 -*-
import os
import sys
import skimage.io
import skimage.transform
import time
import glob
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

# ● Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs_")

# ● Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model", "mask_rcnn_signature.h5")


class SignatureConfig(Config):
    """Configuration for training on the signature dataset.
    Derives from the base Config class and overrides values specific
    to the toy signature dataset.
    """
    # Give the configuration a recognizable name
    NAME = "signature"

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 1 (GPUs * images/GPU).
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


class EMrcnn:
    """Mask-RCNN to detect signature in receipt image.

    Based on the model which has been trained.

    Attributes:
        config: Configuration for training on the signature dataset.
        self.model: Model object in inference mode.
        self.model.load_weights: Weights trained on MS-COCO.
        self.model.keras_model._make_predict_function()
    """

    def __init__(self):
        self.config = SignatureConfig()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        # keep model loaded while running detection in a web service
        # https://github.com/matterport/Mask_RCNN/issues/600
        self.model.keras_model._make_predict_function()

    def test_image(self, img):
        """Use Mask-RCNN to detect signature in the image.

        Use model.detect functio.

        Args:
            img: The image of Receipt.

        Returns:
            list: The scores of each signature. Set to [0] if signature is not found.
        """
        image = skimage.io.imread(img)
        # image = skimage.transform.rescale(image, 0.3)

        # Run detection
        a = time.time()
        results = self.model.detect([image], verbose=1)
        b = time.time()
        print("● ● ● ● ● Mask-RCNN detect Time:.%s Seconds" % (b - a))

        r = results[0]

        # For show: Visualize results
        # COCO Class names: Index of the class in the list is its ID.
        class_names = ['BG', 'signature']
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

        if r['scores'].size:
            return r['scores'], r['rois']
        else:
            return None, None


if __name__ == '__main__':
    # test
    num = '5'
    directory = os.path.join('/Users/gsl/Desktop/e-mrcnn-GPU-test/test', num)
    print(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)

    mask_rcnn = EMrcnn()
    list_of_files = sorted(glob.glob('/Users/gsl/Desktop/e-mrcnn-GPU-test/test/*.jpg'))

    test_start = time.time()
    print("***** The start time:", test_start)
    for file in list_of_files:
        print("\nImage name:", file)
        scores = mask_rcnn.test_image(file)
        print("Scores:", scores[0])
    test_end = time.time()
    print("***** The end time:", test_end)
    print("***** The testing Time for every image:.%s Seconds" % ((test_end - test_start)/len(list_of_files)))

    image_result = []
    test_list_of_files = sorted(glob.glob('/Users/gsl/Desktop/e-mrcnn-GPU-test/test/' + num + '/*.png'))
    for file in list_of_files:
        image_result.append(skimage.io.imread(file))
