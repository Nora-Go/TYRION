from torchvision.transforms import ToPILImage
back = ToPILImage()

IMG_DIR = "target_images"
IMG_MASKS_DIR = "target_masks"
CONTEXT_IMG_DIR = "context_images"
CONTEXT_MASKS_DIR = "context_masks"

CONTEXT = 'image_context'
CONTEXT_OPTICAL = 'image_context_optical'
CONTEXT_SAMPLE = 'image_context_sample'
CONTEXT_TARGET = 'image_context_target'
IMAGE = 'image_target'
IMAGE_SAMPLE = 'image_target_sample'
IMAGE_TARGET = 'image_target_target'
IMAGE_OPTICAL = 'image_target_optical'
IMAGE_TIMESERIES = 'image_target_timeseries'
CONTEXT_MASK = 'mask_context'
IMAGE_MASK = 'mask_target'

STATIC_ZOOM_FACTOR = 7

STONE_ID = 0
NA_AREA_ID = 1
GLACIER_ID = 2
OCEAN_ICE_ID = 3
ID_TO_NAMES = ["Stone", "Na_area","Glacier","Ocean_ice"]

BOUNDING_BOX_DIR = "bounding_boxes"
FRONTS_DIR = "fronts"
SAR_IMAGES_DIR = "sar_images"
ZONES_DIR = "zones"

PARENT_DIR_KEY = "parent_dir"
FRONT_DIR_KEY = "fronts"