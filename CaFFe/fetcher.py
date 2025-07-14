import os.path
import PIL
from PIL import Image
import pickle
from tqdm import tqdm
from CaFFe.utils import *
import numpy as np
from CaFFe.constants import *


def get_nr_of_patches_for_images(img, patch_size, context_size):
    if len(img.shape) == 3:
        _, H, W = img.shape
    else:
        H, W = img.shape

    extra_add = -1
    if ((H + 1) // patch_size * patch_size + context_size) <= H and (
            ((W + 1) // patch_size * patch_size + context_size) <= W):
        extra_add = 0

    HH = (H + 1) // patch_size + extra_add
    WW = (W + 1) // patch_size + extra_add

    return HH * WW


def fetch_patches(parent_dir, split, patch_size, context_size, automatic_resizing=False):
    if split == "train" or split == "val":
        split_idx = pickle.load(open(os.path.join(parent_dir, split + "_idx.txt"), "rb"))
        split = "train"

    else:
        split_sar_dir = os.path.join(parent_dir, SAR_IMAGES_DIR, split)
        n = len(os.listdir(split_sar_dir))
        split_idx = np.arange(0, n)

    sample_names = []
    meta_data = dict()

    split_zones_dir = os.path.join(parent_dir, ZONES_DIR, split)
    split_sar_dir = os.path.join(parent_dir, SAR_IMAGES_DIR, split)

    all_files = os.listdir(split_sar_dir)
    all_files.sort()

    for file_idx in tqdm(split_idx):
        file_name = all_files[file_idx]
        name = file_name[:-4]
        img = Image.open(os.path.join(split_sar_dir, file_name)).convert('L')
        mask = Image.open(os.path.join(split_zones_dir, name + "_zones" + ".png")).convert('L')
        if automatic_resizing:
            resolution_factor = int(name.split('_')[-3])
            rescaling_factor = resolution_factor / STATIC_ZOOM_FACTOR
            rescaled_height = int(rescaling_factor * img.height)
            rescaled_width = int(rescaling_factor * img.width)

            img = img.resize((rescaled_width, rescaled_height), resample=PIL.Image.BICUBIC)
            mask = mask.resize((rescaled_width, rescaled_height), resample=PIL.Image.NEAREST)

        img = preprocess(pad_whole_image(img, patch_size))
        mask = mask_preprocess(pad_whole_image(mask, patch_size))

        nr_of_patches = get_nr_of_patches_for_images(img, patch_size, context_size)

        # Dumping all the patch_names into a list so we can get them at runtime. That way we don't need to precut all the patches
        for i in range(nr_of_patches):
            sample_names.append(name + '_{:03d}'.format(i))

        entry = {
            IMAGE: np.round(img * 255).astype(np.uint8),
            IMAGE_MASK: mask.astype(np.uint8),
        }
        meta_data[name] = entry

    return sample_names, meta_data


def extract_grayscale_patches(img, patch_size_tuple, offset=(0, 0), stride=(1, 1)):
    """ Extracts (typically) overlapping regular patches from a grayscale image
    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!
    Adopted from: http://jamesgregson.ca/extract-image-patches-in-python.html
    :param img: (rows/height x columns/width ndarray) input image from which to extract patches
    :param patch_size_tuple: (2-element arraylike) patch_size_tuple of that patches as (h, w)
    :param offset: (2-element arraylike) offset of the initial point as (y, x)
    :param stride: (2-element arraylike) vertical and horizontal strides
    :return: patches, origin
        patches (ndarray): output image patches as (N,patch_size_tuple[0],patch_size_tuple[1]) array
        origin (2-tuple): array of y_coord and array of x_coord coordinates
    """
    img = np.squeeze(img, axis=0) if img.ndim == 3 else img
    px, py = np.meshgrid(np.arange(patch_size_tuple[1]), np.arange(patch_size_tuple[0]))

    # Get left, top (x, y) coordinates for the patches
    x_tmp = np.arange(offset[0], img.shape[0] - patch_size_tuple[0] + 1, stride[0])
    y_tmp = np.arange(offset[1], img.shape[1] - patch_size_tuple[1] + 1, stride[1])

    # Return coordinate matrices from coordinate vectors
    # (e.g. x_tmp is [0, 1, 2] and y_tmp is [0, 1],
    # then x_coord is [[0, 1, 2], [0, 1, 2]] and y_coord is [[0, 1], [0, 1], [0, 1]]
    x_coord, y_coord = np.meshgrid(x_tmp, y_tmp)

    # Return a contiguous flattened array
    # In the example: x_coord becomes [0, 1, 2, 0, 1, 2] and y_coord [0, 1, 0, 1, 0, 1]
    x_coord = x_coord.ravel()
    y_coord = y_coord.ravel()

    # Get X Indices for each pixel per patch
    x_index_within_patch = np.tile(py[None, :, :], (x_coord.size, 1, 1))
    x_offset_in_image = np.tile(x_coord[:, None, None], (1, patch_size_tuple[0], patch_size_tuple[1]))
    x = x_index_within_patch + x_offset_in_image

    # Get Y Indices for each pixel per patch
    y_offset_in_image = np.tile(y_coord[:, None, None], (1, patch_size_tuple[0], patch_size_tuple[1]))
    y_index_within_patch = np.tile(px[None, :, :], (y_coord.size, 1, 1))
    y = y_offset_in_image + y_index_within_patch

    patches = img[x.ravel(), y.ravel()].reshape((-1, patch_size_tuple[0], patch_size_tuple[1]))
    return patches, (x_coord, y_coord)


def fetch_overlapping_patches(parent_dir, split, patch_size, overlap, context_size, automatic_resizing=False):
    if split == "train" or split == "val":
        split_idx = pickle.load(open(os.path.join(parent_dir, split + "_idx.txt"), "rb"))
        split = "train"

    else:
        split_sar_dir = os.path.join(parent_dir, SAR_IMAGES_DIR, split)
        n = len(os.listdir(split_sar_dir))
        split_idx = np.arange(0, n)

    sample_names = []
    meta_data = dict()

    split_zones_dir = os.path.join(parent_dir, ZONES_DIR, split)
    split_sar_dir = os.path.join(parent_dir, SAR_IMAGES_DIR, split)

    all_files = os.listdir(split_sar_dir)
    all_files.sort()

    for file_idx in tqdm(split_idx):
        file_name = all_files[file_idx]
        name = file_name[:-4]
        img = Image.open(os.path.join(split_sar_dir, file_name)).convert('L')
        mask = Image.open(os.path.join(split_zones_dir, name + "_zones" + ".png")).convert('L')
        if automatic_resizing:
            resolution_factor = int(name.split('_')[-3])
            rescaling_factor = resolution_factor / STATIC_ZOOM_FACTOR
            rescaled_height = int(rescaling_factor * img.height)
            rescaled_width = int(rescaling_factor * img.width)

            img = img.resize((rescaled_width, rescaled_height), resample=PIL.Image.BICUBIC)
            mask = mask.resize((rescaled_width, rescaled_height), resample=PIL.Image.NEAREST)

        img = preprocess(pad_whole_image_overlapping(img, context_size, overlap, patch_size))
        mask = mask_preprocess(pad_whole_image_overlapping(mask, context_size, overlap, patch_size))
        assert overlap >= patch_size, "Overlap must be greater than or equal to patch size"
        patches, coords_image = extract_grayscale_patches(img, (context_size, context_size),
                                                          stride=(context_size - overlap, context_size - overlap))
        nr_of_patches = len(patches)

        # Dumping all the patch_names into a list so we can get them at runtime.
        # That way we don't need to precut all the patches
        for j in range(nr_of_patches):
            add_to_name = '__x_x_' + str(j) + '_' + str(coords_image[0][j]) + '_' + str(coords_image[1][j])
            sample_names.append(name + add_to_name)

        entry = {
            IMAGE: np.round(img * 255).astype(np.uint8),
            IMAGE_MASK: mask.astype(np.uint8),
        }
        meta_data[name] = entry

    return sample_names, meta_data


def fetch_whole_set(parent_dir, split, patch_size=224, padding_mode="symmetric", automatic_resizing=False):
    if split == "val" or split == "train":
        split_idx = pickle.load(open(os.path.join(parent_dir, split + "_idx.txt"), "rb"))

        if split == "val":
            split = "train"
    else:
        split_sar_dir = os.path.join(parent_dir, SAR_IMAGES_DIR, split)
        n = len(os.listdir(split_sar_dir))
        split_idx = np.arange(0, n)

    sample_names = []
    meta_data = dict()

    split_zones_dir = os.path.join(parent_dir, ZONES_DIR, split)
    split_sar_dir = os.path.join(parent_dir, SAR_IMAGES_DIR, split)

    all_files = os.listdir(split_sar_dir)
    all_files.sort()

    for file_idx in tqdm(split_idx):
        file_name = all_files[file_idx]
        name = file_name[:-4]

        img = Image.open(os.path.join(split_sar_dir, file_name)).convert('L')
        mask = Image.open(os.path.join(split_zones_dir, name + "_zones" + ".png")).convert('L')
        if automatic_resizing:
            resolution_factor = int(name.split('_')[-3])
            rescaling_factor = resolution_factor / STATIC_ZOOM_FACTOR
            rescaled_height = int(rescaling_factor * img.height)
            rescaled_width = int(rescaling_factor * img.width)

            img = img.resize((rescaled_width, rescaled_height), resample=PIL.Image.BICUBIC)
            mask = mask.resize((rescaled_width, rescaled_height), resample=PIL.Image.NEAREST)

        img = preprocess(pad_whole_image(img, patch_size))
        mask = mask_preprocess(pad_whole_image(mask, patch_size))

        entry = {
            IMAGE: np.round(img * 255).astype(np.uint8),
            IMAGE_MASK: mask.astype(np.uint8),
        }
        meta_data[name] = entry
        sample_names.append(name)

    return sample_names, meta_data


def preprocess(img_nd):
    if isinstance(img_nd, PIL.Image.Image):
        img_nd = np.array(img_nd)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    img_trans = np.einsum('hwc -> chw', img_nd)
    # img_trans = np.expand_dims(img_nd, axis=0)
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans


def mask_preprocess(pil_img):
    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = np.einsum('hwc -> chw', img_nd)
    img_trans = img_trans[0, :, :]

    C, D = img_trans.shape
    mask = np.ones([C, D]) * 15
    stone = np.where(img_trans == 0)
    na_area = np.where(img_trans == 63)
    na_areas = np.where(img_trans == 64)
    glacier = np.where(img_trans == 127)
    ocean_ice = np.where(img_trans == 254)
    mask[stone] = STONE_ID
    mask[na_area] = NA_AREA_ID
    mask[na_areas] = NA_AREA_ID
    mask[glacier] = GLACIER_ID
    mask[ocean_ice] = OCEAN_ICE_ID
    return mask
