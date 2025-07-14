import os.path
from CaFFe.fetcher import *
import rasterio


def fetch_whole_set_pretraining(dir, patch_size=224):
    sample_names = []
    meta_data = dict()

    all_files = os.listdir(dir)
    all_files.sort()

    for file_idx in tqdm(range(len(all_files))):
        file_name = all_files[file_idx]
        name = file_name[:-4]

        img = preprocess(pad_whole_image(Image.open(os.path.join(dir, file_name)).convert('L'), patch_size))

        entry = {
            IMAGE: np.round(img * 255).astype(np.uint8),
        }
        meta_data[name] = entry
        sample_names.append(name)

    return sample_names, meta_data


def fetch_whole_set_pretraining_optical(dir, patch_size=224):
    sample_names = []
    meta_data = dict()

    all_files = os.listdir(dir)
    all_files.sort()

    for file_idx in tqdm(range(len(all_files))):
        file_name = all_files[file_idx]
        name = file_name[:-4]
        orig_img_bands = []
        orig_img_band_1 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(1))
        orig_img_bands.append(orig_img_band_1)
        orig_img_band_2 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(2))
        orig_img_bands.append(orig_img_band_2)
        orig_img_band_3 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(3))
        orig_img_bands.append(orig_img_band_3)
        orig_img_band_4 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(4))
        orig_img_bands.append(orig_img_band_4)
        orig_img_band_5 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(5))
        orig_img_bands.append(orig_img_band_5)
        orig_img_band_6 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(6))
        orig_img_bands.append(orig_img_band_6)
        orig_img_band_7 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(7))
        orig_img_bands.append(orig_img_band_7)
        orig_img_band_8 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(8))
        orig_img_bands.append(orig_img_band_8)
        orig_img_band_8a = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(9))
        orig_img_bands.append(orig_img_band_8a)
        orig_img_band_9 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(10))
        orig_img_bands.append(orig_img_band_9)
        orig_img_band_11 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(11))
        orig_img_bands.append(orig_img_band_11)
        orig_img_band_12 = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(12))
        orig_img_bands.append(orig_img_band_12)
        orig_img_band_wvp = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(14))
        orig_img_bands.append(orig_img_band_wvp)
        orig_img_band_scl = np.nan_to_num(rasterio.open(os.path.join(dir, file_name)).read(15))
        orig_img_bands.append(orig_img_band_scl)
        # orig_img_band_tci_r = rasterio.open(os.path.join(dir, file_name)).read(16)
        # orig_img_band_tci_g = rasterio.open(os.path.join(dir, file_name)).read(17)
        # orig_img_band_tci_b = rasterio.open(os.path.join(dir, file_name)).read(18)
        # orig_img_band_msk_cldprb = rasterio.open(os.path.join(dir, file_name)).read(19)
        # orig_img_band_msk_snwprb = rasterio.open(os.path.join(dir, file_name)).read(20)

        band_imgs = []
        for band in orig_img_bands:
            band_img = band / np.max(band)
            band_img = Image.fromarray(band_img)
            band_img = pad_whole_image(band_img, patch_size)
            band_img = preprocess(band_img)
            band_imgs.append(np.squeeze(band_img, 0))

        img = np.stack(band_imgs, axis=0)
        entry = {
            IMAGE: np.round(img * 255).astype(np.uint8),
        }
        meta_data[name] = entry
        sample_names.append(name)

    return sample_names, meta_data


def fetch_patches_pretraining(parent_dir, patch_size, context_size):
    n = len(os.listdir(parent_dir))
    split_idx = np.arange(0, n)
    sample_names = []
    meta_data = dict()
    all_files = os.listdir(parent_dir)
    all_files.sort()

    for file_idx in tqdm(split_idx):
        file_name = all_files[file_idx]
        name = file_name[:-4]
        img = preprocess(pad_whole_image(Image.open(os.path.join(parent_dir, file_name)).convert('L'), patch_size))
        nr_of_patches = get_nr_of_patches_for_images(img, patch_size, context_size)
        # Dumping all the patch_names into a list so we can get them at runtime. That way we don't need to precut all the patches
        for i in range(nr_of_patches):
            sample_names.append(name + '_{:03d}'.format(i))
        entry = {
            IMAGE: np.round(img * 255).astype(np.uint8)
        }
        meta_data[name] = entry
    return sample_names, meta_data
