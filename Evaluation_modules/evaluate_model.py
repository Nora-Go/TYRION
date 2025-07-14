from model.utils import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms import *
import torch
from torchmetrics import JaccardIndex
from CaFFe.constants import *
import PIL
import pickle
from scipy.ndimage.filters import gaussian_filter
import cv2
from PIL import Image


def is_subarray(subarray, arr):
    """
    Test whether subarray is a subset of arr
    :param subarray: list of numbers
    :param arr: list of numbers
    :return: boolean
    """
    count = 0
    for element in subarray:
        if element in arr:
            count += 1
    if count == len(subarray):
        return True
    return False


def get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    """
    Returns Gaussian map with size of patch and sig
    :param patch_size: The size of the image patches -> gaussian importance map will have the same size
    :param sigma_scale: A scaling factor
    :return: Gaussian importance map
    """
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def reconstruct_from_grayscale_patches_with_origin(patches, patch_size, context_size, original_image_size, overlap,
                                                   origin, use_gaussian, epsilon=1e-12):
    """Rebuild an image from a set of patches by averaging. The reconstructed image will have different dimensions than
    the original image if the strides and offsets of the patches were changed from the defaults!
    Adopted from: http://jamesgregson.ca/extract-image-patches-in-python.html
    :param patches: (ndarray) input patches as (N,patch_height,patch_width) array
    :param origin: (2-tuple) = row index and column index coordinates of each patch
    :param use_gaussian: Boolean to turn on Gaussian Importance Weighting
    :param epsilon: (scalar) regularization term for averaging when patches some image pixels are not covered by any patch
    :return image, weight
        image (ndarray): output image reconstructed from patches of size (max(origin[0])+patches.shape[1], max(origin[1])+patches.shape[2])
        weight (ndarray): output weight matrix consisting of the count of patches covering each pixel
    """
    patches = np.array(patches)
    origin = np.array(origin)
    bottom = (context_size - overlap) - ((original_image_size[0] - context_size) % (context_size - overlap))
    right = (context_size - overlap) - ((original_image_size[1] - context_size) % (context_size - overlap))
    right = right + (context_size - patch_size)
    bottom = bottom + (context_size - patch_size)
    left_offset = int((context_size - patch_size) // 2)
    top_offset = int((context_size - patch_size) // 2)

    img_height = original_image_size[1] + bottom
    img_width = original_image_size[0] + right

    out = np.zeros((img_width, img_height))
    wgt = np.zeros((img_width, img_height))
    if use_gaussian:
        scale_wgt = get_gaussian((patch_size, patch_size))
    else:
        scale_wgt = np.ones((patch_size, patch_size))
    for i in range(patch_size):
        for j in range(patch_size):
            out[origin[0] + i + left_offset, origin[1] + j + top_offset] += patches[:, i, j] * scale_wgt[i, j]
            wgt[origin[0] + i + left_offset, origin[1] + j + top_offset] += scale_wgt[i, j]

    return out / np.maximum(wgt, epsilon), wgt


def reconstruct_from_patches_return_confi(all_patches, all_patches_names_per_image, ground_truth_dir, patch_size,
                                          context_size, overlap):
    """
    Reconstruct the image from patches in src_directory and store them in dst_directory.
    The src_directory contains masks (patches = number_of_classes x height x width).
    The class with maximum probability will be chosen as prediction after averaging the probabilities across patches
    (if there is an overlap) and the image in dst_directory will only show the prediction (image = height x width)
    :param dst_directory: destination directory
    :return: prediction (image = height x width)
    """
    class_probabilities_complete_images = {}
    for name in list(all_patches_names_per_image.keys()):
        print(f"File: {name}")
        # #####################################################################################################
        # Search all patches that belong to the image with the name "name"
        # #####################################################################################################
        patches_for_image_names = all_patches_names_per_image.get(name)
        assert len(patches_for_image_names) > 0, "No patches found for image " + name
        patches_for_image = []  # Will be Number_Of_Patches x Number_Of_Classes x Height x Width
        irow = []
        icol = []

        for file_name in patches_for_image_names:
            # #####################################################################################################
            # Get the origin of the patches out of their names
            # #####################################################################################################
            # naming convention: nameOfTheOriginalImage__PaddedBottom_PaddedRight_NumberOfPatch_irow_icol.png

            # Mask patches are 3D arrays with class probabilities
            patches_for_image.append(all_patches.get(file_name))
            irow.append(int(os.path.split(file_name)[1].split("_")[-2]))
            icol.append(int(os.path.split(file_name)[1].split("_")[-1]))

        # Images are masks and store the probabilities for each class (patch = number_class x height x width)
        class_patches_for_image = []
        patches_for_image = [np.array(x) for x in patches_for_image]
        patches_for_image = np.array(patches_for_image)
        for class_layer in range(len(patches_for_image[0])):
            class_patches_for_image.append(patches_for_image[:, class_layer, :, :])

        class_probabilities_complete_image = []

        # #####################################################################################################
        # Reconstruct image (with number of channels = classes) from patches
        # #####################################################################################################
        ground_truth_img = cv2.imread(os.path.join(ground_truth_dir, name + "_zones.png"), cv2.IMREAD_GRAYSCALE)
        original_W = ground_truth_img.shape[0]
        original_H = ground_truth_img.shape[1]
        for class_number in range(len(class_patches_for_image)):
            class_probability_complete_image, _ = reconstruct_from_grayscale_patches_with_origin(
                class_patches_for_image[class_number],
                patch_size,
                context_size,
                (original_W, original_H),
                overlap,
                origin=(irow, icol),
                use_gaussian=True)
            # Cut Padding
            class_probability_complete_image = class_probability_complete_image[
                                               :original_W + (context_size - patch_size),
                                               :original_H + (context_size - patch_size)]
            class_probability_complete_image = CenterCrop((original_W, original_H))(
                Image.fromarray(class_probability_complete_image))
            class_probabilities_complete_image.append(np.array(class_probability_complete_image))

        class_probabilities_complete_image = np.array(class_probabilities_complete_image)
        class_probabilities_complete_images[name] = class_probabilities_complete_image
    return class_probabilities_complete_images


def reconstruct_from_patches(all_patches, all_patches_names_per_image, prob_dir, ground_truth_dir, patch_size,
                             context_size, overlap):
    """
    Reconstruct the image from patches in src_directory and store them in dst_directory.
    The src_directory contains masks (patches = number_of_classes x height x width).
    The class with maximum probability will be chosen as prediction after averaging the probabilities across patches
    (if there is an overlap) and the image in dst_directory will only show the prediction (image = height x width)
    :param dst_directory: destination directory
    :return: prediction (image = height x width)
    """
    for name in list(all_patches_names_per_image.keys()):
        print(f"File: {name}")
        # #####################################################################################################
        # Search all patches that belong to the image with the name "name"
        # #####################################################################################################
        patches_for_image_names = all_patches_names_per_image.get(name)
        assert len(patches_for_image_names) > 0, "No patches found for image " + name
        patches_for_image = []  # Will be Number_Of_Patches x Number_Of_Classes x Height x Width
        irow = []
        icol = []

        for file_name in patches_for_image_names:
            # #####################################################################################################
            # Get the origin of the patches out of their names
            # #####################################################################################################
            # naming convention: nameOfTheOriginalImage__PaddedBottom_PaddedRight_NumberOfPatch_irow_icol.png

            # Mask patches are 3D arrays with class probabilities
            patches_for_image.append(all_patches.get(file_name))
            irow.append(int(os.path.split(file_name)[1].split("_")[-2]))
            icol.append(int(os.path.split(file_name)[1].split("_")[-1]))

        # Images are masks and store the probabilities for each class (patch = number_class x height x width)
        class_patches_for_image = []
        patches_for_image = [np.array(x) for x in patches_for_image]
        patches_for_image = np.array(patches_for_image)
        for class_layer in range(len(patches_for_image[0])):
            class_patches_for_image.append(patches_for_image[:, class_layer, :, :])

        class_probabilities_complete_image = []

        # #####################################################################################################
        # Reconstruct image (with number of channels = classes) from patches
        # #####################################################################################################
        ground_truth_img = cv2.imread(os.path.join(ground_truth_dir, name + "_zones.png"), cv2.IMREAD_GRAYSCALE)
        original_W = ground_truth_img.shape[0]
        original_H = ground_truth_img.shape[1]
        for class_number in range(len(class_patches_for_image)):
            class_probability_complete_image, _ = reconstruct_from_grayscale_patches_with_origin(
                class_patches_for_image[class_number],
                patch_size,
                context_size,
                (original_W, original_H),
                overlap,
                origin=(irow, icol),
                use_gaussian=True)
            # Cut Padding
            class_probability_complete_image = class_probability_complete_image[
                                               :original_W + (context_size - patch_size),
                                               :original_H + (context_size - patch_size)]
            class_probability_complete_image = CenterCrop((original_W, original_H))(
                Image.fromarray(class_probability_complete_image))
            class_probabilities_complete_image.append(np.array(class_probability_complete_image))

        class_probabilities_complete_image = np.array(class_probabilities_complete_image)
        pickle.dump(class_probabilities_complete_image.astype(np.float16),
                    open(os.path.join(prob_dir, name + ".pkl"), "wb"))
    return


def evaluate_model_overlapping_patches_return_dict(model, dataloader, complete_directory, ground_truth_dir):
    whole_dict = dict()
    all_patches = dict()
    all_patches_names_per_image = dict()
    for index, batch in enumerate(dataloader):
        cuda_batch = {key: value.to("cuda") for key, value in batch.items() if key != "name"}
        cuda_batch["name"] = batch["name"]
        patch = model.get_input(cuda_batch)
        out, _ = model(patch)
        pred = model.get_mid_window(out)
        pred = torch.nn.functional.softmax(pred, dim=1)
        name = batch['name']
        for i in range(len(name)):
            suffix = name[i].split('__')[0]
            if all_patches_names_per_image.get(suffix) is None:
                all_patches_names_per_image[suffix] = []
            all_patches_names_per_image[suffix].append(name[i])
            all_patches[name[i]] = pred[i].cpu().detach().numpy()

    patch_size = dataloader.dataset.patch_size
    context_size = dataloader.dataset.context_factor * patch_size
    class_probabilities_complete_images = reconstruct_from_patches_return_confi(all_patches,
                                                                                all_patches_names_per_image,
                                                                                ground_truth_dir, patch_size,
                                                                                context_size,
                                                                                dataloader.dataset.overlap)

    # Choose class with highest probability as prediction
    for name in list(all_patches_names_per_image.keys()):
        class_probabilities_complete_image = class_probabilities_complete_images.get(name)
        prediction = np.argmax(class_probabilities_complete_image, axis=0)
        whole_dict[name] = prediction.astype(np.uint8)

        # Convert [0, 1] to [0, 255] range
        prediction[prediction == 0] = 0
        prediction[prediction == 1] = 64
        prediction[prediction == 2] = 127
        prediction[prediction == 3] = 254
        assert (is_subarray(np.unique(prediction), [0, 64, 127, 254])), "Unique zone values are not correct"
        cv2.imwrite(os.path.join(complete_directory, name + ".png"), prediction)
    return whole_dict


def evaluate_model_overlapping_patches(model, dataloader, prob_dir, ground_truth_dir):
    all_patches = dict()
    all_patches_names_per_image = dict()
    for index, batch in enumerate(dataloader):
        cuda_batch = {key: value.to("cuda") for key, value in batch.items() if key != "name"}
        cuda_batch["name"] = batch["name"]
        patch = model.get_input(cuda_batch)
        out, _ = model(patch)
        pred = model.get_mid_window(out)
        pred = torch.nn.functional.softmax(pred, dim=1)
        name = batch['name']
        for i in range(len(name)):
            suffix = name[i].split('__')[0]
            if all_patches_names_per_image.get(suffix) is None:
                all_patches_names_per_image[suffix] = []
            all_patches_names_per_image[suffix].append(name[i])
            all_patches[name[i]] = pred[i].cpu().detach().numpy()

    patch_size = dataloader.dataset.patch_size
    context_size = dataloader.dataset.context_factor * patch_size
    reconstruct_from_patches(all_patches, all_patches_names_per_image, prob_dir, ground_truth_dir, patch_size,
                             context_size, dataloader.dataset.overlap)
    del all_patches
    return


def evaluate_model(model, DL, ground_truth, whole_save):
    whole_dict = dict()
    logger = TensorBoardLogger(save_dir="../TMP_RESULTS", name="TMP_NAME")
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=logger)
    trainer.test(model, dataloaders=DL)

    # Not the prettiest way to get the values out of the model but a possible way :)
    predicted_names = model.suffix_to_names
    predicted_patches = model.test_results

    IOU = JaccardIndex(task='multiclass', num_classes=4, average='none')
    iou_ratio = 0.0
    success_ratio = 0.0
    patch_size = model.patch_size

    for large_gt_name in os.listdir(ground_truth):
        gt = Image.open(os.path.join(ground_truth, large_gt_name)).convert('L')
        original_W, original_H = gt.size
        gt = whole_preprocess(gt)
        suffix = large_gt_name.split('.')[0][0:-6]

        if model.automatic_resizing:
            resolution_factor = int(suffix.split('_')[-3])
            rescaling_factor = resolution_factor / STATIC_ZOOM_FACTOR
        else:
            rescaling_factor = 1

        W = int(original_W * rescaling_factor)
        H = int(original_H * rescaling_factor)

        HH = H // patch_size + 1
        WW = W // patch_size + 1
        length = patch_size
        all_names = predicted_names[suffix]
        all_names.sort()
        all_patches = []

        for i in range(len(all_names)):
            all_patches.append(predicted_patches[all_names[i]])

        whole = Image.new('L', (WW * length, HH * length))

        for k in range(len(all_patches)):
            whole.paste(back(all_patches[k]),
                        (length * (k % WW), length * (k // WW), length * (k % WW + 1), length * (k // WW + 1)))

        whole = CenterCrop((H, W))(whole)
        if model.automatic_resizing:
            whole = whole.resize((original_W, original_H), resample=PIL.Image.NEAREST)

        whole.save(os.path.join(whole_save, suffix + '.png'))

        whole_dict[suffix] = whole_preprocess(whole).astype(np.uint8)
        h, w = whole.size

        whole = whole_preprocess(whole)
        success_ratio += np.sum(np.where(whole == gt, 1, 0)) / (h * W)

        iou = IOU(torch.from_numpy(whole).type(torch.int64), torch.from_numpy(gt).type(torch.int64))
        iou_ratio += iou

    iou_whole = iou_ratio / len(os.listdir(ground_truth))

    print("################# My output#############")
    print("STONE_ID = 0, NA_AREA_ID = 1, GLACIER_ID = 2, OCEAN_ICE_ID = 3")
    print("IoU whole: ", iou_whole)
    print("IoU avg", torch.mean(iou_whole))
    print("##########################################")
    iou_ratio = iou_ratio / len(os.listdir(ground_truth))
    ave_iou = sum(iou_ratio) / len(iou_ratio)
    success_ratio = success_ratio / len(os.listdir(ground_truth))

    return success_ratio, ave_iou, iou_ratio, whole_dict
