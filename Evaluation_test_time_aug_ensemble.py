from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torchmetrics
from argparse import ArgumentParser
from CaFFe.data import postprocess_zone_segmenation, extract_front_from_zones
import re
from scipy.spatial import distance
import skimage
from tqdm import tqdm
from Evaluation_modules.evaluate_model import evaluate_model_overlapping_patches
from model.utils import *
from CaFFe.constants import *
import pickle
import shutil
import cv2
from PIL import Image
import csv
from Evaluation_modules import csv_files_utils
os.environ["GIT_PYTHON_REFRESH"] = "quiet"


def front_error(prediction, label):
    """
    Calculate the distance errors between prediction and label
    :param prediction: mask of the front prediction (WxH)
    :param label: mask of the front label (WxH)
    :return: boolean whether a front pixel is present in the prediction and the distance errors as np.array
    """
    front_is_present_flag = True
    polyline_pred = np.nonzero(prediction)
    polyline_label = np.nonzero(label)

    # Generate Nx2 matrix of pixels that represent the front
    pred_coords = np.array(list(zip(polyline_pred[0], polyline_pred[1])))
    mask_coords = np.array(list(zip(polyline_label[0], polyline_label[1])))

    # Return NaN if front is not detected in either pred or mask
    if pred_coords.shape[0] == 0 or mask_coords.shape[0] == 0:
        front_is_present_flag = False
        return front_is_present_flag, np.nan

    # Generate the pairwise distances between each point and the closest point in the other array
    distances1 = distance.cdist(pred_coords, mask_coords).min(axis=1)
    distances2 = distance.cdist(mask_coords, pred_coords).min(axis=1)
    distances = np.concatenate((distances1, distances2))

    return front_is_present_flag, distances


def torch_multi_class_metric(metric_function, complete_predicted_mask, complete_target):
    metrics = []
    metric_na, metric_stone, metric_glacier, metric_ocean = metric_function(complete_predicted_mask,
                                                                            complete_target).to('cpu')
    metric_macro_average = (metric_na + metric_stone + metric_glacier + metric_ocean) / 4
    metrics.append(metric_macro_average.item())
    metrics.append(metric_na.item())
    metrics.append(metric_stone.item())
    metrics.append(metric_glacier.item())
    metrics.append(metric_ocean.item())
    return metrics


def turn_colors_to_class_labels_zones(mask):
    mask_class_labels = np.copy(mask)
    mask_class_labels[mask == 0] = 0
    mask_class_labels[mask == 64] = 1
    mask_class_labels[mask == 127] = 2
    mask_class_labels[mask == 254] = 3
    return mask_class_labels


def turn_class_labels_zones_to_colors(mask):
    mask_colors = np.copy(mask)
    mask_colors[mask == 0] = 0
    mask_colors[mask == 1] = 64
    mask_colors[mask == 2] = 127
    mask_colors[mask == 3] = 254
    return mask_colors


def print_zone_metrics(metric_name, list_of_metrics, log_list):
    metrics = [metric for [metric, _, _, _, _] in list_of_metrics if not np.isnan(metric)]
    metrics_na = [metric_na for [_, metric_na, _, _, _] in list_of_metrics if not np.isnan(metric_na)]
    metrics_stone = [metric_stone for [_, _, metric_stone, _, _] in list_of_metrics if not np.isnan(metric_stone)]
    metrics_glacier = [metric_glacier for [_, _, _, metric_glacier, _] in list_of_metrics if
                       not np.isnan(metric_glacier)]
    metrics_ocean = [metric_ocean for [_, _, _, _, metric_ocean] in list_of_metrics if not np.isnan(metric_ocean)]
    log_list.append(sum(metrics) / len(metrics))
    log_list.append(sum(metrics_na) / len(metrics_na))
    log_list.append(sum(metrics_stone) / len(metrics_stone))
    log_list.append(sum(metrics_glacier) / len(metrics_glacier))
    log_list.append(sum(metrics_ocean) / len(metrics_ocean))
    print(f"Average {metric_name}: {sum(metrics) / len(metrics)}")
    print(f"Average {metric_name} NA Area: {sum(metrics_na) / len(metrics_na)}")
    print(f"Average {metric_name} Stone: {sum(metrics_stone) / len(metrics_stone)}")
    print(f"Average {metric_name} Glacier: {sum(metrics_glacier) / len(metrics_glacier)}")
    print(f"Average {metric_name} Ocean and Ice Melange: {sum(metrics_ocean) / len(metrics_ocean)}")


def get_matching_out_of_folder(file_name, folder):
    files = os.listdir(folder)
    matching_files = [a for a in files if
                      re.match(pattern=os.path.split(file_name)[1][:-4], string=os.path.split(a)[1])]
    if len(matching_files) > 1:
        print("Something went wrong!")
        print(f"targets_matching: {matching_files}")
    if len(matching_files) < 1:
        print("Something went wrong! No matches found")
    return matching_files[0]


def evaluate_model_on_given_dataset(mode, prob_dir, complete_test_directory, uncertainty_dir, data_parent_dir):
    for model_nr in range(5):
        model_config = "Ensemble_" + str(model_nr) + ".yaml"
        model = instantiate_completely('model', model_config)
        model.eval()
        model.cuda()
        for rot in range(4):
            if mode == "test":
                ground_truth = os.path.join(data_parent_dir, 'data_raw_rot_' + str(rot), 'zones', "test")
                data_config = "test_nora_rot_" + str(rot) + ".yaml"
            else:
                ground_truth = os.path.join(data_parent_dir, 'data_raw_val_as_test_set_rot_' + str(rot), 'zones', "test")
                data_config = "val_nora_rot_" + str(rot) + ".yaml"

            loader = instantiate_completely('Data', data_config, stage="test").test_dataloader()
            model_rot_prob_dir = os.path.join(prob_dir, "model_" + str(model_nr) + "_rot_" + str(rot))
            os.makedirs(model_rot_prob_dir, exist_ok=True)
            evaluate_model_overlapping_patches(model, loader, model_rot_prob_dir, ground_truth)

    # take the average of the 5 models and 4 rotations
    whole_dict_preds = dict()
    whole_dict_probs = dict()
    ensemble_prob_list = dict()
    for model_nr in range(5):
        for rot in range(4):
            for file_name in os.listdir(os.path.join(prob_dir, "model_" + str(model_nr) + "_rot_" + str(rot))):
                mask = pickle.load(open(os.path.join(prob_dir, "model_" + str(model_nr) + "_rot_" + str(rot), file_name), "rb"))
                mask = np.rot90(mask, k=rot, axes=(1, 2))
                # back(einops.rearrange(mask[0:3, :, :]*255, "c h w -> h w c").astype(np.uint8)).show()
                if ensemble_prob_list.get(file_name[:-4]) is None:
                    ensemble_prob_list[file_name[:-4]] = []
                ensemble_prob_list[file_name[:-4]].append(mask)
    for file_name in os.listdir(os.path.join(prob_dir, "model_" + str(model_nr) + "_rot_" + str(rot))):
        ensemble_pred = np.average(np.array(ensemble_prob_list[file_name[:-4]]), axis=0).astype(np.float32)
        ensemble_pred = np.argmax(ensemble_pred, axis=0)
        ensemble_prob = np.std(np.array(ensemble_prob_list[file_name[:-4]]), axis=0).astype(np.float32)
        whole_dict_preds[file_name[:-4]] = ensemble_pred
        whole_dict_probs[file_name[:-4]] = ensemble_prob

        save_image = np.zeros(ensemble_pred.shape)
        save_image[ensemble_pred == 0] = 0
        save_image[ensemble_pred == 1] = 64
        save_image[ensemble_pred == 2] = 127
        save_image[ensemble_pred == 3] = 254
        cv2.imwrite(os.path.join(complete_test_directory, file_name[:-4] + ".png"), save_image)
        for class_nr in range(4):
            class_name = "NA" if class_nr == 0 else "Rock" if class_nr == 1 else "Glacier" if class_nr == 2 else "Ocean"
            cv2.imwrite(os.path.join(uncertainty_dir, file_name[:-4] + "_" + class_name + ".png"), ensemble_prob[class_nr] * 255)
    del ensemble_prob_list
    shutil.rmtree(os.path.join(prob_dir))
    return whole_dict_preds, whole_dict_probs


def evaluate_model_on_given_dataset_no_testtimeaug(mode, prob_dir, complete_test_directory, uncertainty_dir, data_parent_dir):
    for model_nr in range(5):
        model_config = "Ensemble_" + str(model_nr) + ".yaml"
        model = instantiate_completely('model', model_config)
        model.eval()
        model.cuda()
        if mode == "test":
            ground_truth = os.path.join(data_parent_dir, 'data_raw', 'zones', "test")
            data_config = "test.yaml"
        else:
            ground_truth = os.path.join(data_parent_dir, 'data_raw_val_as_test_set', 'zones', "test")
            data_config = "val.yaml"

        loader = instantiate_completely('Data', data_config, stage="test").test_dataloader()
        model_prob_dir = os.path.join(prob_dir, "model_" + str(model_nr))
        os.makedirs(model_prob_dir, exist_ok=True)
        evaluate_model_overlapping_patches(model, loader, model_prob_dir, ground_truth)

    # take the average of the 5 models and 4 rotations
    whole_dict_preds = dict()
    whole_dict_probs = dict()
    ensemble_prob_list = dict()
    for model_nr in range(5):
        for file_name in os.listdir(os.path.join(prob_dir, "model_" + str(model_nr))):
            mask = pickle.load(open(os.path.join(prob_dir, "model_" + str(model_nr), file_name), "rb"))
            # back(einops.rearrange(mask[0:3, :, :]*255, "c h w -> h w c").astype(np.uint8)).show()
            if ensemble_prob_list.get(file_name[:-4]) is None:
                ensemble_prob_list[file_name[:-4]] = []
            ensemble_prob_list[file_name[:-4]].append(mask)
    for file_name in os.listdir(os.path.join(prob_dir, "model_" + str(model_nr))):
        ensemble_pred = np.average(np.array(ensemble_prob_list[file_name[:-4]]), axis=0).astype(np.float32)
        ensemble_pred = np.argmax(ensemble_pred, axis=0)
        ensemble_prob = np.std(np.array(ensemble_prob_list[file_name[:-4]]), axis=0).astype(np.float32)
        whole_dict_preds[file_name[:-4]] = ensemble_pred
        whole_dict_probs[file_name[:-4]] = ensemble_prob

        save_image = np.zeros(ensemble_pred.shape)
        save_image[ensemble_pred == 0] = 0
        save_image[ensemble_pred == 1] = 64
        save_image[ensemble_pred == 2] = 127
        save_image[ensemble_pred == 3] = 254
        cv2.imwrite(os.path.join(complete_test_directory, file_name[:-4] + ".png"), save_image)
        for class_nr in range(4):
            class_name = "NA" if class_nr == 0 else "Rock" if class_nr == 1 else "Glacier" if class_nr == 2 else "Ocean"
            cv2.imwrite(os.path.join(uncertainty_dir, file_name[:-4] + "_" + class_name + ".png"), ensemble_prob[class_nr] * 255)
    del ensemble_prob_list
    shutil.rmtree(os.path.join(prob_dir))
    return whole_dict_preds, whole_dict_probs


def calculate_segmentation_metrics(target_mask_modality, names, complete_predicted_dict, whole_zones_gt_dict, log_list):
    """
    Calculate IoU, Precision, Recall and F1-Score for all predicted segmentations
    :param target_mask_modality: either "zones" or "fronts"
    :param complete_predicted_masks: All file names of complete predictions
    :param complete_dir: the directory which is going to hold the complete predicted images
    :param directory_of_complete_targets: directory holding the complete ground truth segmentations
    :return:
    """
    print("Calculate segmentation metrics ...\n\n")
    print(f"Target {target_mask_modality}:")
    list_of_ious = []
    list_of_precisions = []
    list_of_recalls = []
    list_of_f1_scores = []
    Jacard = torchmetrics.JaccardIndex(task='multiclass', num_classes=4, average='none').to('cuda')
    Precision = torchmetrics.Precision(task='multiclass', num_classes=4, average='none').to('cuda')
    Recall = torchmetrics.Recall(task='multiclass', num_classes=4, average='none').to('cuda')
    F1Score = torchmetrics.F1Score(task='multiclass', num_classes=4, average='none').to('cuda')

    for file_name in tqdm(names):
        # print(f"File: {file_name}")
        complete_predicted_mask_class_labels = torch.from_numpy(complete_predicted_dict[file_name]).to('cuda')
        complete_target_class_labels = torch.from_numpy(whole_zones_gt_dict[file_name]).to('cuda')

        # Segmentation evaluation metrics
        list_of_ious.append(
            torch_multi_class_metric(Jacard, complete_predicted_mask_class_labels, complete_target_class_labels))
        list_of_precisions.append(
            torch_multi_class_metric(Precision, complete_predicted_mask_class_labels, complete_target_class_labels))
        list_of_recalls.append(
            torch_multi_class_metric(Recall, complete_predicted_mask_class_labels, complete_target_class_labels))
        list_of_f1_scores.append(
            torch_multi_class_metric(F1Score, complete_predicted_mask_class_labels, complete_target_class_labels))

    print_zone_metrics("Precision", list_of_precisions, log_list)
    print_zone_metrics("Recall", list_of_recalls, log_list)
    print_zone_metrics("F1 Score", list_of_f1_scores, log_list)
    print_zone_metrics("IoU", list_of_ious, log_list)


def mask_prediction_with_bounding_box(post_complete_predicted_mask, file_name, bounding_boxes_directory):
    matching_bounding_box_file = get_matching_out_of_folder(file_name, bounding_boxes_directory)
    with open(os.path.join(bounding_boxes_directory, matching_bounding_box_file)) as f:
        coord_file_lines = f.readlines()
    left_upper_corner_x, left_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[1].split(",")]
    left_lower_corner_x, left_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[2].split(",")]
    right_lower_corner_x, right_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[3].split(",")]
    right_upper_corner_x, right_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[4].split(",")]

    # Make sure the Bounding Box coordinates are within the image
    if left_upper_corner_x < 0:
        left_upper_corner_x = 0
    if left_lower_corner_x < 0:
        left_lower_corner_x = 0
    if right_upper_corner_x > len(post_complete_predicted_mask[0]):
        right_upper_corner_x = len(post_complete_predicted_mask[0]) - 1
    if right_lower_corner_x > len(post_complete_predicted_mask[0]):
        right_lower_corner_x = len(post_complete_predicted_mask[0]) - 1
    if left_upper_corner_y > len(post_complete_predicted_mask):
        left_upper_corner_y = len(post_complete_predicted_mask) - 1
    if left_lower_corner_y < 0:
        left_lower_corner_y = 0
    if right_upper_corner_y > len(post_complete_predicted_mask):
        right_upper_corner_y = len(post_complete_predicted_mask) - 1
    if right_lower_corner_y < 0:
        right_lower_corner_y = 0

    # remember cv2 images have the shape (height, width)
    post_complete_predicted_mask[:right_lower_corner_y, :] = 0.0
    post_complete_predicted_mask[left_upper_corner_y:, :] = 0.0
    post_complete_predicted_mask[:, :left_upper_corner_x] = 0.0
    post_complete_predicted_mask[:, right_lower_corner_x:] = 0.0

    return post_complete_predicted_mask


def post_processing(names, complete_predicted_dict, bounding_boxes_directory, complete_postprocessed_directory,
                    save_images=True):
    meter_threshold = 750  # in meter

    post_processed_dict = dict()
    print("Post-processing ...\n\n")
    for file_name in names:
        # print(f"File: {file_name}")
        resolution = int(int(os.path.split(file_name)[1][:-4].split('_')[3]))
        # pixel_threshold (pixel) * resolution (m/pixel) = meter_threshold (m)
        pixel_threshold = meter_threshold / resolution
        complete_predicted_mask = complete_predicted_dict[file_name]

        post_complete_predicted_mask = postprocess_zone_segmenation(
            turn_class_labels_to_zones_torch(complete_predicted_mask))
        post_complete_predicted_mask = extract_front_from_zones(post_complete_predicted_mask, pixel_threshold)

        post_complete_predicted_mask = mask_prediction_with_bounding_box(post_complete_predicted_mask, file_name,
                                                                         bounding_boxes_directory)
        if save_images:
            Image.fromarray(post_complete_predicted_mask.astype(np.uint8)).save(
                os.path.join(complete_postprocessed_directory,
                             file_name + ".png"))
        post_processed_dict[file_name] = post_complete_predicted_mask.astype(np.uint8)

    return post_processed_dict


def front_dic_to_lists(dic, names):
    list_front_exists = []
    list_of_all_front_errors = []
    list_of_picture_wise_mean = []
    list_of_front_lengths = []

    counter_missing = 0

    for key in names:
        front_exists = dic[key]["front_exists"]
        list_front_exists.append(front_exists)
        if not front_exists:
            counter_missing += 1
            continue
        list_of_all_front_errors = np.concatenate((list_of_all_front_errors, dic[key]["list_of_all_front_errors"]))
        list_of_picture_wise_mean.append(np.mean(dic[key]["list_of_all_front_errors"]))
        list_of_front_lengths.append(dic[key]["nr_of_front_pixels"])

    return list_of_all_front_errors, list_front_exists, counter_missing, list_of_picture_wise_mean, list_of_front_lengths


def calculate_front_delineation_metric(names,
                                       post_processed_predicted_masks,
                                       target_fronts_dictionary):
    """
    Calculate distance errors
    :param post_processed_predicted_masks: All file names of post-processed complete predictions
    :param directory_of_target_fronts: directory holding the complete ground truth fronts
    :return: a list of all distance errors, if a prediction does not show a single front pixel, that image is neglected
    """
    list_of_all_front_errors = []
    number_of_images_with_no_predicted_front = 0
    front_dict = dict()

    for name in names:
        post_processed_predicted_mask_class_labels = post_processed_predicted_masks[name]
        target_front_class_labels = target_fronts_dictionary[name]
        resolution = int(os.path.split(name)[1][:-4].split('_')[3])
        # images need to be turned into a Tensor [0, ..., n_classes-1]
        front_is_present_flag, errors = front_error(post_processed_predicted_mask_class_labels,
                                                    target_front_class_labels)
        nr_of_front_pixels = np.where(np.array(target_front_class_labels) > 0, 1, 0).sum()
        if not front_is_present_flag:
            number_of_images_with_no_predicted_front += 1
            front_dict[name] = {
                "list_of_all_front_errors": -1,
                "front_exists": False
            }

        else:
            list_of_all_front_errors = np.concatenate((list_of_all_front_errors, resolution * errors))
            list_of_current_errors = resolution * errors

            errors_without_nan = [front_err for front_err in list_of_current_errors if
                                  not np.isnan(front_err)]

            front_dict[name] = {
                "list_of_all_front_errors": errors_without_nan,
                "front_exists": True,
                "nr_of_front_pixels": nr_of_front_pixels
            }

    """
    print(f"\t- Number of images with no predicted front: {number_of_images_with_no_predicted_front}")
    if number_of_images_with_no_predicted_front >= len(post_processed_predicted_masks):
        print(f"\t- Number of images with no predicted front is equal to complete set of images. No metrics can be calculated.")
        log_list.append(len(post_processed_predicted_masks))
        log_list.append("/")
        return
    log_list.append(number_of_images_with_no_predicted_front)
    list_of_all_front_errors_without_nan = [front_error for front_error in list_of_all_front_errors if
                                            not np.isnan(front_error)]
    list_of_all_front_errors_without_nan = np.array(list_of_all_front_errors_without_nan)
    mean = np.mean(list_of_all_front_errors_without_nan)
    log_list.append(mean)
    print(f"\t- Mean distance error: {mean}")
    """
    list_of_all_front_errors_without_nan = [front_error for front_error in list_of_all_front_errors if
                                            not np.isnan(front_error)]
    return list_of_all_front_errors_without_nan, front_dict


def check_whether_winter_half_year(name):
    split_name = name[:-4].split('_')
    if split_name[0] == "COL" or split_name[0] == "JAC":
        nord_halbkugel = True
    else:  # Jorum, Maple, Crane, SI, DBE
        nord_halbkugel = False
    month = int(split_name[1].split('-')[1])
    if nord_halbkugel:
        if month < 4 or month > 8:
            winter = True
        else:
            winter = False
    else:
        if month < 4 or month > 8:
            winter = False
        else:
            winter = True
    return winter


def front_delineation_print(dic, names, log_list):
    list_of_existing_front_errors, list_of_all_front_errors, nr_of_missing_fronts, mean_per_picture, list_of_front_lengths = front_dic_to_lists(
        dic, names)
    if len(list_of_all_front_errors) == 0:
        log_list.append("/")
        log_list.append("/")
        return

    print(f"- Results images")
    print(f"\t- Number of images:", {len(list_of_all_front_errors)})
    print("f Missing fronts: ", nr_of_missing_fronts)
    list_of_all_front_errors = np.array(list_of_existing_front_errors)
    mean = np.mean(list_of_all_front_errors)
    std = np.std(list_of_all_front_errors)

    mean_picture_wise = np.mean(np.array(mean_per_picture))
    std_picture_wise = np.array(mean_per_picture).std()

    mean_with_front_lengths = np.sum(
        np.array(mean_per_picture) * np.array(list_of_front_lengths) / np.sum(list_of_front_lengths))

    print(f"Mean distance error (in meters): {mean} and its std {std}")
    print(f"Mean distance error picture_normalized (in meters): {mean_picture_wise} and its std {std_picture_wise}")
    print(f"Mean distance error normalized accounting for front length (in meters): {mean_with_front_lengths}")

    if nr_of_missing_fronts >= len(names):
        print(f"\t- Number of images with no predicted front is equal to complete set of images. No metrics can be calculated.")
        log_list.append(len(names))
        log_list.append("/")
        return
    log_list.append("/")
    log_list.append(mean)


def parse_front_name(name):
    return ('_').join(name.split('_')[:-1])


def front_error_individual_print(dic, names):
    for key in names:
        front_exists = dic[key]["front_exists"]
        if not front_exists:
            print("name: ", key, " and mean: ", -1)
            continue

        print("name: ", key, " and mean: ", np.mean(dic[key]["list_of_all_front_errors"]))


def load_fronts(front_directory):
    front_dictionary = dict()
    for file in os.listdir(front_directory):
        img = Image.open(os.path.join(front_directory, file)).convert('L')
        name = parse_front_name(file)
        front_dictionary[name] = img

    return front_dictionary


def load_images(img_directory):
    img_dictionary = dict()
    for file in os.listdir(img_directory):
        img = Image.open(os.path.join(img_directory, file)).convert('L')
        name = file[:-4]
        img_dictionary[name] = img

    return img_dictionary


def load_sar_images(sar_directory):
    sar_dict = dict()
    for file in os.listdir(sar_directory):
        name = file.split('.')[0]
        img = Image.open(os.path.join(sar_directory, file)).convert('L')
        sar_dict[name] = img
    return sar_dict


def load_whole_zones(whole_zone_directory):
    zone_dict = dict()
    for file in os.listdir(whole_zone_directory):
        name = ('_').join(file.split('_')[:-1])
        img = whole_preprocess(Image.open(os.path.join(whole_zone_directory, file)).convert('L'))
        zone_dict[name] = img.astype(np.uint8)
    return zone_dict


def calculate_front_error_for_validation(names, post_processed_dictionary, dictionary_of_target_fronts):
    """
        names: list of names in post_processed_dictionary
        post_processed_dictionary: a dictionary with all the post processed segmentation masks
        dictionary_of_target_fronts: a dictionary with all the target fronts
    """
    _, front_error_dict = calculate_front_delineation_metric(names, post_processed_dictionary,
                                                             dictionary_of_target_fronts)

    return front_delineation_for_validation(front_error_dict, names)


def front_delineation_for_validation(dic, names):
    list_of_existing_front_errors, list_of_all_front_errors, nr_of_missing_fronts, mean_per_picture, list_of_front_lengths = front_dic_to_lists(
        dic, names)
    if len(list_of_all_front_errors) == 0:
        return 7777, 7777, 7777, nr_of_missing_fronts

    list_of_all_front_errors = torch.from_numpy(np.array(list_of_existing_front_errors)).to('cuda')
    mean = torch.mean(list_of_all_front_errors)

    mean_per_picture = torch.from_numpy(np.array(mean_per_picture)).to('cuda')
    list_of_front_lengths = torch.from_numpy(np.array(list_of_front_lengths)).to('cuda')

    mean_picture_wise = torch.mean(mean_per_picture)
    mean_with_front_lengths = torch.sum(mean_per_picture * list_of_front_lengths / torch.sum(list_of_front_lengths))

    return mean, mean_picture_wise, mean_with_front_lengths, nr_of_missing_fronts


def front_delineation_metric(names, post_processed_dictionary, dictionary_of_target_fronts, log_list):
    """
    Calculate the mean distance error (MDE) and break it up by glacier, season, resolution and satellite
    :param complete_postprocessed_directory: the directory holding the complete predicted and post-processed images
    :param directory_of_target_fronts: directory holding the complete ground truth fronts
    :return:
    """

    print("Calculating distance errors ...\n\n")

    print("")
    print(f"- Results for all images")
    print(f"\t- Number of images: {len(post_processed_dictionary)}")
    list_of_all_front_errors_without_nan, front_error_dict = calculate_front_delineation_metric(names,
                                                                                                post_processed_dictionary,
                                                                                                dictionary_of_target_fronts)

    front_error_individual_print(front_error_dict, names)

    front_delineation_print(front_error_dict, names, log_list)

    # Season subsetting
    seasons_dic = {
        "winter": [],
        "summer": []
    }
    for name in names:
        if check_whether_winter_half_year(name):
            seasons_dic["winter"].append(name)
        else:
            seasons_dic["summer"].append(name)

    for season in ["winter", "summer"]:
        print("")
        print(f"# Results for only images in {season}")
        target_names = seasons_dic[season]
        print(f"Number of images: {len(target_names)}")
        if len(target_names) == 0:
            continue
        front_delineation_print(front_error_dict, target_names, log_list)

    # glaciers
    glaciers_dic = {
        "Mapple": [],
        "COL": [],
        "Crane": [],
        "DBE": [],
        "JAC": [],
        "Jorum": [],
        "SI": []
    }
    for name in names:
        glaciers_dic[name[:-4].split('_')[0]].append(name)

    for glacier in glaciers_dic:
        print("")
        print(f"- Results for only images from {glacier}")
        front_delineation_print(front_error_dict, glaciers_dic[glacier], log_list)

    # Sensor subsetting
    sensors_dic = {
        "RSAT": [],
        "S1": [],
        "ENVISAT": [],
        "ERS": [],
        "PALSAR": [],
        "TSX/TDX": []
    }

    for name in names:
        key = name[:-4].split('_')[2]
        if key == "TSX" or key == "TDX":
            key = "TSX/TDX"
        sensors_dic[key].append(name)

    for sensor in sensors_dic.keys():
        print("")
        print(f"- Results for only images from sensor {sensor}")
        front_delineation_print(front_error_dict, sensors_dic[sensor], log_list)

    resolution_dic = {
        "20": [],
        "17": [],
        "12": [],
        "7": [],
        "6": []
    }
    for name in names:
        resolution_dic[name[:-4].split('_')[3]].append(name)

    for res in resolution_dic:
        print("")
        print(f"- Results for only images with a resolution of {res}")
        front_delineation_print(front_error_dict, resolution_dic[res], log_list)

    for glacier in glaciers_dic.keys():
        glacier_names = glaciers_dic[glacier]
        for season in seasons_dic.keys():
            print("")
            print(f"- Results for only images in {season} and from {glacier}")
            season_names = seasons_dic[season]
            filtered_list = list(set(glacier_names) & set(season_names))
            front_delineation_print(front_error_dict, filtered_list, log_list)

    for glacier in glaciers_dic.keys():
        glacier_names = glaciers_dic[glacier]
        for sensor in sensors_dic.keys():
            print("")
            print(f"- Results for only images of {sensor} and from {glacier}")
            sensor_names = sensors_dic[sensor]
            filtered_list = list(set(glacier_names) & set(sensor_names))
            front_delineation_print(front_error_dict, filtered_list, log_list)

    for glacier in glaciers_dic.keys():
        glacier_names = glaciers_dic[glacier]
        for res in resolution_dic.keys():
            print("")
            print(f"- Results for only images with resolution {res} and from {glacier}")
            res_names = resolution_dic[res]
            filtered_list = list(set(glacier_names) & set(res_names))
            front_delineation_print(front_error_dict, filtered_list, log_list)


def visualizations(names, postprocessed_dictionary, dictionary_of_target_fronts, sar_dict,
                   bounding_boxes_directory, visualizations_dir):
    print("Creating visualizations ...\n\n")
    for file_name in names:
        resolution = int(os.path.split(file_name)[1][:-4].split('_')[3])
        if resolution < 10:
            dilation = 9
        else:
            dilation = 3

        post_processed_predicted_mask = postprocessed_dictionary[file_name]
        target_front = dictionary_of_target_fronts[file_name]
        sar_image = sar_dict[file_name]

        predicted_front = np.array(post_processed_predicted_mask)
        ground_truth_front = np.array(target_front)
        kernel = np.ones((dilation, dilation), np.uint8)
        predicted_front = cv2.dilate(predicted_front, kernel, iterations=1)
        ground_truth_front = cv2.dilate(ground_truth_front, kernel, iterations=1)

        sar_image = np.array(sar_image)
        sar_image_rgb = skimage.color.gray2rgb(sar_image)
        sar_image_rgb = np.uint8(sar_image_rgb)

        sar_image_rgb[predicted_front > 0] = [0, 255, 255]  # b, g, r
        sar_image_rgb[ground_truth_front > 0] = [255, 51, 51]
        correct_prediction = np.logical_and(predicted_front, ground_truth_front)
        sar_image_rgb[correct_prediction > 0] = [255, 0, 255]

        # Insert Bounding Box
        matching_bounding_box_file = get_matching_out_of_folder(file_name, bounding_boxes_directory)
        with open(os.path.join(bounding_boxes_directory, matching_bounding_box_file)) as f:
            coord_file_lines = f.readlines()
        left_upper_corner_x, left_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[1].split(",")]
        left_lower_corner_x, left_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[2].split(",")]
        right_lower_corner_x, right_lower_corner_y = [round(float(coord)) for coord in coord_file_lines[3].split(",")]
        right_upper_corner_x, right_upper_corner_y = [round(float(coord)) for coord in coord_file_lines[4].split(",")]

        bounding_box = np.zeros((len(sar_image_rgb), len(sar_image_rgb[0])))

        if left_upper_corner_x < 0:
            left_upper_corner_x = 0
        if left_lower_corner_x < 0:
            left_lower_corner_x = 0
        if right_upper_corner_x > len(sar_image_rgb[0]):
            right_upper_corner_x = len(sar_image_rgb[0]) - 1
        if right_lower_corner_x > len(sar_image_rgb[0]):
            right_lower_corner_x = len(sar_image_rgb[0]) - 1
        if left_upper_corner_y > len(sar_image_rgb):
            left_upper_corner_y = len(sar_image_rgb) - 1
        if left_lower_corner_y < 0:
            left_lower_corner_y = 0
        if right_upper_corner_y > len(sar_image_rgb):
            right_upper_corner_y = len(sar_image_rgb) - 1
        if right_lower_corner_y < 0:
            right_lower_corner_y = 0

        bounding_box[left_upper_corner_y, left_upper_corner_x:right_upper_corner_x] = 1
        bounding_box[left_lower_corner_y, left_lower_corner_x:right_lower_corner_x] = 1
        bounding_box[left_lower_corner_y:left_upper_corner_y, left_upper_corner_x] = 1
        bounding_box[right_lower_corner_y:right_upper_corner_y, right_lower_corner_x] = 1
        bounding_box = cv2.dilate(bounding_box, kernel, iterations=1)
        sar_image_rgb[bounding_box > 0] = [255, 255, 0]

        # back(sar_image_rgb).save(os.path.join(visualizations_dir, file_name + ".png"))
        cv2.imwrite(os.path.join(visualizations_dir, file_name + ".png"), sar_image_rgb)


def load_zones_from_file(zones_dir):
    whole_dict = dict()

    for file in os.listdir(zones_dir):
        img = Image.open(os.path.join(zones_dir, file))
        name = file.split('.')[0]
        whole_dict[name] = whole_preprocess(img).astype(np.uint8)

    return whole_dict


def start_post_process(mode, names, prob_dir, complete_test_directory, uncertainty_dir,
                       complete_postprocessed_directory, visualizations_dir,
                       data_raw_parent_dir, data_parent_dir, zones_csv, experiment_name, on_cluster,
                       testtimeaugs=True, load_results_from_file=False, make_visualizations=True):
    log_list = []

    # #############################################################################################################
    # EVALUATE MODEL ON GIVEN DATASET
    # #############################################################################################################
    if load_results_from_file:
        complete_predicted_dict = load_zones_from_file(complete_test_directory)
    elif not testtimeaugs:
        print("No test time augmentations")
        complete_predicted_dict, _ = evaluate_model_on_given_dataset_no_testtimeaug(mode, prob_dir=prob_dir,
                                                                                    complete_test_directory=complete_test_directory,
                                                                                    uncertainty_dir=uncertainty_dir,
                                                                                    data_parent_dir=data_raw_parent_dir)
    else:
        complete_predicted_dict, _ = evaluate_model_on_given_dataset(mode, prob_dir=prob_dir,
                                                                     uncertainty_dir=uncertainty_dir,
                                                                     complete_test_directory=complete_test_directory,
                                                                     data_parent_dir=data_raw_parent_dir)

    # ###############################################################################################
    # CALCULATE SEGMENTATION METRICS (IoU & Hausdorff Distance)
    # ###############################################################################################
    if mode == "test":
        directory_of_complete_targets = os.path.join(data_raw_parent_dir, "data_raw", 'zones', 'test')
    else:
        directory_of_complete_targets = os.path.join(data_raw_parent_dir, "data_raw", 'zones', 'train')

    whole_zones_gt = load_whole_zones(directory_of_complete_targets)

    calculate_segmentation_metrics('zones', names, complete_predicted_dict, whole_zones_gt, log_list=log_list)

    # ###############################################################################################
    # POST-PROCESSING
    # ###############################################################################################
    bounding_boxes_directory = os.path.join(data_raw_parent_dir, "data_raw", "bounding_boxes")
    post_processed_dictionary = post_processing(names, complete_predicted_dict, bounding_boxes_directory,
                                                complete_postprocessed_directory, save_images=True)

    # ###############################################################################################
    # CALCULATE FRONT DELINEATION METRIC (Mean distance error)
    # ###############################################################################################
    if mode == "test":
        directory_of_target_fronts = os.path.join(data_raw_parent_dir, "data_raw", "fronts", 'test')
    else:
        directory_of_target_fronts = os.path.join(data_raw_parent_dir, "data_raw", "fronts", 'train')

    front_dictionary = load_fronts(front_directory=directory_of_target_fronts)
    front_delineation_metric(names, post_processed_dictionary, front_dictionary, log_list)

    with open(zones_csv, "a", newline='') as fp:
        writer = csv.writer(fp, dialect='excel-tab', delimiter=";")
        log_list.insert(0, experiment_name)
        writer.writerow(log_list)

    # ###############################################################################################
    # MAKE VISUALIZATIONS
    # ###############################################################################################
    if not make_visualizations:
        return
    if mode == "test":
        directory_of_sar_images = os.path.join(data_raw_parent_dir, "data_raw", "sar_images", 'test')
    else:
        directory_of_sar_images = os.path.join(data_raw_parent_dir, "data_raw", "sar_images", 'train')
    sar_dict = load_sar_images(directory_of_sar_images)

    # post_processed_dictionary = load_images(complete_postprocessed_directory)
    visualizations(names, post_processed_dictionary, front_dictionary, sar_dict,
                   bounding_boxes_directory, visualizations_dir)


if __name__ == '__main__':
    src = os.getcwd()
    outputdir = os.path.join(os.getcwd(), "checkpoints")

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--mode', default="test", help="Either 'val' or 'test'.")
    parser.add_argument('--data_raw_parent_dir', default=r"..",
                        help="Where the data_raw directory lies, default: .")

    parser.add_argument('--experiment_name', default="OptSimMIM_EnsembleTestTimeAugmentationOverlap",
                        help="The experiment name under which the validation results shall be saved. ")
    parser.add_argument('--on_cluster', action='store_true', help='If run on cluster no repo is used')

    load_results_from_file = False
    testtimeaugs = True

    hparams = parser.parse_args()
    data_raw_parent_dir = hparams.data_raw_parent_dir
    os.makedirs(os.path.join(os.getcwd(), "Ensemble_results", hparams.experiment_name), exist_ok=True)
    if hparams.mode == "test":
        output_parent_dir = os.path.join(os.getcwd(), "Ensemble_results", hparams.experiment_name, "output_images")
    else:
        output_parent_dir = os.path.join(os.getcwd(), "Ensemble_results", hparams.experiment_name, "output_images_val")

    data_parent_dir = data_raw_parent_dir

    visualizations_dir = os.path.join(output_parent_dir, "visualizations")
    complete_postprocessed_directory = os.path.join(output_parent_dir, "complete_postprocessed_images")
    complete_test_directory = os.path.join(output_parent_dir, "complete_images")
    prob_dir = os.path.join("D:\\", "TMP_hookformer", "complete_probabilities")
    uncertainty_directory = os.path.join(output_parent_dir, "uncertainty")
    if hparams.mode == "test":
        names_raw = os.listdir(os.path.join(data_raw_parent_dir, 'data_raw', 'sar_images', "test"))
    else:
        names_raw = os.listdir(os.path.join(data_raw_parent_dir, 'data_raw_val_as_test_set', 'sar_images', "test"))
    names = []
    for x in names_raw:
        names.append(x.split('.')[0])

    experiment_name = hparams.experiment_name
    csv_file = os.path.join(src, "results_" + hparams.mode + ".csv")
    on_cluster = hparams.on_cluster

    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline='') as fp:
            writer = csv.writer(fp, dialect='excel-tab')
            first_line = csv_files_utils.get_first_line()
            writer.writerow(first_line)

    os.makedirs(complete_test_directory, exist_ok=True)
    os.makedirs(complete_postprocessed_directory, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    os.makedirs(uncertainty_directory, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)

    import time
    start = time.time()

    start_post_process(hparams.mode, names, prob_dir, complete_test_directory, uncertainty_directory,
                       complete_postprocessed_directory, visualizations_dir,
                       data_raw_parent_dir, data_parent_dir, csv_file, experiment_name,
                       on_cluster, testtimeaugs=testtimeaugs, load_results_from_file=load_results_from_file,
                       make_visualizations=True)

    print("----------- TIME -------------")
    print(time.time() - start)
