import numpy as np
import skimage.measure
import skimage.color
import torch


# ################################################################################################################
# POSTPROCESSING PUTS THE PATCHES TOGETHER, SUBSTRACTS THE PADDING
# AND CHOOSES THE CLASS WITH HIGHEST PROBABILITY AS PREDICTION.
# SECONDLY, THE FRONT LINE IS EXTRACTED FROM THE PREDICTION
# ################################################################################################################


def postprocess_zone_segmenation(mask):
    """
    First, fill gaps in ocean region.
    Second, create Cluster of ocean mask and removes clusters except for the largest -> left with one big ocean
    :param
        mask: a numpy array representing the segmentation mask with 1 channel
    :return
        mask: a numpy array representing the filtered mask with 1 channel
    """

    # #############################################################################################
    # Fill Gaps in Ocean
    # #############################################################################################
    # get inverted ocean mask
    ocean_mask = mask == 254
    ocean_mask = np.invert(ocean_mask)
    labeled_image, num_cluster = skimage.measure.label(ocean_mask, connectivity=2, return_num=True)

    # extract largest cluster
    cluster_size = np.zeros(num_cluster + 1)
    for cluster_label in range(1, num_cluster + 1):
        cluster = labeled_image == cluster_label
        cluster_size[cluster_label] = cluster.sum()

    # create map of the gaps in ocean area
    if num_cluster > 1:
        labeled_image_torch = torch.from_numpy(labeled_image).to('cuda')
        most_frequent_cluster = torch.argmax((labeled_image_torch).unique(return_counts=True)[1][1:]) + 1
        gaps_mask = torch.where(labeled_image_torch >= 1, 1, 0)
        gaps_mask = torch.where(labeled_image_torch == most_frequent_cluster, 0, gaps_mask)
        # fill gaps
        mask = np.where(np.array(gaps_mask.to('cpu')) == 1, 254, mask)

    # Take largest connected component of ocean as ocean
    ocean_mask = mask >= 254  # Ocean (254)
    labeled_image, num_cluster = skimage.measure.label(ocean_mask, connectivity=2, return_num=True)
    if num_cluster == 0:
        return mask

    labeled_image_torch = torch.from_numpy(labeled_image).to('cuda')
    final_cluster = (torch.argmax((labeled_image_torch).unique(return_counts=True)[1][1:]) + 1)
    final_mask = labeled_image_torch == final_cluster
    final_mask = final_mask.to('cpu')

    mask[mask == 254] = 127
    mask[final_mask] = 254
    return mask


def extract_front_from_zones(zone_mask, front_length_threshold):
    # detect edge between ocean and glacier
    mask_mi = np.pad(zone_mask, ((1, 1), (1, 1)), mode='constant')
    mask_le = np.pad(zone_mask, ((1, 1), (0, 2)), mode='constant')
    mask_ri = np.pad(zone_mask, ((1, 1), (2, 0)), mode='constant')
    mask_do = np.pad(zone_mask, ((0, 2), (1, 1)), mode='constant')
    mask_up = np.pad(zone_mask, ((2, 0), (1, 1)), mode='constant')

    front = np.logical_and(mask_mi == 254,
                           np.logical_or.reduce((mask_do == 127, mask_up == 127, mask_ri == 127, mask_le == 127)))
    front = front[1:-1, 1:-1].astype(float)

    # delete too short fronts
    labeled_front, num_cluster = skimage.measure.label(front, connectivity=2, return_num=True)
    if num_cluster == 0:
        return front * 255

    cluster_frequencies = (torch.from_numpy(labeled_front).to('cuda')).unique(return_counts=True)[1]
    idx_of_kept_cluster = torch.argwhere(
        torch.where(cluster_frequencies >= front_length_threshold, cluster_frequencies, 0))
    front = np.array((torch.where(
        torch.isin(torch.from_numpy(labeled_front).to('cuda'), idx_of_kept_cluster.flatten()[1:]), 1, 0) * 255).to(
        'cpu'))
    return front
