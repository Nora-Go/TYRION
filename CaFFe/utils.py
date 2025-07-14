from torchvision import transforms


def pad_whole_image(img, target_size):
    W, H = img.size

    WW = (W // target_size) + 2
    HH = (H // target_size) + 2
    # If the image is smaller than the size given, then CenterCrop pads the image accordingly
    # pil_img = transforms.CenterCrop((HH * target_size, WW * target_size))(img)
    crop_height, crop_width = (HH * target_size, WW * target_size)
    image_width, image_height = img.size
    padding_ltrb = [
        int(round((crop_width - image_width) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height) / 2.0)) if crop_height > image_height else 0,
        int(round((crop_width - image_width + 1) / 2.0)) if crop_width > image_width else 0,
        int(round((crop_height - image_height + 1) / 2.0)) if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, fill=0, padding_mode="symmetric")(img)

    return pil_img


def pad_whole_image_overlapping(img, target_size, overlap, patch_size):
    # Pad for the extraction of middle part of the image
    padding_ltrb = [
        int((target_size - patch_size) // 2),
        int((target_size - patch_size) // 2),
        int(((target_size - patch_size) // 2) + ((target_size - patch_size) % 2)),
        int(((target_size - patch_size) // 2) + (target_size - patch_size) % 2),
    ]
    pad1 = transforms.Pad(padding_ltrb, fill=0, padding_mode="symmetric")(img)

    # Pad to due to not fitting size for target size
    padded_img_size = (pad1.size[0], pad1.size[1])
    bottom = target_size - (padded_img_size[1] % target_size)
    bottom = bottom % target_size  # if bottom is exactly patch_size then now it is 0
    right = target_size - (padded_img_size[0] % target_size)
    right = right % target_size  # if right is exactly patch_size then now it is 0
    if overlap > 0:
        bottom = (target_size - overlap) - ((padded_img_size[1] - target_size) % (target_size - overlap))
        right = (target_size - overlap) - ((padded_img_size[0] - target_size) % (target_size - overlap))
    image_width, image_height = padded_img_size
    crop_width, crop_height = (image_width + right, image_height + bottom)
    padding_ltrb = [
        0,
        0,
        right if crop_width > image_width else 0,
        bottom if crop_height > image_height else 0,
    ]
    pil_img = transforms.Pad(padding_ltrb, fill=0, padding_mode="symmetric")(pad1)
    return pil_img
