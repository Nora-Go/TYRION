from torch.utils.data import Dataset
from PIL import Image
import torchvision
from CaFFe.fetcher import *
from model.utils import *
import omegaconf
from CaFFe.augmentations.OtherTransforms import NoTransform2, NoTransform4
from torchvision.transforms.functional import resize
from CaFFe.constants import *

"""
    Abstract Dataset to avoid redundancy
    args: 
        mask_suffix : suffix of the mask
        scale: I also want to know what that is
        prob_mix_up: probability for the mix_up augmentation
        augmentation: The augmentations like (contrast, brightness etc). Applies to image and context
        double_augmentation: Second set of augmentations that also applies to masks (like rotation etc)
        context_without_resize : If set the context is not resized to the same resolution as the image. This is kinda experimental. And has some bugs... WIP
    
"""


class AbstractDataset(Dataset):
    def __init__(self, mask_suffix='_zones_NA', scale=1, prob_mix_up=0.0, augmentation=NoTransform2(),
                 double_augmentation=NoTransform4(), context_without_resize=False, prob_rezoom=0.0):
        super(AbstractDataset, self).__init__()
        self.scale = scale
        self.context_without_resize = context_without_resize
        self.prob_mix_up = prob_mix_up
        self.prob_rezoom = prob_rezoom

        if isinstance(augmentation, omegaconf.dictconfig.DictConfig):
            augmentation = instantiate_from_config(augmentation)

        if isinstance(double_augmentation, omegaconf.dictconfig.DictConfig):
            double_augmentation = instantiate_from_config(double_augmentation)

        self.double_augmentation = double_augmentation
        self.augmentation = augmentation
        self.mask_suffix = mask_suffix

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, i):
        return NotImplementedError

    def mix_up_ocean_ice(self, img1, img2, mask, w_1, w_2):
        img1_np = img1.astype(np.float32)
        img2_np = img2.astype(np.float32)

        mask = np.expand_dims(mask, axis=0)

        # Case of a RGB image
        if img1.shape[0] == 3:
            mask = np.concatenate((mask, mask, mask))
            img2 = np.concatenate((img2, img2, img2))

        img = np.where(mask == OCEAN_ICE_ID, (w_1 * img1_np + w_2 * img2_np) / (w_1 + w_2), img1_np)
        img = np.clip(img, a_min=0, a_max=255)

        return img.astype(np.uint8)

    # TODO reintegrate tormentor maybe? It actually seems kinda nice
    # TODO calving front positions?
    # TODO I actually need to call augments on context and target image lmao

    def apply_augmentations(self, image, mask):

        image = self.augmentation(image)
        image = torch.from_numpy(image).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.int64)
        image, mask = self.double_augmentation(image, mask)
        return image, mask


"""
    Dataset that takes random patches of images
    args:
        patch_size: img_size essentially
        context_factor: How much larger is the context than the target img (usually 2x)
        parent_dir: where is the data 
        padding_mode: theoretically how things are padded. But I think I hardcoded symmetric somewhere...
        nr_of_samples_per_epoch: Since there is not set nr of samples. You can just hardcode how many steps are an epoch here
    
"""


class RandomPatchDataset(AbstractDataset):
    def __init__(self, patch_size, context_factor, parent_dir, split, padding_mode="symmetric",
                 nr_of_samples_per_epoch=59463, automatic_resizing=False, **kwargs):

        super().__init__(**kwargs)

        self.nr_of_samples_per_epoch = nr_of_samples_per_epoch
        self.padding_mode = padding_mode
        self.names, self.meta_data = fetch_whole_set(parent_dir, split, int(context_factor * patch_size), padding_mode,
                                                     automatic_resizing=automatic_resizing)
        self.patch_size = patch_size
        self.context_factor = context_factor
        self.center = torchvision.transforms.CenterCrop((self.patch_size, self.patch_size))

    def __len__(self):
        return self.nr_of_samples_per_epoch

    def get_random_patch(self, image, mask, patch_size):
        # images are padded by patch_size in h and width. That's why we can only take image.shape-1 patch_size. also not to confuse with the other patch_size
        random_zoom_factor = 1.0
        if self.prob_rezoom > torch.rand(1):
            random_zoom_factor = torch.rand(1) * 2.5 + 0.33

        random_x = torch.randint(0, max(1, image.shape[1] - int(random_zoom_factor * patch_size)), (1,)).item()
        random_y = torch.randint(0, max(1, image.shape[2] - int(patch_size * random_zoom_factor)), (1,)).item()

        img = image[:, random_x:min(image.shape[1], random_x + int(random_zoom_factor * patch_size)),
              random_y:min(image.shape[2], random_y + int(patch_size * random_zoom_factor))]
        mask = mask[random_x:min(image.shape[1], random_x + int(random_zoom_factor * patch_size)),
               random_y:min(image.shape[2], random_y + int(patch_size * random_zoom_factor))]

        if self.prob_rezoom > 0.0:
            img = np.array(resize(torch.from_numpy(img), (patch_size, patch_size)))
            mask = np.array(resize(torch.from_numpy(mask).unsqueeze(dim=0), (patch_size, patch_size))[0])

        return img, mask

    def get_image_from_context(self, context_img, context_mask):
        img = np.array(self.center(torch.from_numpy(context_img)))
        mask = np.array(self.center(torch.from_numpy(context_mask)))
        return img, mask

    def get_double_crop(self, image, mask, image2, mask2, patch_size):
        random_x = torch.randint(0, image.shape[1] - (self.patch_size * self.context_factor), (1,)).item()
        random_y = torch.randint(0, image.shape[2] - (self.patch_size * self.context_factor), (1,)).item()

        img = image[:, random_x:random_x + patch_size, random_y:random_y + patch_size]
        mask = mask[random_x:random_x + patch_size, random_y:random_y + patch_size]

        img2 = image2[:, random_x:random_x + patch_size, random_y:random_y + patch_size]
        mask2 = mask2[random_x:random_x + patch_size, random_y:random_y + patch_size]

        return img, mask, img2, mask2

    def __getitem__(self, i):
        name = self.names[i % len(self.names)]
        data = self.meta_data[name]

        whole_image = data[IMAGE]
        whole_mask = data[IMAGE_MASK]

        context, mask_context = self.get_random_patch(whole_image, whole_mask, self.patch_size * self.context_factor)

        if self.prob_mix_up > torch.rand(1):
            mix_up_factor = torch.rand(1) * 0.15 + 0.15
            data_mixup = self.meta_data[self.names[int(len(self.names) * torch.rand(1))]]
            mix_up_img = data_mixup[IMAGE]
            mix_up_mask = data_mixup[IMAGE_MASK]
            mix_up_context, _ = self.get_random_patch(mix_up_img, mix_up_mask,
                                                      self.patch_size * self.context_factor)  # self.get_all(mix_up_img, mix_up_mask)
            context = self.mix_up_ocean_ice(context, mix_up_context, mask_context, 1.0, mix_up_factor.item())

        context = torch.from_numpy(context)
        context, mask_context = self.apply_augmentations(context, mask_context)

        image, mask_image = self.get_image_from_context(np.array(context), np.array(mask_context))
        image = torch.from_numpy(image)
        mask_image = torch.from_numpy(mask_image)

        if not self.context_without_resize:
            context = resize(context, torch.from_numpy(image[0]).shape)
            mask_context = resize(mask_context, torch.from_numpy(mask_image).shape)

        return {
            "name": name,
            IMAGE: image,
            IMAGE_MASK: mask_image,
            CONTEXT: context,
            CONTEXT_MASK: mask_context,
        }


"""
    Organized Patch Dataset. This Dataset cuts the entire image into symmetrically padded images and then cuts them into patches on the fly.
    This avoids huge memory consumption and loading times to start. Also you can cut any patch_size without waiting.
    args: 
         look at RandomPatchDataset
         split: train/val/test
"""


class PatchDataSet(AbstractDataset):
    def __init__(self, patch_size, context_factor, parent_dir, split, automatic_resizing=False, **kwargs):

        super().__init__(**kwargs)

        self.names, self.meta_data = fetch_patches(parent_dir, split, patch_size, int(context_factor * patch_size),
                                                   automatic_resizing=automatic_resizing)
        self.patch_size = patch_size
        self.context_factor = context_factor
        print(len(self.names), "nr_of samples ")

    def __len__(self):
        return len(self.names)

    def get_patch(self, img, patch_size, context_size, idx):

        if len(img.shape) == 3:
            _, H, W = img.shape
        else:
            H, W = img.shape

        extra_add = -1
        if ((H + 1) // patch_size * patch_size + context_size) <= H and (
                ((W + 1) // patch_size * patch_size + context_size) <= W):
            extra_add = 0

        # HH = (H + 1) // patch_size + extra_add
        WW = (W + 1) // patch_size + extra_add

        # defines the top left corner of the patch
        row = idx // WW
        column = idx % WW

        if len(img.shape) == 3:
            context_crop = img[:, row * patch_size:(row) * patch_size + context_size,
                           column * patch_size:(column) * patch_size + context_size]
            # context_crop = context_crop[0]

        else:
            context_crop = img[row * patch_size:(row) * patch_size + context_size,
                           column * patch_size:(column) * patch_size + context_size]

        # output_img = np.array(transforms.CenterCrop((patch_size, patch_size))((torch.from_numpy(context_crop))))
        if not self.context_without_resize:
            if len(context_crop.shape) == 2:
                context_crop = np.array(transforms.Resize((patch_size, patch_size), interpolation=PIL.Image.NEAREST)(
                    torch.from_numpy(context_crop).unsqueeze(dim=0)))[0]

            else:
                context_crop = np.array(transforms.Resize((patch_size, patch_size), interpolation=PIL.Image.NEAREST)(
                    torch.from_numpy(context_crop)))
        else:
            context_crop = np.array((torch.from_numpy(context_crop)))

        return context_crop

    def get_image_from_context(self, context, patch_size):

        output_img = transforms.CenterCrop((patch_size, patch_size))(context)

        return output_img

    def __getitem__(self, item):
        name = self.names[item]

        suffix = ('_').join(name.split('_')[:-1])
        idx = int(name.split('_')[-1])

        data = self.meta_data[suffix]

        whole_image = data[IMAGE]
        whole_mask = data[IMAGE_MASK]

        context = self.get_patch(whole_image, self.patch_size, int(self.context_factor * self.patch_size), idx)
        mask_context = self.get_patch(whole_mask, self.patch_size, int(self.context_factor * self.patch_size), idx)
        # mask = np.round(mask*255).astype(np.uint8)
        # mask_context = np.round(mask_context*255).astype(np.uint8)

        context, mask_context = self.apply_augmentations(context, mask_context)
        image = self.get_image_from_context(context, self.patch_size)
        mask_image = self.get_image_from_context(mask_context, self.patch_size)

        return {
            "name": name,
            IMAGE: image,
            IMAGE_MASK: mask_image,
            CONTEXT: context,
            CONTEXT_MASK: mask_context,
        }


class OverlappingPatchDataSet(AbstractDataset):
    def __init__(self, patch_size, context_factor, overlap, parent_dir, split, **kwargs):
        super().__init__(**kwargs)
        self.names, self.meta_data = fetch_overlapping_patches(parent_dir, split, patch_size, overlap,
                                                               int(context_factor * patch_size))
        self.patch_size = patch_size
        self.context_factor = context_factor
        self.overlap = overlap
        print(len(self.names), "nr_of samples ")

    def __len__(self):
        return len(self.names)

    def get_patch(self, img, patch_size, context_size, row, column):
        if len(img.shape) == 3:
            context_crop = img[:, row:row + context_size, column:column + context_size]

        else:
            context_crop = img[row:row + context_size, column:column + context_size]
        context_crop = np.array((torch.from_numpy(context_crop)))
        return context_crop

    def get_image_from_context(self, context, patch_size):
        output_img = transforms.CenterCrop((patch_size, patch_size))(context)
        return output_img

    def __getitem__(self, item):
        name = self.names[item]

        suffix = name.split('__')[:-1][0]
        y = int(name.split('_')[-1])
        x = int(name.split('_')[-2])

        data = self.meta_data[suffix]

        whole_image = data[IMAGE]
        whole_mask = data[IMAGE_MASK]

        context = self.get_patch(whole_image, self.patch_size, int(self.context_factor * self.patch_size), x, y)
        mask_context = self.get_patch(whole_mask, self.patch_size, int(self.context_factor * self.patch_size), x, y)

        context, mask_context = self.apply_augmentations(context, mask_context)
        # back(mask_context.type(torch.FloatTensor) * 80).show()
        image = self.get_image_from_context(context, self.patch_size)
        mask_image = self.get_image_from_context(mask_context, self.patch_size)

        return {
            "name": name,
            IMAGE: image,
            IMAGE_MASK: mask_image,
            CONTEXT: context,
            CONTEXT_MASK: mask_context,
        }
