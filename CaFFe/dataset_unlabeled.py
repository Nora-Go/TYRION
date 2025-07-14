from torch.utils.data import Dataset
import torchvision
from CaFFe.fetcher_pretraining import *
from model.utils import *
import omegaconf
from CaFFe.augmentations.OtherTransforms import NoTransform, NoTransform2, NoTransform4
import torchvision.transforms as T
import random
from CaFFe.constants import *
import cv2


class AbstractUnlabeledDataset(Dataset):
    def __init__(self, scale=1, augmentation_sample=NoTransform(), augmentation_target=NoTransform(),
                 augmentation_optical=NoTransform(),
                 double_augmentation=NoTransform2(), quadruple_augmentation=NoTransform4()):
        super(AbstractUnlabeledDataset, self).__init__()
        self.scale = scale

        if isinstance(augmentation_sample, omegaconf.dictconfig.DictConfig):
            augmentation_sample = instantiate_from_config(augmentation_sample)

        if isinstance(augmentation_target, omegaconf.dictconfig.DictConfig):
            augmentation_target = instantiate_from_config(augmentation_target)

        if isinstance(augmentation_optical, omegaconf.dictconfig.DictConfig):
            augmentation_optical = instantiate_from_config(augmentation_optical)

        if isinstance(double_augmentation, omegaconf.dictconfig.DictConfig):
            double_augmentation = instantiate_from_config(double_augmentation)

        if isinstance(quadruple_augmentation, omegaconf.dictconfig.DictConfig):
            quadruple_augmentation = instantiate_from_config(quadruple_augmentation)

        self.double_augmentation = double_augmentation
        self.quadruple_augmentation = quadruple_augmentation
        self.augmentation_sample = augmentation_sample
        self.augmentation_target = augmentation_target
        self.augmentation_optical = augmentation_optical

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, i):
        return NotImplementedError

    def apply_augmentations(self, image):
        image_sample = self.augmentation_sample(image)
        image_target = self.augmentation_target(image)

        if not torch.is_tensor(image_sample):
            image_sample = torch.from_numpy(image_sample)           # .type(torch.FloatTensor)
            image_target = torch.from_numpy(image_target)           # .type(torch.FloatTensor)

        image_sample, image_target = self.double_augmentation(image_sample, image_target)
        return image_sample, image_target

    def apply_augmentation_to_get_target(self, image):
        image_target = self.augmentation_target(image)
        if not torch.is_tensor(image_target):
            image_target = torch.from_numpy(image_target)           # .type(torch.FloatTensor)
        return image_target

    def apply_augmentations_target_existent(self, image, target):
        image = self.augmentation_sample(image)
        target = self.augmentation_target(target)
        # print("Shapes of image {} and target {} after single augment".format(image.shape, target.shape))

        # print("shape of sample: ", image.shape)
        # print("shape of target: ", target.shape)
        # back(np.einsum('chw -> hwc', image)).show()
        # back(np.einsum('chw -> hwc', target)).show()
        # # sys.exit(0)
        # print("after shape of sample: ", image.shape)
        # print("after shape of target: ", target.shape)

        if not torch.is_tensor(image):
            image = torch.from_numpy(image)     #.type(torch.FloatTensor)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)       #.type(torch.FloatTensor)

        image_sample, image_target = self.double_augmentation(image, target)
        return image_sample, image_target

    def apply_augmentations_all(self, image, optical, time_series):
        image_sample = self.augmentation_sample(image)
        image_target = self.augmentation_target(image)
        image_optical = self.augmentation_optical(optical)
        image_time = self.augmentation_target(time_series)

        if not torch.is_tensor(image_sample):
            image_sample = torch.from_numpy(image_sample)       #.type(torch.FloatTensor)
            image_target = torch.from_numpy(image_target)       #.type(torch.FloatTensor)
            image_time = torch.from_numpy(image_time)           #.type(torch.FloatTensor)
        if not torch.is_tensor(image_optical):
            image_optical = torch.from_numpy(image_optical)     #.type(torch.FloatTensor)

        image_sample, image_target, image_optical, image_time = self.quadruple_augmentation(image_sample, image_target, image_optical, image_time)
        return image_sample, image_target, image_optical, image_time


class Unlabeled_PatchDataSet_OptSimMIM(AbstractUnlabeledDataset):
    def __init__(self, patch_size, context_factor, parent_dir, parent_dir_optical, **kwargs):
        super().__init__(**kwargs)
        self.inner_patch_size = int(patch_size / context_factor)
        self.names, self.meta_data = fetch_patches_pretraining(parent_dir, self.inner_patch_size,
                                                               int(context_factor * self.inner_patch_size))
        self.names_optical_whole, self.meta_data_optical = fetch_whole_set_pretraining_optical(parent_dir_optical,
                                                                                               self.inner_patch_size)
        self.context_factor = context_factor
        print(len(self.names), "nr_of samples ")
        self.parent_dir = parent_dir

    def __len__(self):
        return len(self.names)

    def get_patch(self, img, inner_patch_size, context_size, idx):
        if len(img.shape) == 3:
            _, H, W = img.shape
        else:
            H, W = img.shape
        extra_add = -1
        if ((H + 1) // inner_patch_size * inner_patch_size + context_size) <= H and (
                ((W + 1) // inner_patch_size * inner_patch_size + context_size) <= W):
            extra_add = 0
        # HH = (H + 1) // inner_patch_size + extra_add
        WW = (W + 1) // inner_patch_size + extra_add
        # defines the top left corner of the patch
        row = idx // WW
        column = idx % WW
        if len(img.shape) == 3:
            context_crop = img[:, row * inner_patch_size:(row) * inner_patch_size + context_size,
                           column * inner_patch_size:(column) * inner_patch_size + context_size]
        else:
            context_crop = img[row * inner_patch_size:(row) * inner_patch_size + context_size,
                           column * inner_patch_size:(column) * inner_patch_size + context_size]

        return context_crop

    def __getitem__(self, item):
        name = self.names[item]
        suffix = ('_').join(name.split('_')[:-1])
        idx = int(name.split('_')[-1])
        data = self.meta_data[suffix]
        if os.path.basename(os.path.normpath(self.parent_dir))[0:9] == "unlabeled":
            glacier_name = name.split('_')[1]
        else:
            glacier_name = name.split('_')[0]
        if glacier_name == "COL":
            glacier_name = "Columbia"
        # print("names_optical[0].split('_')[0] ", self.names_optical[0].split('_')[0])
        idx_optical = [index for (index, item) in enumerate(self.names_optical_whole) if
                       item.split('_')[0] == glacier_name]
        data_optical = self.meta_data_optical[self.names_optical_whole[idx_optical[0]]]
        whole_image = data[IMAGE]
        whole_image_optical = data_optical[IMAGE]
        context = self.get_patch(whole_image, self.inner_patch_size, int(self.context_factor * self.inner_patch_size),
                                 idx)
        context_optical = self.get_patch(whole_image_optical, self.inner_patch_size,
                                         int(self.context_factor * self.inner_patch_size), idx)

        if not torch.is_tensor(context):
            context = torch.from_numpy(context).type(torch.FloatTensor)
            context_optical = torch.from_numpy(context_optical).type(torch.FloatTensor)
        return {
            "name": name,
            CONTEXT: context/255.0,
            CONTEXT_OPTICAL: context_optical/255.0
        }


class Unlabeled_PatchDataSet_OptTranslator(AbstractUnlabeledDataset):
    def __init__(self, patch_size, context_factor, parent_dir, parent_dir_optical, **kwargs):
        super().__init__(**kwargs)
        self.inner_patch_size = int(patch_size / context_factor)
        self.names, self.meta_data = fetch_patches_pretraining(parent_dir, self.inner_patch_size,
                                                               int(context_factor * self.inner_patch_size))
        self.names_optical_whole, self.meta_data_optical = fetch_whole_set_pretraining_optical(parent_dir_optical,
                                                                                               self.inner_patch_size)
        self.context_factor = context_factor
        print(len(self.names), "nr_of samples ")
        self.parent_dir = parent_dir

    def __len__(self):
        return len(self.names)

    def get_patch(self, img, inner_patch_size, context_size, idx):
        if len(img.shape) == 3:
            _, H, W = img.shape
        else:
            H, W = img.shape
        extra_add = -1
        if ((H + 1) // inner_patch_size * inner_patch_size + context_size) <= H and (
                ((W + 1) // inner_patch_size * inner_patch_size + context_size) <= W):
            extra_add = 0
        # HH = (H + 1) // inner_patch_size + extra_add
        WW = (W + 1) // inner_patch_size + extra_add
        # defines the top left corner of the patch
        row = idx // WW
        column = idx % WW
        if len(img.shape) == 3:
            context_crop = img[:, row * inner_patch_size:(row) * inner_patch_size + context_size,
                           column * inner_patch_size:(column) * inner_patch_size + context_size]
        else:
            context_crop = img[row * inner_patch_size:(row) * inner_patch_size + context_size,
                           column * inner_patch_size:(column) * inner_patch_size + context_size]

        return context_crop

    def __getitem__(self, item):
        name = self.names[item]
        suffix = ('_').join(name.split('_')[:-1])
        idx = int(name.split('_')[-1])
        data = self.meta_data[suffix]
        if os.path.basename(os.path.normpath(self.parent_dir))[0:9] == "unlabeled":
            glacier_name = name.split('_')[1]
        else:
            glacier_name = name.split('_')[0]
        if glacier_name == "COL":
            glacier_name = "Columbia"
        # print("names_optical[0].split('_')[0] ", self.names_optical[0].split('_')[0])
        idx_optical = [index for (index, item) in enumerate(self.names_optical_whole) if
                       item.split('_')[0] == glacier_name]
        data_optical = self.meta_data_optical[self.names_optical_whole[idx_optical[0]]]
        whole_image = data[IMAGE]
        whole_image_optical = data_optical[IMAGE]
        context = self.get_patch(whole_image, self.inner_patch_size, int(self.context_factor * self.inner_patch_size),
                                 idx)
        context_optical = self.get_patch(whole_image_optical, self.inner_patch_size,
                                         int(self.context_factor * self.inner_patch_size), idx)

        if not torch.is_tensor(context):
            context = torch.from_numpy(context).type(torch.FloatTensor)
            context_optical = torch.from_numpy(context_optical).type(torch.FloatTensor)
        return {
            "name": name,
            CONTEXT: context / 255.0,
            CONTEXT_OPTICAL: context_optical / 255.0
        }


class Random_Dataset_OptSimMIM(Dataset):
    def __init__(self, patch_size, parent_dir, parent_dir_optical, nr_of_samples_per_epoch=59463, min_scale=0.7,
                 random=True, **kwargs):
        super().__init__(**kwargs)
        self.nr_of_samples_per_epoch = nr_of_samples_per_epoch
        self.names, self.meta_data = fetch_whole_set_pretraining(parent_dir, patch_size)
        self.names_optical, self.meta_data_optical = fetch_whole_set_pretraining_optical(parent_dir_optical, patch_size)
        self.patch_size = patch_size
        self.center = torchvision.transforms.CenterCrop((self.patch_size, self.patch_size))
        print(nr_of_samples_per_epoch, "nr_of samples ")
        self.random = random
        self.min_scale = min_scale
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((self.patch_size, self.patch_size), interpolation=T.InterpolationMode.BICUBIC)

    def __len__(self):
        return self.nr_of_samples_per_epoch

    def get_random_patch(self, image, optical, patch_size):
        # images are padded by patch_size in h and width. That's why we can only take image.shape-1 patch_size.
        # also not to confuse with the other patch_size
        random_x = torch.randint(0, image.shape[1] - self.patch_size, (1,)).item()
        random_y = torch.randint(0, image.shape[2] - self.patch_size, (1,)).item()

        img = image[:, random_x:random_x + patch_size, random_y:random_y + patch_size]
        opt = optical[:, random_x:random_x + patch_size, random_y:random_y + patch_size]

        # "RandomResizedCrop"
        random_scale = torch.rand(1).item() * (1.0 - self.min_scale) + self.min_scale
        tmp_patch_size = int(self.patch_size * random_scale)
        random_x = torch.randint(0, img.shape[1] - tmp_patch_size, (1,)).item()
        random_y = torch.randint(0, img.shape[2] - tmp_patch_size, (1,)).item()
        img = img[:, random_x:random_x + tmp_patch_size, random_y:random_y + tmp_patch_size]
        opt = opt[:, random_x:random_x + tmp_patch_size, random_y:random_y + tmp_patch_size]
        img = np.expand_dims(cv2.resize(img[0], (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC), 0)
        opts = []
        for i in range(14):
            opt_i = cv2.resize(opt[i], (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            opts.append(opt_i)
        opt = np.stack(opts, axis=0)
        return img, opt

    def __getitem__(self, idx):
        name = self.names[idx % len(self.names)]
        glacier_name = name.split('_')[1]
        idx_optical = [index for (index, item) in enumerate(self.names_optical) if item.split('_')[0] == glacier_name]
        data = self.meta_data[name]
        data_optical = self.meta_data_optical[self.names_optical[idx_optical[0]]]
        whole_image = data[IMAGE]
        whole_image_optical = data_optical[IMAGE]
        context, context_optical = self.get_random_patch(whole_image, whole_image_optical, self.patch_size)

        if random.random() > 0.5:
            context = cv2.flip(context, 1)
            context_optical = cv2.flip(context_optical, 1)

        context = self.to_tensor(np.einsum('hwc -> chw', context))
        context_optical = self.to_tensor(np.einsum('hwc -> chw', context_optical))

        # print("shape of sample: ", context.shape)
        # print("shape of target: ", context_optical.shape)
        # print("shape of sample einsum: ", torch.einsum('hwc -> chw', context).shape)
        # print("shape of target einsum: ", torch.einsum('hwc -> chw', context_optical).shape)
        # back(torch.einsum('hwc -> chw', (context*255).type(torch.uint8))).show()
        # back(torch.einsum('hwc -> chw', (context_optical*255).type(torch.uint8))).show()
        return {
            "name": name,
            CONTEXT: context.permute(2, 0, 1),
            CONTEXT_OPTICAL: context_optical.permute(2, 0, 1)
        }


class Random_Dataset_OptTranslator(Dataset):
    def __init__(self, patch_size, parent_dir, parent_dir_optical, nr_of_samples_per_epoch=59463, min_scale=0.7,
                 random=True, **kwargs):
        super().__init__(**kwargs)
        self.nr_of_samples_per_epoch = nr_of_samples_per_epoch
        self.names, self.meta_data = fetch_whole_set_pretraining(parent_dir, patch_size)
        self.names_optical, self.meta_data_optical = fetch_whole_set_pretraining_optical(parent_dir_optical, patch_size)
        self.patch_size = patch_size
        self.center = torchvision.transforms.CenterCrop((self.patch_size, self.patch_size))
        print(nr_of_samples_per_epoch, "nr_of samples ")
        self.random = random
        self.min_scale = min_scale
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((self.patch_size, self.patch_size), interpolation=T.InterpolationMode.BICUBIC)

    def __len__(self):
        return self.nr_of_samples_per_epoch

    def get_random_patch(self, image, optical, patch_size):
        # images are padded by patch_size in h and width. That's why we can only take image.shape-1 patch_size.
        # also not to confuse with the other patch_size
        random_x = torch.randint(0, image.shape[1] - self.patch_size, (1,)).item()
        random_y = torch.randint(0, image.shape[2] - self.patch_size, (1,)).item()

        img = image[:, random_x:random_x + patch_size, random_y:random_y + patch_size]
        opt = optical[:, random_x:random_x + patch_size, random_y:random_y + patch_size]

        # "RandomResizedCrop"
        random_scale = torch.rand(1).item() * (1.0 - self.min_scale) + self.min_scale
        tmp_patch_size = int(self.patch_size * random_scale)
        random_x = torch.randint(0, img.shape[1] - tmp_patch_size, (1,)).item()
        random_y = torch.randint(0, img.shape[2] - tmp_patch_size, (1,)).item()
        img = img[:, random_x:random_x + tmp_patch_size, random_y:random_y + tmp_patch_size]
        opt = opt[:, random_x:random_x + tmp_patch_size, random_y:random_y + tmp_patch_size]
        img = np.expand_dims(cv2.resize(img[0], (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC), 0)
        opts = []
        for i in range(14):
            opt_i = cv2.resize(opt[i], (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            opts.append(opt_i)
        opt = np.stack(opts, axis=0)
        return img, opt

    def __getitem__(self, idx):
        name = self.names[idx % len(self.names)]
        glacier_name = name.split('_')[1]
        idx_optical = [index for (index, item) in enumerate(self.names_optical) if item.split('_')[0] == glacier_name]
        data = self.meta_data[name]
        data_optical = self.meta_data_optical[self.names_optical[idx_optical[0]]]
        whole_image = data[IMAGE]
        whole_image_optical = data_optical[IMAGE]
        context, context_optical = self.get_random_patch(whole_image, whole_image_optical, self.patch_size)

        if random.random() > 0.5:
            context = cv2.flip(context, 1)
            context_optical = cv2.flip(context_optical, 1)

        context = self.to_tensor(np.einsum('hwc -> chw', context))
        context_optical = self.to_tensor(np.einsum('hwc -> chw', context_optical))

        # print("shape of sample: ", context.shape)
        # print("shape of target: ", context_optical.shape)
        # print("shape of sample einsum: ", torch.einsum('hwc -> chw', context).shape)
        # print("shape of target einsum: ", torch.einsum('hwc -> chw', context_optical).shape)
        # back(torch.einsum('hwc -> chw', (context*255).type(torch.uint8))).show()
        # back(torch.einsum('hwc -> chw', (context_optical*255).type(torch.uint8))).show()
        return {
            "name": name,
            CONTEXT: context.permute(2, 0, 1),
            CONTEXT_OPTICAL: context_optical.permute(2, 0, 1)
        }
