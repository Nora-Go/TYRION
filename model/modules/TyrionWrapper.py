from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import os.path
import pickle
from model.loss.PseudoSave import MyPseudoSave
import torchvision.transforms
import pytorch_lightning as pl
from CaFFe.constants import *
from model.loss.ImprovedLosses import OGDiceLoss, ClassWiseSmoothCE
from torchmetrics import JaccardIndex, Precision, Recall, F1Score
from model.utils import *
from Evaluation import post_processing, calculate_front_error_for_validation
from PIL import Image

logger = logging.getLogger(__name__)

"""
    Wrapper class for all the networks. self.swin_unet is the actual model which get's set via a config

"""


class TyrionWrapper(pl.LightningModule):
    def __init__(self, config, img_size=224, num_classes=4, zero_head=False, vis=False,
                 weights_classification=[0.5, 0.5], weights_segmentation=[0.5, 0.5], weights_deep=None,
                 cla_label_smoothing=0.1, lr=0.0001, ce_mode="class", beta=1.0, seg_loss=None, optimizer="SGD",
                 parent_dir_for_validation=None, patch_metrics=True, automatic_resizing=False):
        super(TyrionWrapper, self).__init__()

        self.patch_metrics = patch_metrics
        self.detailed_valdidation = parent_dir_for_validation is not None
        self.automatic_resizing = automatic_resizing
        if parent_dir_for_validation is not None:
            self.parent_dir = parent_dir_for_validation
            self.fronts_dir = os.path.join(self.parent_dir, FRONT_DIR_KEY, "train")
            zones_dir = os.path.join(self.parent_dir, ZONES_DIR, "train")
            val_idx = pickle.load(open(os.path.join(self.parent_dir, "val" + "_idx.txt"), "rb"))
            self.gt_whole = dict()
            self.fronts_dict = dict()
            self.id_to_suffix = dict()
            self.suffix_to_id = dict()

            self.bounding_box_dir = os.path.join(self.parent_dir, BOUNDING_BOX_DIR)

            # get all the sizes for zones. We don't actually need the zones since we got the patches hehe
            all_zones = os.listdir(zones_dir)
            all_zones.sort()
            for idx in val_idx:
                image_name = all_zones[idx]
                img = Image.open(os.path.join(zones_dir, image_name))
                suffix = ("_").join(image_name.split("_")[:-1])
                self.gt_whole[suffix] = turn_colors_to_class_labels_zones_torch(torch.from_numpy(np.array(img)))

                self.suffix_to_id[suffix] = idx
                self.id_to_suffix[idx] = suffix

            all_fronts = os.listdir(self.fronts_dir)
            all_fronts.sort()

            for idx in val_idx:
                image_name = all_fronts[idx]
                suffix = ("_").join(image_name.split("_")[:-1])
                img = Image.open(os.path.join(self.fronts_dir, image_name))
                self.fronts_dict[suffix] = img

        self.save_metric = MyPseudoSave()
        self.optimizer_type = optimizer
        self.patch_size = img_size
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.classification_loss = ClassWiseSmoothCE(eps=cla_label_smoothing, mode=ce_mode)
        if seg_loss is None:
            self.segmentation_loss = OGDiceLoss(beta=beta)
        else:
            self.segmentation_loss = instantiate_from_config(seg_loss)
        self.IoU = JaccardIndex(task='multiclass', num_classes=num_classes, average='none')
        self.precision = Precision(task='multiclass', num_classes=num_classes, average='none')
        self.recall = Recall(task='multiclass', num_classes=num_classes, average='none')
        self.f1_score = F1Score(task='multiclass', num_classes=num_classes, average='none')

        self.weights_classification = weights_classification
        self.weights_segmentation = weights_segmentation
        self.weights_deep = weights_deep
        self.lr = lr

        self.suffix_to_names = dict()
        self.test_results = dict()
        self.gt_test_results = dict()
        self.suffix_collection = set()

        self.swin_unet = instantiate_from_config(config, num_classes=self.num_classes)
        self.get_mid_window = torchvision.transforms.CenterCrop((img_size, img_size))

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        logits, side_outputs = self.swin_unet(x)
        return logits, side_outputs

    def training_step(self, batch, batch_idx):
        img = self.get_input(batch)
        pred_context, side_outputs = self(img)
        pred_img = self.get_mid_window(pred_context)
        l = self.calculate_loss(pred_img, pred_context, side_outputs, batch, "train")

        return l

    def validation_step(self, batch, batch_idx):
        img = self.get_input(batch)
        pred_context, side_outputs = self(img)
        pred_img = self.get_mid_window(pred_context)
        l = self.calculate_loss(pred_img, pred_context, side_outputs, batch, "val")

        return l

    def clean_up_before_epoch(self):
        if not self.detailed_valdidation:
            return

        del self.test_results
        del self.suffix_collection
        del self.suffix_to_names

        self.test_results = dict()
        self.suffix_collection = set()
        self.suffix_to_names = dict()

    def on_validation_start(self) -> None:
        if not self.detailed_valdidation:
            return

        self.clean_up_before_epoch()
        self.save_metric.reset()

    def on_validation_epoch_end(self):  #

        if not self.detailed_valdidation:
            return

        patches, idx_output = self.save_metric.compute()

        # This is to avoid sanity checks. We don't wanna compute the mde with 2 patches and crash
        if self.trainer.sanity_checking:
            self.log("val/mde", 7777, on_epoch=True)
            self.log("val/mde_picture_normalized", 7777, on_epoch=True)
            self.log("val/mde_front_length_normalized", 7777, on_epoch=True)
            self.log("val/missing_fronts", 7777, on_epoch=True)
            return

        # reconstruct patches with names from pseudo save output
        test_results = dict()
        suffix_to_names = dict()

        for i in range(idx_output.shape[0]):
            suffix = self.id_to_suffix[int(idx_output[i, 0, 0].item())]
            patch_nr = '_{:03d}'.format((int(idx_output[i, 1, 1].item())))
            fullname = suffix + "_" + patch_nr
            if suffix_to_names.get(suffix) is None:
                suffix_to_names[suffix] = set()
            suffix_to_names[suffix].add(fullname)
            test_results[fullname] = patches[i]

        suffix_list = list(self.suffix_collection)
        whole_dict = dict()
        iou = []
        f1 = []
        precision = []
        recall = []

        # now go through all suffixes-> get patches for suffix -> and build image -> compute small metric
        for suffix in suffix_list:
            img_patches = []
            names = list(suffix_to_names[suffix])
            names.sort()

            for exact_name in names:
                img_patches.append(test_results[exact_name])

            whole_gt = self.gt_whole[suffix]  # .to('cuda')
            W, H = whole_gt.shape
            if self.automatic_resizing:
                resolution_factor = int(suffix.split('_')[-3])
                rescaling_factor = resolution_factor / STATIC_ZOOM_FACTOR
                W = int(rescaling_factor * W)
                H = int(rescaling_factor * H)

            HH = H // self.patch_size + 1
            WW = W // self.patch_size + 1
            length = self.patch_size

            whole_img = torch.zeros((WW * self.patch_size, HH * self.patch_size), dtype=torch.int32)  # ,device='cuda')
            for k in range(len(img_patches)):
                whole_img[length * (k // HH):length * (k // HH + 1), length * (k % HH):length * (k % HH + 1)] = \
                img_patches[k]

            whole_img = torchvision.transforms.CenterCrop((W, H))(whole_img)
            if self.automatic_resizing:
                whole_img = torchvision.transforms.Resize(whole_gt.shape,
                                                          interpolation=torchvision.transforms.InterpolationMode.NEAREST)(
                    whole_img.unsqueeze(dim=0))[0]

            whole_dict[suffix] = whole_img.to('cpu')
            # in case you wanna save images to look at
            # back(turn_class_labels_to_zones_torch(whole_img)).save(os.path.join("path",suffix+".png"))

            whole_img = whole_img.to('cuda')
            whole_gt = whole_gt.to('cuda')

            iou.append(self.IoU(whole_img, whole_gt))
            f1.append(self.f1_score(whole_img, whole_gt))
            recall.append(self.recall(whole_img, whole_gt))
            precision.append(self.precision(whole_img, whole_gt))

        # aggregate over image metrics
        iou = sum(iou) / len(iou)
        f1 = sum(f1) / len(f1)
        recall = sum(recall) / len(recall)
        precision = sum(precision) / len(precision)

        total_iou = 0.0
        total_precision = 0.0
        total_f1 = 0.0
        total_recall = 0.0
        for i in range(self.num_classes):
            self.log("val/iou_" + ID_TO_NAMES[i] + "_epoch", iou[i].mean(), on_epoch=True)
            self.log("val/f1_" + ID_TO_NAMES[i] + "_epoch", f1[i].mean(), on_epoch=True)
            self.log("val/recall_" + ID_TO_NAMES[i] + "_epoch", recall[i].mean(), on_epoch=True)
            self.log("val/precision_" + ID_TO_NAMES[i] + "_epoch", precision[i].mean(), on_epoch=True)
            total_recall += recall[i]
            total_f1 += f1[i]
            total_precision += precision[i]
            total_iou += iou[i]

        self.log("val/iou_epoch", total_iou / self.num_classes, on_epoch=True)
        self.log("val/f1_epoch", total_f1 / self.num_classes, on_epoch=True)
        self.log("val/recall_epoch", total_recall / self.num_classes, on_epoch=True)
        self.log("val/precision_epoch", total_precision / self.num_classes, on_epoch=True)

        # compute the mde
        post_processed_dict = post_processing(suffix_list, whole_dict, self.bounding_box_dir, None, save_images=False)
        front_error, front_error_per_image, front_error_front_length_normalized, missing_images = calculate_front_error_for_validation(
            suffix_list, post_processed_dict, self.fronts_dict)
        if torch.isnan(front_error):
            self.log("val/mde", 7777, on_epoch=True)
            self.log("val/mde_picture_normalized", 7777, on_epoch=True)
            self.log("val/mde_front_length_normalized", 7777, on_epoch=True)
            self.log("val/missing_fronts", 7777, on_epoch=True)
        else:
            self.log("val/mde", front_error, on_epoch=True)
            self.log("val/mde_picture_normalized", front_error_per_image, on_epoch=True)
            self.log("val/mde_front_length_normalized", front_error_front_length_normalized, on_epoch=True)
            self.log("val/missing_fronts", missing_images, on_epoch=True)

        print(front_error)

    def compute_patch_metrics(self, pred_img, target_img, split, name, recall=True):
        IoU_img = self.IoU(torch.nn.functional.softmax(pred_img, dim=1), target_img)
        IoU_img[torch.isnan(IoU_img)] = 0.0
        one_hot = torch.nn.functional.one_hot(target_img, num_classes=4)
        nr_of_instances_per_class_img = one_hot.flatten(end_dim=-2).sum(dim=0)
        self.log(split + "/IoU_" + name, IoU_img.sum() / torch.where(nr_of_instances_per_class_img > 0, 1, 0).sum())

        for i in range(len(ID_TO_NAMES)):
            if nr_of_instances_per_class_img[i] != 0.0:
                self.log(split + "/IoU_" + name + "_" + ID_TO_NAMES[i], IoU_img[i])

        if recall:
            pred_labels_img = torch.argmax(torch.nn.functional.softmax(pred_img, dim=1), dim=1)
            recall = self.recall(pred_labels_img, target_img)
            for i in range(len(ID_TO_NAMES)):
                if nr_of_instances_per_class_img[i] != 0.0:
                    self.log(split + "/Recall_" + name + "_" + ID_TO_NAMES[i], recall[i])

    def test_step(self, batch, batch_idx):
        img = self.get_input(batch)
        pred_context, side_outputs = self(img)
        pred_img = self.get_mid_window(pred_context)
        l = self.calculate_loss(pred_img, pred_context, side_outputs, batch, "test")

        return l

    def get_input(self, batch):
        context = batch[CONTEXT]
        return context

    def get_labels(self, batch):
        target_img = batch[IMAGE_MASK]
        context_target = batch[CONTEXT_MASK]

        return target_img, context_target

    def calculate_loss(self, pred_img, pred_context, xx, batch, split):
        target_img, target_context = self.get_labels(batch)

        masks = [target_img]
        for i in range(self.swin_unet.downsample_steps):
            big_mask = masks[-1]
            small_mask = torch.nn.functional.avg_pool2d(big_mask.cpu(), 2)
            masks.append(small_mask.to('cuda'))

        # Calculate losses
        loss_ce_img = self.classification_loss(pred_img, target_img)
        loss_ce_context = self.classification_loss(pred_context, target_context)

        loss_dice_img = self.segmentation_loss(pred_img, target_img)
        loss_dice_context = self.segmentation_loss(pred_context, target_context)

        loss_unet = self.weights_classification[0] * loss_ce_img + self.weights_classification[1] * loss_ce_context + \
                    self.weights_segmentation[0] * loss_dice_img + self.weights_segmentation[1] * loss_dice_context

        # Logging everything
        self.log(split + "/ce_img", loss_ce_img)
        self.log(split + "/ce_context", loss_ce_context)
        self.log(split + "/dice_img", loss_dice_img)
        self.log(split + "/dice_context", loss_dice_context)

        # Deep supervision logging. It's extra because some models don't have it

        loss_dice_deep = 0.0
        loss_ce_deep = 0.0
        loss_deep = 0.0
        if not self.weights_deep is None:
            loss_ce_deep = self.classification_loss(xx, masks[-1][:])
            loss_dice_deep = self.segmentation_loss(xx, masks[-1][:])
            self.log(split + "/ce_deep", loss_ce_deep)
            self.log(split + "/dice_deep", loss_dice_deep)
            loss_deep = self.weights_deep[0] * loss_ce_deep + self.weights_deep[1] * loss_dice_deep

        loss = loss_unet + loss_deep
        self.log(split + "/loss", loss)

        # logging IoU etc. Turned it off for training to avoid unnecessary computations
        if (split == "val" or split == ' test') and self.patch_metrics:
            self.compute_patch_metrics(pred_img, target_img, split, "img", recall=True)
            self.compute_patch_metrics(pred_context, target_context, split, "context", recall=False)

            # Testing has it's on special evaluation script don't need to do it here
            if self.detailed_valdidation and split == "val":
                self.save_results(pred_img, target_img, batch["name"])

        # TODO refactor
        if split == "test":

            pred_label_img = torch.argmax(torch.nn.functional.softmax(pred_img, dim=1), dim=1)
            pred_label_context = torch.argmax(torch.nn.functional.softmax(pred_context, dim=1), dim=1)

            total_pixel = 1
            for di in pred_label_context.shape:
                total_pixel = total_pixel * di

            for i in range(len(batch["name"])):
                suffix = batch["name"][i].split('_')[:-1]
                suffix = '_'.join(suffix)
                if self.suffix_to_names.get(suffix) is None:
                    self.suffix_to_names[suffix] = []

                self.suffix_to_names[suffix].append(batch["name"][i])
                self.test_results[batch["name"][i]] = turn_class_labels_to_zones_torch(pred_label_img[i])

        return loss

    def save_results(self, pred_img, target_img, names):
        predicted_labels = torch.argmax(torch.nn.functional.softmax(pred_img, dim=1), dim=1)
        for i in range(len(names)):

            patch_id = int(names[i].split('_')[-1].split('.')[0])
            suffix = names[i].split('_')[:-1]
            suffix = '_'.join(suffix)
            global_id = int(self.suffix_to_id[suffix])

            if self.suffix_to_names.get(suffix) is None:
                self.suffix_to_names[suffix] = []

            pseudo_target = torch.zeros(target_img[i].shape, device='cuda')
            pseudo_target[0, 0] = global_id
            pseudo_target[1, 1] = patch_id

            self.suffix_collection.add(suffix)
            self.save_metric.update(predicted_labels[i].unsqueeze(dim=0), pseudo_target.unsqueeze(dim=0))

    def configure_optimizers(self):
        if self.optimizer_type == "SGD":

            optim = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

        elif self.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10)
            metric = "val/IoU_img"
            interval = "epoch"
            optim = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": metric,
                    "interval": interval,
                    "frequency": 1,
                }}
        else:
            return NotImplementedError

        return optim

    def load_from(self, pretrained_path):  # config):
        # pretrained_path = config.MODEL.RESUME
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})

                    current_layer_num = 3 - int(k[7:8]) - 1
                    print(current_layer_num)
                    current_k = "layers_upt." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})

                    current_k = "layerst." + k[7:]
                    full_dict.update({current_k: v})


            for k, v in pretrained_dict.items():
                if "patch_embed." in k:
                    current_k = "patch_embedt." + k[12:]
                    full_dict.update({current_k: v})

            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                                  model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")
