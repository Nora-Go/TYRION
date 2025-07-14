import pytorch_lightning as pl
from lightly.models import utils
from model.utils import *
import copy
from CaFFe.constants import *
from model.modules.SwinV2 import *


class SimMIM(pl.LightningModule):
    def __init__(self, config, num_classes=4, mask_ratio=0.75, lr=0.06, weight_decay=1e-4, momentum=0.9):
        super().__init__()
        self.swin_unet = instantiate_from_config(config, num_classes=num_classes)
        self.mask_ratio = mask_ratio
        self.token_size = self.swin_unet.patch_embed.patch_size[0]
        self.sequence_length = self.swin_unet.patch_embed.num_patches
        decoder_dim = self.swin_unet.embed_dim # 96
        self.decoder = nn.Linear(decoder_dim, self.token_size**2 * 14)
        self.criterion = nn.L1Loss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.last_model_decoder_layer = torch.nn.Conv2d(in_channels=self.swin_unet.embed_dim, out_channels=3,
                                                        kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.swin_unet.embed_dim), requires_grad=False)
        vars = config['params']
        self.patch_embed_after_decoder = PatchEmbed(
            img_size=vars['img_size'], patch_size=vars['patch_size'], in_chans=vars['in_chans'],
            embed_dim=vars['embed_dim'], norm_layer=nn.LayerNorm if vars['patch_norm'] else None)

    def forward_encoder(self, images, batch_size, idx_mask):
        # pass all the tokens to the encoder, both masked and non masked ones
        x = self.swin_unet.prep_forward(images)
        if idx_mask is not None:
            x = utils.mask_at_index(x, idx_mask, self.mask_token)
        x, x_downsample = self.swin_unet.forward_down(x)
        return x, x_downsample

    def forward_model_decoder(self, x_encoded, x_downsample):
        # change last layer
        B, L, C = x_encoded.shape
        x = self.swin_unet.pred_target[0](x_encoded.reshape(B, int(L ** 0.5), int(L ** 0.5), C).permute(0, 3, 1, 2))
        for i in range(1, self.swin_unet.num_layers + 1):
            B, L, C = x_downsample[self.swin_unet.num_layers - i].shape
            x = self.swin_unet.pred_target[i](torch.cat((x, (
                x_downsample[self.swin_unet.num_layers - i].reshape(B, int(L ** 0.5), int(L ** 0.5), C).permute(0, 3, 1, 2))), dim=1))
        x = self.swin_unet.pred_target[-2](x)
        x = self.last_model_decoder_layer(x)
        return x

    def forward_decoder(self, x_encoded):
        return self.decoder(x_encoded)

    def step(self, batch_image, batch_image_optical, batch_idx):
        if batch_image.size()[1] == 1:
            batch_image = batch_image.repeat(1, 3, 1, 1)
        batch_size = batch_image.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=batch_image.device,
        )
        x_encoded, x_downsample = self.forward_encoder(batch_image, batch_size, idx_mask)
        x_encoded = self.forward_model_decoder(x_encoded, x_downsample)
        x_encoded = self.patch_embed_after_decoder(x_encoded)
        x_encoded_masked = utils.get_at_index(x_encoded, idx_mask)
        x_out = self.forward_decoder(x_encoded_masked)
        # get image patches for masked tokens
        patches = utils.patchify(batch_image_optical, self.token_size)
        target = utils.get_at_index(patches, idx_mask)
        loss = self.criterion(x_out, target)
        return loss

    def training_step(self, batch, batch_idx):
        batch_image = batch[CONTEXT]
        batch_image_optical = batch[CONTEXT_OPTICAL]
        loss = self.step(batch_image, batch_image_optical, batch_idx)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5),
                "monitor": "val/loss",
                "frequency": 1,
            },
        }

    def validation_step(self, batch, batch_idx):
        batch_image = batch[CONTEXT]
        batch_image_optical = batch[CONTEXT_OPTICAL]
        loss = self.step(batch_image, batch_image_optical, batch_idx)
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        batch_image = batch[CONTEXT]
        batch_image_optical = batch[CONTEXT_OPTICAL]
        loss = self.step(batch_image, batch_image_optical, batch_idx)
        self.log("test/loss", loss)
        return loss

    def load_from(self, pretrained_path):
        swin_unet = self.swin_unet
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
                msg = swin_unet.load_state_dict(pretrained_dict, strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})

                    current_layer_num = 3 - int(k[7:8]) - 1
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

            msg = swin_unet.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")
