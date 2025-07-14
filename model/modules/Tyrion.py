import torch.nn as nn
from model.custom.collection import *
from model.modules.SwinV2 import *
import copy


class ResUpSampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, up_sample=True, **kwargs):
        super().__init__()
        self.block = ResnetBlock(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.up_sample = up_sample
        if self.up_sample:
            self.up = Upsample(in_channels=out_channels, with_conv=True)

    def forward(self, x):
        x = self.block(x)
        if self.up_sample:
            x = self.up(x)

        return x


class Tyrion(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=4,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 4, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, increased_window_factor=2,
                 use_checkpoint=False, final_upsample="expand_first", pretrained_window_sizes=[0, 0, 0, 0],
                 checkpoint_transformer=None, checkpoint_context=None, out_channels=256,
                 **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.downsample_steps = len(depths) + 1
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.window_size = window_size
        self.increased_window_factor = increased_window_factor

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_upt = nn.ModuleList()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)

            layer = BasicLayer(dim=dim,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])

            self.layers.append(layer)

        self.pred_target = nn.ModuleList()
        self.pred_target.append(ResUpSampleBlock(768, 768, up_sample=False))

        for i_layer in reversed(range(self.num_layers)):
            dim = int(embed_dim * 2 ** i_layer)
            out_channels = int(dim / 2) if (i_layer != 0) else dim

            self.pred_target.append(ResUpSampleBlock(2 * dim, out_channels))

        # Need one extra set of upsampling because patch embeddings are usually dowonscaling it by a factor of 4
        self.pred_target.append(ResUpSampleBlock(dim, out_channels))

        self.pred_target.append(torch.nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1))

        for bly in self.layers:
            bly._init_respostnorm()

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")

            self.outputt = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

        for bly in self.layers_upt:
            if isinstance(bly, BasicLayer_up):
                bly._init_respostnorm()

        self.load_from_pretrained(checkpoint_transformer)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_pretrained(self, ckpt_path):
        if ckpt_path is None:
            print("No checkpoint path for Decoder ")
            return

        pretrained_dict = torch.load(ckpt_path)['model']
        model_dict = self.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)

        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                              model_dict[k].shape))
                    del full_dict[k]
        msg = self.load_state_dict(full_dict, strict=False)
        print("********************************************************************************")
        print("Loaded from swin encoder")
        print(msg)
        print("********************************************************************************")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_down(self, x):

        x_downsample = []
        for inx, layer in enumerate(self.layers):
            x, x_down = layer(x)
            x_downsample.append(x_down)

        return x, x_downsample

    def forward(self, x):
        x = self.prep_forward(x)

        x, x_downsample = self.forward_down(x)

        B, L, C = x.shape

        x = self.pred_target[0](x.reshape(B, int(L ** 0.5), int(L ** 0.5), C).permute(0, 3, 1, 2))

        for i in range(1, self.num_layers + 1):
            B, L, C = x_downsample[self.num_layers - i].shape
            x = self.pred_target[i](torch.cat((x, (
                x_downsample[self.num_layers - i].reshape(B, int(L ** 0.5), int(L ** 0.5), C).permute(0, 3, 1, 2))),
                                              dim=1))

        x = self.pred_target[-2](x)

        x = self.pred_target[-1](x)
        return x, x_downsample

    def prep_forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
