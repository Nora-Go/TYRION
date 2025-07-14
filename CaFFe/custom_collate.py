from CaFFe.constants import *
import torch


def custom_collate(batch):
    names = []
    imgs = []
    img_labels = []
    contexts = []
    context_labels = []


    for item in batch:
        names.append(item["name"])
        imgs.append(item[IMAGE])
        img_labels.append(item[IMAGE_MASK])
        contexts.append(item[CONTEXT])
        context_labels.append(item[CONTEXT_MASK])

    imgs_t = None
    if len(imgs) > 0:
        imgs_t = torch.stack(imgs)

    contexts_t = None
    if len(contexts) > 0:
        contexts_t = torch.stack(contexts)

    img_labels_t = None
    if len(img_labels) > 0:
        img_labels_t = torch.stack(img_labels)

    context_labels_t = None
    if len(context_labels) > 0:
        context_labels_t = torch.stack(context_labels)

    out_dict = {
        "name": names,
        IMAGE: imgs_t / 255.0,
        CONTEXT: contexts_t / 255.0,
        IMAGE_MASK: img_labels_t,
        CONTEXT_MASK: context_labels_t,
    }

    return out_dict
