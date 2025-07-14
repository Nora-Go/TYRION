import torch
from torch import Tensor
from torchmetrics import Metric


# This is here to save results so we can make the full image again
class MyPseudoSave(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        # parse inputs
        preds = self.preds
        target = self.target

        if isinstance(preds, list):
            preds = torch.cat(preds, dim=0)
            target = torch.cat(target, dim=0)

        return preds, target
