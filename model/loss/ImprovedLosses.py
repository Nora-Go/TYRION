import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Dice loss implementation

"""

class OGDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(OGDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# Classical Cross entropy with label smoothing. Class mode is still WIP
class ClassWiseSmoothCE(nn.Module):

    def __init__(self, eps=0.0, reduction='mean', weights=None, mode="normal"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.weights = weights
        self.mode = mode
        if mode == "class":
            self.eps = 0.0

    def forward(self, pred, gold, reduc='mean'):
        # casting gold if necessary
        if gold.dtype == torch.int32 or gold.dtype == torch.int8 or gold.dtype == torch.uint8:
            gold = gold.type(torch.int64)

        elif gold.dtype == torch.float32:
            gold = torch.round(gold).type(torch.int64)

        if self.eps >= 0.:
            pred = pred.permute(0, 2, 3, 1).flatten(end_dim=2)
            gold = gold.flatten().type(torch.int64)

            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred, device='cuda')

            one_hot = one_hot.scatter(1, gold.view(-1, 1), 1)
            one_hot_eps = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            if self.mode == "class":
                loss = -(one_hot_eps * log_prb).sum(dim=1)
                loss = torch.matmul(loss, one_hot_eps) / (one_hot_eps.sum(dim=0) + 0.00000001)
                return loss.sum() / torch.where(one_hot_eps.sum(dim=0) != 0, 1, 0).sum()

            else:
                loss = -(one_hot_eps * log_prb).sum(dim=1)
            if self.weights is not None:
                loss = self.weights * loss

            if reduc == 'nothing':
                return loss

            loss = torch.mean(loss)

            return loss

        return None
