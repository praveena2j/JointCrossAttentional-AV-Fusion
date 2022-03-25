import torch.nn as nn
import torch

class CCCLoss(nn.Module):
    """
    Lin's Concordance correlation coefficient
    """
    def __init__(self, ignore=-5.0):
        super(CCCLoss, self).__init__()
        self.ignore = ignore

    def forward(self, y_pred, y_true):
        """
        y_true: shape of (N, )
        y_pred: shape of (N, )
        """
        batch_size = y_pred.size(0)
        device = y_true.device
        index = y_true != self.ignore
        index.requires_grad = False

        y_true = y_true[index]
        y_pred = y_pred[index]
        if y_true.size(0) <= 1:
            loss = torch.tensor(0.0, requires_grad=True).to(device)
            return loss
        x_m = torch.mean(y_pred)
        y_m = torch.mean(y_true)

        x_std = torch.std(y_true)
        y_std = torch.std(y_pred)

        v_true = y_true - y_m
        v_pred = y_pred - x_m

        s_xy = torch.sum(v_pred * v_true)

        numerator = 2 * s_xy
        denominator = x_std ** 2 + y_std ** 2 + (x_m - y_m) ** 2 + 1e-8

        ccc = numerator / (denominator * batch_size)
        loss = torch.mean(1 - ccc)
        return loss