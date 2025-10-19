import torch

def ranking_loss(y_pred, y_true, scale_ = 2.0, margin_ = 1):
    y_pred = y_pred * scale_
    y_true_ = y_true.float()
    tmp = margin_ - y_pred[:, None, :] + y_pred[:, :, None]
    partial_losses = torch.maximum(torch.zeros_like(tmp), tmp)
    loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sum(loss, dim=-1)
    return torch.mean(loss)

def contrastive_loss(logits, target):
    gt = target

    loss1 = -torch.sum(gt * torch.nn.functional.log_softmax(logits / 2.0, dim=1)) / gt.sum()
    loss2 = -torch.sum(gt.t() * torch.nn.functional.log_softmax(logits.t() / 2.0, dim=1)) / gt.sum()
    return (loss1 + loss2) / 2
    