import torch
import torch.nn as nn
import torch.nn.functional as F

class Evaluation_Metrics():
    def __init__(self):
        self.se = nn.MSELoss()
        pass

    def multi_class_dice_score(self, preds, targets):
        dice_score = 2 * torch.sum(preds * targets, axis=[2, 3]) / (torch.sum(preds, axis=[2, 3]) + torch.sum(targets, axis=[2, 3]))
        return dice_score.mean(axis=0)
    
    def accuracy(self, preds, targets):
        pred = F.one_hot(torch.argmax(preds, axis=1)).permute(0, 3, 1, 2)
        acc = (pred==targets).float().mean(axis=[2,3]).mean(axis=0)
        return acc

    def iou(self, preds, targets):
        intersection = torch.sum(preds * targets, axis=[2, 3])
        total_area = torch.sum(preds, axis=[2, 3]) + torch.sum(targets, axis=[2, 3])
        dice_score = intersection / (total_area - intersection)
        return dice_score.mean(axis=0)

    def evaluate(self, preds, targets):
        preds = F.softmax(preds, 1)
        targets = F.one_hot(targets).permute(0, 3, 1, 2)
        dice_score = self.multi_class_dice_score(preds, targets)
        accuracy = self.accuracy(preds, targets)
        iou = self.iou(preds, targets)
        return [dice_score.cpu().numpy(), accuracy.cpu().numpy(), iou.cpu().numpy()]
