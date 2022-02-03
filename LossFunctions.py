import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from utils import *

class FocalLoss(nn.Module):
    def __init__(self, weights, alpha=0.9, gamma=2, beta=0.9):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.betas = torch.ones((weights.shape[0]))*beta
        self.weights = (1 - self.betas) / (1 - self.betas**torch.from_numpy(weights))
        self.CEloss = nn.CrossEntropyLoss(self.weights.float().to(self.device)).to(self.device)

    def forward(self, preds, targets):
        ce_loss = self.CEloss(preds, targets) 
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        # focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        preds = F.softmax(preds, 1)
        targets = F.one_hot(targets).permute(0, 3, 1, 2)
        dice_loss = 2 * torch.sum(preds*targets, axis=[2, 3]) / (torch.sum(preds, axis=[2, 3]) + torch.sum(targets, axis=[2, 3]))
        return 1 - dice_loss.mean()

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.CEloss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        # focal loss
        ce_loss = self.CEloss(preds, targets) 
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()

        # dice loss
        preds = F.softmax(preds, 1)
        targets = F.one_hot(targets).permute(0, 3, 1, 2)
        dice_loss = 2 * torch.sum(preds*targets, axis=[2, 3]) / (torch.sum(preds, axis=[2, 3]) + torch.sum(targets, axis=[2, 3]))
        dice_loss = 1 - dice_loss.mean()
       
        loss = focal_loss + dice_loss
        return loss

class CEDiceLoss(nn.Module):
    def __init__(self):
        super(CEDiceLoss, self).__init__()
        self.CEloss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        ce_loss = self.CEloss(preds, targets) 
        preds = F.softmax(preds, 1)
        targets = F.one_hot(targets).permute(0, 3, 1, 2)
        dice_loss = 2 * torch.sum(preds*targets, axis=[2, 3]) / (torch.sum(preds, axis=[2, 3]) + torch.sum(targets, axis=[2, 3]))
        dice_loss = 1 - dice_loss.mean()
        return ce_loss + dice_loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights, beta=0.9):
        super(WeightedDiceLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.betas = torch.ones((weights.shape[0]))*beta
        self.weights = (1 - self.betas) / (1 - self.betas**torch.from_numpy(weights))
        self.CEloss = nn.CrossEntropyLoss(self.weights.float().to(self.device)).to(self.device)

    def forward(self, preds, targets):
        ce_loss = self.CEloss(preds, targets) 
        preds = F.softmax(preds, 1)
        targets = F.one_hot(targets).permute(0, 3, 1, 2)
        dice_loss = 2 * torch.sum(preds*targets, axis=[2, 3]) / (torch.sum(preds, axis=[2, 3]) + torch.sum(targets, axis=[2, 3]))
        dice_loss = 1 - dice_loss.mean()
        return ce_loss + dice_loss

class SSIM(nn.Module):
	"""Layer to compute the SSIM loss between a pair of images"""
	def __init__(self):
		super(SSIM, self).__init__()
		self.mu_x_pool   = nn.AvgPool2d(3, 1)
		self.mu_y_pool   = nn.AvgPool2d(3, 1)
		self.sig_x_pool  = nn.AvgPool2d(3, 1)
		self.sig_y_pool  = nn.AvgPool2d(3, 1)
		self.sig_xy_pool = nn.AvgPool2d(3, 1)
		self.refl = nn.ReflectionPad2d(1)
		self.C1 = 0.01 ** 2
		self.C2 = 0.03 ** 2

	def forward(self, x, y):
		x = self.refl(x)
		y = self.refl(y)
		mu_x = self.mu_x_pool(x)
		mu_y = self.mu_y_pool(y)
		sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
		sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
		sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
		SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
		SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
		return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class ModdedLoss(nn.Module):
    def __init__(self):
        super(ModdedLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CEloss = nn.CrossEntropyLoss().to(self.device)
        self.SSIMloss = SSIM().to(self.device)

    def get_batch_weights(self, targets):
        unique_classes = torch.unique(targets)
        weights = torch.zeros((len(unique_classes)))
        pixels_batch = targets.shape[0] * targets.shape[1] * targets.shape[2]
        for idx, class_idx in enumerate(unique_classes):
            weights[idx] = 1 - (len(torch.where(targets==class_idx)[0]) / pixels_batch)
        return weights

    def edge_loss(self, pred, target):
        grad_pred_x = torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]), 1, keepdim=True)
        grad_pred_y = torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]), 1, keepdim=True)
        grad_target_x = torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True)
        grad_target_y = torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True)

        grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        grad_target_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        grad_target_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        grad_pred = grad_pred_x[:,:,:-1,:] + grad_pred_y[:,:,:,:-1]
        grad_target = grad_target_x[:,:,:-1,:] + grad_target_y[:,:,:,:-1]

        return (grad_target - grad_pred).mean()

    def forward(self, preds, targets):
        batch_weights = self.get_batch_weights(targets)
        ce_loss = nn.CrossEntropyLoss(batch_weights.float().to(self.device)).to(self.device)(preds, targets)
        targets_one_hot = F.one_hot(targets).permute(0, 3, 1, 2)
        preds = F.softmax(preds, 1)
        edge_loss = self.edge_loss(preds, targets_one_hot.float())
        dice_loss = 2 * torch.sum(preds*targets_one_hot, axis=[2, 3]) / (torch.sum(preds, axis=[2, 3]) + torch.sum(targets_one_hot, axis=[2, 3]))
        dice_loss = 1 - dice_loss.mean()
        return ce_loss + dice_loss + edge_loss # + self.SSIMloss(torch.argmax(preds, axis=1).float(),targets.float()).mean()

class Lossfn(nn.Module):
    def __init__(self, loss_fn, dataset_properties=None):
        super(Lossfn, self).__init__()
        if loss_fn=='CE':
            self.loss = nn.CrossEntropyLoss()
        elif loss_fn == 'Focal':
            self.loss = FocalLoss(dataset_properties)
        elif loss_fn == 'Dice':
            self.loss = DiceLoss()
        elif loss_fn == 'DiceFocal':
            self.loss = DiceFocalLoss(alpha=0.99)
        elif loss_fn =='CEDice':
            self.loss = CEDiceLoss()
        elif loss_fn == 'WeightedDice':
            self.loss = WeightedDiceLoss(dataset_properties)
        elif loss_fn == 'MSE':
            self.loss = nn.MSE()
        elif loss_fn == 'ModdedLoss':
            self.loss = ModdedLoss()