import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(masks):
	return F.one_hot(torch.argmax(masks, axis=-1))

def un_one_hot(masks):
	return torch.argmax(masks, axis=-1)