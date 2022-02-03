import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config.Options import Config
from Models import *
from DatasetClass import DatasetClass
from LossFunctions import *
from EvaluationMetrics import *


class Main:
	def __init__(self):
		self.cfg = Config()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.train_dataset = DatasetClass(self.cfg)
		self.train_dataloader = DataLoader(self.train_dataset, self.cfg.batch_size, shuffle=True,
											num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
		self.load_model()
		if self.cfg.train_model:
			self.writer = SummaryWriter(self.cfg.tensorboard_log)
			self.loss_fn = nn.MSELoss()
			self.train()
		elif self.cfg.test_model:
			self.test()

	def load_model(self):
		self.model = Unet(image_channels=3, hidden_size=32).to(self.device)
		self.optim = optim.Adam(self.model.parameters(), self.cfg.lr, weight_decay=self.cfg.reg)
		self.epoch, self.best_loss = 0, 10
		if self.cfg.load_model and os.path.isfile(self.cfg.model_path):
			checkpoint = torch.load(self.cfg.model_path, map_location=self.device)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
			self.epoch = checkpoint['epoch']
			self.best_loss = checkpoint['loss']
			print("LOADED MODEL")

	def save_model(self, val_epoch_loss):
		if val_epoch_loss < self.best_loss:
			self.best_loss = val_epoch_loss
			torch.save({'epoch': self.epoch, 'model_state_dict': self.model.state_dict(),
						'optimizer_state_dict': self.optim.state_dict(), 'loss': self.best_loss},
						self.cfg.model_path)
			print("Saved model")

	def log_data_tensorboard(self, data):
		for key in data.keys():
			self.writer.add_scalar(key, data[key], self.epoch)

	def train(self):
		for self.epoch in range(self.epoch, self.cfg.epochs):
			train_epoch_loss = 0
			val_epoch_loss = 0
			for batch_idx, (data, label) in enumerate(self.train_dataloader):
				data, label = data.to(self.device), label.to(self.device)
				if batch_idx != len(self.train_dataloader) - 1:	# Train
					self.model.train()
					out = self.model(data)
					loss = self.loss_fn(out, label)
					self.optim.zero_grad()
					loss.backward()
					self.optim.step()
					train_epoch_loss += loss
				else:	# Validation
					self.model.eval()
					with torch.no_grad():
						out = self.model(data)
						loss = self.loss_fn(out, label)
						val_epoch_loss = loss
			log_data = {'Train Loss':train_epoch_loss.cpu().item() / (1 + batch_idx),
						'Val Loss': val_epoch_loss}
			self.log_data_tensorboard(log_data)
			self.save_model(val_epoch_loss)
			print("EPOCH {}: ".format(self.epoch), train_epoch_loss, val_epoch_loss)

	def test(self):
		self.model.eval()
		self.test_data = glob.glob(os.path.join(self.cfg.test_dir, 'image_2', "*.png"))
		for i, file in enumerate(self.test_data):
			img = cv2.resize(cv2.imread(file, cv2.IMREAD_UNCHANGED), (self.cfg.input_width, self.cfg.input_height))
			self.img_tensor = (torch.from_numpy(np.expand_dims(img.astype(np.uint8), 0)).to(
				self.device).float()/255).permute(0, 3, 1, 2)
			with torch.no_grad():
				out =  self.model(self.img_tensor).permute(0, 2, 3, 1).cpu().numpy()[0]
				cv2.imshow("OUTPUT", cv2.resize(out, (640, 480)))
				cv2.imshow("INPUT", cv2.resize(img, (640, 480)))
				k = cv2.waitKey()
				if k == ord('q'):
					exit()


if __name__ == '__main__':
	Main()
