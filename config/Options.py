import os
import sys
import glob

class Config:
    def __init__(self):
        self.train_dir = 'data/training'
        self.val_dir = 'data/val'
        self.test_dir = 'data/testing'
        self.model_path = "models/seg.pt"
        self.tensorboard_log='runs/test_1'
        self.train_model = False
        self.test_model = True
        self.load_model = True
        self.input_width, self.input_height = 512, 224
        self.epochs, self.lr, self.reg = 100, 1e-3, 1e-4
        self.batch_size, self.num_workers = 8, 0
        self.augment_data = False
        self.train_dataset_size = len(glob.glob(os.path.join(self.train_dir, 'image_2', "*.png")))
        if not os.path.exists("models"):
        	os.makedirs("models")
        if not os.path.exists("runs"):
        	os.makedirs("runs")
