import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from torchstat import stat
from torchsummary import summary

class DownBlockA(torch.nn.Module):
    def __init__(self, input_channels=16, output_channels=16):
        super(DownBlockA, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm1 = nn.GroupNorm(output_channels//4, output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm2 = nn.GroupNorm(output_channels//4, output_channels)
        self.pool = nn.MaxPool2d(2, 2, padding=0)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        self.layer1 = F.leaky_relu(self.group_norm1(self.conv1(x)))
        self.layer2 = F.leaky_relu(self.group_norm2(self.conv2(self.layer1)))
        self.out = self.dropout(self.pool(self.layer2))
        return self.out, self.layer2

class UpBlockA(torch.nn.Module):
    def __init__(self, input_channels=16, output_channels=16):
        super(UpBlockA, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upsample = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=output_channels*2, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm1 = nn.GroupNorm(output_channels//4, output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm2 = nn.GroupNorm(output_channels//4, output_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, x_cat):
        x = self.upsample(x)
        x = torch.cat((x, x_cat), 1)
        self.layer1 = F.leaky_relu(self.group_norm1(self.conv1(x)))
        self.layer2 = F.leaky_relu(self.group_norm2(self.conv2(self.layer1)))
        self.out = self.dropout(self.layer2)
        return self.out

class UpBlockB(torch.nn.Module):
    def __init__(self, input_channels=16, output_channels=16):
        super(UpBlockB, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upsample = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm1 = nn.GroupNorm(output_channels//4, output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm2 = nn.GroupNorm(output_channels//4, output_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, x_cat):
        x = self.upsample(x)
        self.layer1 = F.leaky_relu(self.group_norm1(self.conv1(x)))
        self.layer2 = F.leaky_relu(self.group_norm2(self.conv2(self.layer1)))
        self.out = self.dropout(self.layer2)
        return self.out

class DownBlockResnetA(torch.nn.Module):
    def __init__(self, input_channels=16, output_channels=16):
        super(DownBlockResnetA, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm = nn.GroupNorm(output_channels//4, output_channels)
        self.pool = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        self.block1 = F.leaky_relu(self.conv1(x))
        self.block2_1 = F.leaky_relu(self.conv2_1(self.block1))
        self.block2_2 = F.leaky_relu(self.block1 + self.conv2_2(self.block2_1))
        self.out = self.dropout(F.leaky_relu(self.group_norm(self.pool(self.block2_2))))
        return self.out, self.block2_2

class UpBlockResnetB(torch.nn.Module):
    def __init__(self, input_channels=16, output_channels=16):
        super(UpBlockResnetB, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.ConvTranspose2d(in_channels=output_channels, out_channels=output_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm1 = nn.GroupNorm(output_channels//4, output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.group_norm2 = nn.GroupNorm(output_channels//4, output_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, x_cat):
        x = self.upsample(self.block1(x))
        self.layer1 = F.leaky_relu(self.group_norm1(self.conv1(x)))
        self.layer2 = F.leaky_relu(x + self.group_norm2(self.conv2(self.layer1)))
        self.out = self.dropout(self.layer2)
        return self.out

class Unet(torch.nn.Module):
    def __init__(self, image_channels=3, hidden_size=32, n_classes=4):
        super(Unet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # filters = [32, 128, 512, 1536, 2048]
        filters = [64, 128, 256, 512, 1024]
        # Encoder                                                                            # 512x224x3 = 3,44,064
        self.conv1 = DownBlockResnetA(input_channels=image_channels, output_channels=filters[0])  # 256x112x32 = 9,17,504
        self.conv2 = DownBlockResnetA(input_channels=filters[0], output_channels=filters[1])   # 128x56x128 = 9,17,504
        self.conv3 = DownBlockResnetA(input_channels=filters[1], output_channels=filters[2])     # 64x28x512 = 9,17,504
        self.conv4 = DownBlockResnetA(input_channels=filters[2], output_channels=filters[3])    # 32x14x1536 = 6,88,128

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(in_channels=filters[3], out_channels=filters[4], kernel_size=3, stride=1, padding=1)    # 16x7x2048 = 2,29,376
        self.bottleneck_batch = nn.GroupNorm(filters[4]//4, filters[4])
        self.dropout = nn.Dropout(0.2)

        # Decoder
        self.up1 = UpBlockResnetB(input_channels=filters[4], output_channels=filters[3])     # 32x14x1536 = 6,88,128
        self.up2 = UpBlockResnetB(input_channels=filters[3], output_channels=filters[2])    # 64x28x512 = 9,17,504
        self.up3 = UpBlockResnetB(input_channels=filters[2], output_channels=filters[1])     # 128x56x128 = 9,17,504
        self.up4 = UpBlockResnetB(input_channels=filters[1], output_channels=filters[0])       # 256x112x32 = 9,17,504

        # Final Layer
        self.conv_out = nn.Conv2d(in_channels=filters[0], out_channels=image_channels, kernel_size=1, stride=1, padding=0) # 512x224x3 = 3,44,064


    def forward(self, x):
        self.enc_layer1, self.f1 = self.conv1(x)
        self.enc_layer2, self.f2 = self.conv2(self.enc_layer1)
        self.enc_layer3, self.f3 = self.conv3(self.enc_layer2)
        self.enc_layer4, self.f4 = self.conv4(self.enc_layer3)

        self.bottleneck_layer = self.dropout(F.leaky_relu(self.bottleneck_batch(self.bottleneck_conv(self.enc_layer4))))

        self.dec_layer1 = self.up1(self.bottleneck_layer, self.f4)
        self.dec_layer2 = self.up2(self.dec_layer1, self.f3)
        self.dec_layer3 = self.up3(self.dec_layer2, self.f2)
        self.dec_layer4 = self.up4(self.dec_layer3, self.f1)

        self.out = torch.sigmoid(self.conv_out(self.dec_layer4))
        # print(self.enc_layer1.shape, self.enc_layer2.shape, self.enc_layer3.shape, self.enc_layer4.shape, self.bottleneck_layer.shape,
        #         self.dec_layer1.shape, self.dec_layer2.shape, self.dec_layer3.shape, self.dec_layer4.shape, self.out.shape)

        return self.out

if __name__ == '__main__':
    model = Unet()
    input_tensor = torch.zeros((1, 3, 512, 224))
    # torch.save({'epoch': 1, 'model_state_dict': model.state_dict()}, 'model.pt')
    print(model(input_tensor).shape)
    print(summary(model, (3, 512, 224)))
