# based on vgg16

import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool1 = nn.Upsample(size=(13, 11), mode='bilinear')

        self.reconstruct1 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1)
        )

        # self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool2 = nn.Upsample(size=(27, 22), mode='bilinear')

        self.reconstruct2 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        )

        # self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool3 = nn.Upsample(size=(54, 44), mode='bilinear')

        self.reconstruct3 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
        )

        # self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool4 = nn.Upsample(size=(109, 89), mode='bilinear')

        self.reconstruct4 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
        )

        # self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool5 = nn.Upsample(size=(218, 178), mode='bilinear')

        self.reconstruct5 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, batch_size, p1_idx, p2_idx):#, p3_idx, p4_idx, p5_idx):
        # x = self.unpool1(x, p5_idx, output_size=(batch_size, 512, 13, 11))
        # x = self.unpool1(x)
        # x = self.reconstruct1(x)
        # # x = self.unpool2(x, p4_idx, output_size=(batch_size, 512, 27, 22))
        # x = self.unpool2(x)
        # x = self.reconstruct2(x)
        # # x = self.unpool3(x, p3_idx, output_size=(batch_size, 256, 54, 44))
        # x = self.unpool3(x)
        # x = self.reconstruct3(x)
        # x = self.unpool4(x, p2_idx, output_size=(batch_size, 128, 109, 89))
        x = self.unpool4(x)
        x = self.reconstruct4(x)
        # x = self.unpool5(x, p1_idx, output_size=(batch_size, 64, 218, 178))
        x = self.unpool5(x)
        x = self.reconstruct5(x)

        return x
