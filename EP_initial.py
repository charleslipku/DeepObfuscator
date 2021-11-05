# v1 2019/7/16 update encoder & public classifier
# v2 2019/7/25 solve bugs in encoder & public classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import os
import argparse
import json

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# CelebA dataset
class CelebA(Dataset):
    def __init__(self, img_dir, label_root):
        self.img_dir = os.listdir(img_dir)
        self.label_root = np.load(label_root)
        self.root = img_dir

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        filename = self.img_dir[idx]
        img = Image.open(os.path.join(self.root, filename))
        label = self.label_root[int(filename[:-4])-1]
        for i in range(len(label)):
            if label[i] < 0:
                label[i] = 0
        img = preprocess(img)
        sample = {'images': img, 'labels': label}

        return sample


if __name__ == '__main__':

    #args
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-img_dir', type=str, default='/home/al380/CelebA/data/img_align_celeba/',
                        help='image dictionary path')
    parser.add_argument('-label_root', type=str, default='/home/al380/CelebA/data/labels.npy',
                        help='label root path')
    parser.add_argument('-labels', type=list, default=[20], help='label index list(default: smiling only)')
    parser.add_argument('-epoch', type=int, default=20, help='epoch number for training')

    parser.add_argument('-w', type=int, default=32, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')

    args = parser.parse_args()

    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5'

    # data loader
    print('data loading')
    celeba_dataset = CelebA(args.img_dir, args.label_root)
    train_size = int(0.8 * len(celeba_dataset))
    test_size = len(celeba_dataset) - train_size
    train_dataset, test_dataset = random_split(celeba_dataset, [train_size, test_size])
    celeba_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=args.s, num_workers=args.w, pin_memory=True)
    celeba_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=args.s, num_workers=args.w, pin_memory=True)
    print(len(celeba_dataset))
    print(len(train_dataset))
    print(len(celeba_train_loader.dataset))
    print(len(celeba_test_loader.dataset))
    print('done')

    E = torch.load('/home/al380/CelebA/output/initial/Encoder_epoch=9.pth')
    P = torch.load('/home/al380/CelebA/output/initial/Public_Classifier_epoch=9.pth')

    if args.gpu:
        E = E.cuda()
        D = P.cuda()
        E = nn.DataParallel(E, device_ids=[0])
        P = nn.DataParallel(P, device_ids=[0])

    # loss & optimizer
    loss_func = nn.BCELoss()
    P_optimizer = optim.Adam(P.parameters(), lr=args.lr)
    P_scheduler = optim.lr_scheduler.StepLR(P_optimizer, step_size=10, gamma=0.1)

    for epoch in range(args.epoch):

        # training phase
        P_scheduler.step(epoch)
        train_loss = []
        train_acc = []
        P.train()
        for batch_idx, sample in enumerate(celeba_train_loader):

            images = sample['images']
            labels = sample['labels']

            images = images.type(torch.FloatTensor)
            labels = labels[:, args.labels]
            labels = labels.type(torch.FloatTensor)

            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            P_optimizer.zero_grad()

            features, p1_idx, p2_idx = E(images)
            out = P(features)

            loss = loss_func(out, labels)
            train_loss.append(loss.cpu().data.numpy())
            out = out.cpu().data.numpy()
            labels = labels.cpu().data.numpy()
            for i in range(len(out)):
                for j in range(len(args.labels)):
                    if out[i, j] < 0.5:
                        out[i, j] = 0.
                    else:
                        out[i, j] = 1.

            train_acc.append(np.sum(labels == out) / (len(out) * len(args.labels)))

            loss.backward()
            P_optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccurcay: {:.6f}'.format(
                epoch, batch_idx * len(images), len(celeba_train_loader.dataset),
                100. * batch_idx / len(celeba_train_loader), loss.item(), train_acc[-1]))

        # testing phase
        test_loss = []
        test_acc = []
        P.eval()

        with torch.no_grad():

            for batch_idx, sample in enumerate(celeba_test_loader):

                images = sample['images']
                labels = sample['labels']

                images = images.type(torch.FloatTensor)
                labels = labels[:, args.labels]
                labels = labels.type(torch.FloatTensor)

                if args.gpu:
                    images = images.cuda()
                    labels = labels.cuda()

                features, p1_idx, p2_idx = E(images)
                out = P(features)

                loss = loss_func(out, labels)
                test_loss.append(loss.cpu().data.numpy())
                print(loss.item())
                out = out.cpu().data.numpy()
                labels = labels.cpu().data.numpy()
                for i in range(len(out)):
                    for j in range(len(args.labels)):
                        if out[i, j] < 0.5:
                            out[i, j] = 0.
                        else:
                            out[i, j] = 1.
                test_acc.append(np.sum(labels == out) / (args.b * len(args.labels)))
        with open('/home/al380/CelebA/output/initial/P_acc'+str(epoch)+'.json', 'w') as file:
            json.dump(test_acc, file)
        file.close()
        

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            np.mean(test_loss), np.mean(test_acc), len(celeba_test_loader.dataset),
            100. * np.mean(test_acc) / len(celeba_test_loader.dataset)))

        print('Epoch:', epoch, '| train loss: %.4f' % np.mean(train_loss), '| train accuracy: %.4f' % np.mean(train_acc),
              '| test loss: %.4f' % np.mean(test_loss), '| test accuracy: %.4f' % np.mean(test_acc))
        test_acc = [] 
        torch.save(P, '/home/al380/CelebA/output/initial/Privacy_Extractor_epoch='+str(epoch)+'.pth')
