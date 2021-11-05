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

import MS_SSIM_loss

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
        noise = Image.open('/home/al380/CelebA/data/noise.jpg')
      #  noise = Image.open('/home/al380/CelebA/data/img_blurred/'+filename)
        noise = preprocess(noise)
        sample = {'images': img, 'labels': label, 'noise': noise}

        return sample


if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-img_dir', type=str, default='/home/al380/CelebA/data/img_align_celeba/',
                        help='image dictionary path')
    parser.add_argument('-label_root', type=str, default='/home/al380/CelebA/data/labels.npy',
                        help='label root path')
    parser.add_argument('-E_Model',type=str, default='/home/al380/CelebA/output/initial/Encoder_epoch=9.pth')
    parser.add_argument('-C_Model', type=str, default='/home/al380/CelebA/output/initial/Public_Classifier_epoch=9.pth')
    parser.add_argument('-R_Model', type=str, default='/home/al380/CelebA/output/initial/Reconstruct_Decoder_epoch=19.pth')
    parser.add_argument('-P_Model', type=str, default='/home/al380/CelebA/output/initial/Privacy_Extractor_epoch=0.pth')
    parser.add_argument('-labels', type=list, default=[31], help='label index list(default: smiling only)')
    parser.add_argument('-epoch', type=int, default=60, help='epoch number for training')
    parser.add_argument('-self_lambda', type=float, default=0.5)

    parser.add_argument('-w', type=int, default=32, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')

    args = parser.parse_args()

    # data loader
    print('data loading')
    celeba_dataset = CelebA(args.img_dir, args.label_root)
    train_size = int(0.8 * len(celeba_dataset))
    test_size = len(celeba_dataset) - train_size
    train_dataset, test_dataset = random_split(celeba_dataset, [train_size, test_size])
    celeba_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=args.s, num_workers=args.w, pin_memory=True)
    celeba_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=args.s, num_workers=args.w, pin_memory=True)
    print('done')

    E = torch.load(args.E_Model)
    C = torch.load(args.C_Model)
    R = torch.load(args.R_Model)
    P = torch.load(args.P_Model)

    if args.gpu:
        E = E.cuda()
        C = C.cuda()
        R = R.cuda()
        P = P.cuda()
        E = nn.DataParallel(E, device_ids=[0])
        C = nn.DataParallel(C, device_ids=[0])
        R = nn.DataParallel(R, device_ids=[0])
        P = nn.DataParallel(P, device_ids=[0])

    # loss & optimizer
    BCE = nn.BCELoss()
    MS_SSIM = MS_SSIM_loss.MS_SSIM(max_val=1)
    MSE = nn.MSELoss()
    E_optimizer = optim.Adam(E.parameters(), lr=args.lr)
    C_optimizer = optim.Adam(C.parameters(), lr=args.lr)
    R_optimizer = optim.Adam(R.parameters(), lr=args.lr)
    P_optimizer = optim.Adam(P.parameters(), lr=args.lr)

    E_scheduler = optim.lr_scheduler.StepLR(E_optimizer, step_size=30, gamma=0.1)
    C_scheduler = optim.lr_scheduler.StepLR(C_optimizer, step_size=30, gamma=0.1)
    R_scheduler = optim.lr_scheduler.StepLR(R_optimizer, step_size=30, gamma=0.1)
    P_scheduler = optim.lr_scheduler.StepLR(P_optimizer, step_size=30, gamma=0.1)

    C_loss_logger = []
    R1_loss_logger = []
    R2_loss_logger = []
    P_loss_logger = []
    C_train_acc = []
    P_train_acc = []
    C_test_acc = []
    P_test_acc = []

    for epoch in range(args.epoch):

        # training phase
        E_scheduler.step(epoch)
        C_scheduler.step(epoch)
        R_scheduler.step(epoch)
        P_scheduler.step(epoch)

        E.train()
        C.train()
        R.train()
        P.train()
        
        par = 10
        if epoch > 10:
            par = 20 - epoch
        print('par:',par)

        for batch_idx, sample in enumerate(celeba_train_loader):

            images = sample['images']
            labels = sample['labels']
            noise = sample['noise']
            # noise = 1 - noise

            images = images.type(torch.FloatTensor)
            smiling = labels[:, args.labels]
            smiling = smiling.type(torch.FloatTensor)
            gender = labels[:, [20]]
            gender = gender.type(torch.FloatTensor)


            if args.gpu:
                noise = noise.cuda()
                images = images.cuda()
                smiling = smiling.cuda()
                gender = gender.cuda()

            E_optimizer.zero_grad()
            C_optimizer.zero_grad()
            R_optimizer.zero_grad()
            P_optimizer.zero_grad()

            features, p1_idx, p2_idx = E(images)
            out = C(features)
            reimg = R(features, args.b, p1_idx, p2_idx)
            privacy = P(features)
            
            C_loss = BCE(out, smiling)
            R1_loss = 1 - MS_SSIM(reimg, images)
            R2_loss = 1 - MS_SSIM(reimg, noise)
            R3_loss = MSE(reimg, images)
            P_loss = BCE(privacy, gender)
            if R1_loss.item() != R1_loss.item():
                R1_loss = R3_loss

            if batch_idx % 4 == 0:
                E_loss = C_loss + par * R2_loss
                E_loss.backward()
                E_optimizer.step()

            elif batch_idx % 4 == 1:
                E_loss = C_loss - par * R1_loss
                E_loss.backward()
                E_optimizer.step()

            # elif batch_idx % 5 == 2:
            #     E_loss = C_loss - par * P_loss
            #     E_loss.backward()
            #     E_optimizer.step()

            elif batch_idx % 4 == 2:
                R1_loss.backward(retain_graph=True)
                R_optimizer.step()
                P_loss.backward()
                P_optimizer.step()

            else:
                C_loss.backward()
                C_optimizer.step()

            
            C_loss_logger.append(C_loss.item())
            R1_loss_logger.append(R1_loss.item())
            R2_loss_logger.append(R2_loss.item())
            P_loss_logger.append(P_loss.item())

            out = out.cpu().data.numpy()
            privacy = privacy.cpu().data.numpy()

            smiling = smiling.cpu().data.numpy()
            gender = gender.cpu().data.numpy()

            for i in range(len(out)):
                for j in range(len(args.labels)):
                    if out[i, j] < 0.5:
                        out[i, j] = 0.
                    else:
                        out[i, j] = 1.

            for i in range(len(privacy)):
                for j in range(len(args.labels)):
                    if privacy[i, j] < 0.5:
                        privacy[i, j] = 0.
                    else:
                        privacy[i, j] = 1.

            C_acc = np.sum(smiling == out) / (len(out) * len(args.labels))
            P_acc = np.sum(gender == privacy) / (len(out) * len(args.labels))

            C_train_acc.append(C_acc)
            P_train_acc.append(P_acc)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tC Loss: {:.6f}\tR1 Loss: {:.6f}\t{:.6f}\tR2 Loss: {:.6f}\tP Loss: {:.6f}'
                  '\tC Accuracy: {:.6f}\tP Accuracy: {:.6f}'.format(
                epoch, batch_idx * len(images), len(celeba_train_loader.dataset),
                100. * batch_idx / len(celeba_train_loader), C_loss.item(), R1_loss.item(),R3_loss.item(), R2_loss.item(), P_loss.item(), C_acc, P_acc))

        # save loss
        with open('/home/al380/CelebA/output/loss_R_only/C_loss'+str(epoch)+'.json', 'w') as file:
            json.dump(C_loss_logger, file)
        file.close()

        with open('/home/al380/CelebA/output/loss_R_only/R1_loss' + str(epoch) + '.json', 'w') as file:
            json.dump(R1_loss_logger, file)
        file.close()

        with open('/home/al380/CelebA/output/loss_R_only/R2_loss' + str(epoch) + '.json', 'w') as file:
            json.dump(R2_loss_logger, file)
        file.close()

        with open('/home/al380/CelebA/output/loss_R_only/P_loss' + str(epoch) + '.json', 'w') as file:
            json.dump(P_loss_logger, file)
        file.close()

        #save acc
        with open('/home/al380/CelebA/output/loss_R_only/C_train_acc'+str(epoch)+'.json', 'w') as file:
            json.dump(C_train_acc, file)
        file.close()
        C_train_acc = []

        with open('/home/al380/CelebA/output/loss_R_only/P_train_acc'+str(epoch)+'.json', 'w') as file:
            json.dump(P_train_acc, file)
        file.close()
        P_train_acc = []

        # save models
        torch.save(E, '/home/al380/CelebA/output/adv_R_only/Encoder_epoch=' + str(epoch) + '.pth')
        torch.save(C, '/home/al380/CelebA/output/adv_R_only/Public_Classifier_epoch=' + str(epoch) + '.pth')
        torch.save(R, '/home/al380/CelebA/output/adv_R_only/Reconstruct_Decoder_epoch=' + str(epoch) + '.pth')
        torch.save(P, '/home/al380/CelebA/output/adv_R_only/Privacy_Extractor_epoch=' + str(epoch) + '.pth')

        # testing phase
        test_loss = []
        test_acc = []
        E.eval()
        C.eval()
        R.eval()
        P.eval()

        with torch.no_grad():

            for batch_idx, sample in enumerate(celeba_test_loader):

                images = sample['images']
                labels = sample['labels']
                noise = sample['noise']
                # noise = 1 - noise

                images = images.type(torch.FloatTensor)
                smiling = labels[:, args.labels]
                smiling = smiling.type(torch.FloatTensor)
                gender = labels[:, [20]]
                gender = gender.type(torch.FloatTensor)

                if args.gpu:
                    noise = noise.cuda()
                    images = images.cuda()
                    smiling = smiling.cuda()
                    gender = gender.cuda()

                E_optimizer.zero_grad()
                C_optimizer.zero_grad()
                R_optimizer.zero_grad()
                P_optimizer.zero_grad()

                features, p1_idx, p2_idx = E(images)
                out = C(features)
                reimg = R(features, args.b, p1_idx, p2_idx)
                privacy = P(features)

                C_loss = BCE(out, smiling)
                R1_loss = 1 - MS_SSIM(reimg, images)
                R2_loss = 1 - MS_SSIM(reimg, noise)
                R3_loss = MSE(reimg, images)
                P_loss = BCE(privacy, gender)

                out = out.cpu().data.numpy()
                privacy = privacy.cpu().data.numpy()

                smiling = smiling.cpu().data.numpy()
                gender = gender.cpu().data.numpy()

                for i in range(len(out)):
                    for j in range(len(args.labels)):
                        if out[i, j] < 0.5:
                            out[i, j] = 0.
                        else:
                            out[i, j] = 1.

                for i in range(len(privacy)):
                    for j in range(len(args.labels)):
                        if privacy[i, j] < 0.5:
                            privacy[i, j] = 0.
                        else:
                            privacy[i, j] = 1.

                C_acc = np.sum(smiling == out) / (len(out) * len(args.labels))
                P_acc = np.sum(gender == privacy) / (len(out) * len(args.labels))

                C_test_acc.append(C_acc)
                P_test_acc.append(P_acc)

                print(
                    'Test Epoch: {} [{}/{} ({:.0f}%)]\tC Loss: {:.6f}\tR1 Loss: {:.6f}\t{:.6f}\tR2 Loss: {:.6f}\tP Loss: {:.6f}'
                    '\tC Accurcay: {:.6f}\tP Accurcay: {:.6f}'.format(
                        epoch, batch_idx * len(images), len(celeba_test_loader.dataset),
                               100. * batch_idx / len(celeba_test_loader), C_loss.item(), R1_loss.item(), R3_loss.item(),
                        R2_loss.item(), P_loss.item(), C_acc, P_acc))

            # save acc
            with open('/home/al380/CelebA/output/loss_R_only/C_test_acc' + str(epoch) + '.json', 'w') as file:
                json.dump(C_test_acc, file)
            file.close()
            C_test_acc = []

            with open('/home/al380/CelebA/output/loss_R_only/P_test_acc' + str(epoch) + '.json', 'w') as file:
                json.dump(P_test_acc, file)
            file.close()
            P_test_acc = []
