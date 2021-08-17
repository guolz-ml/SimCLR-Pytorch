import os
import numpy as np
import torch
import torchvision
import argparse
import time
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# SimCLR
from simclr.simclr import SimCLR
from simclr.contrastive_loss import ContrastiveLoss
from simclr.augmentation_simclr import TransformSimCLR
from utils.misc import *
from models.simclr_backbone import get_backbone
from simclr.lars import LARS
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBE_DEVICES')
parser.add_argument('--workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'STL10'], help='dataset name')
parser.add_argument('--n_classes', type=int, default=10,
                    help='number of classes in the dataset')
parser.add_argument('--backbone', default='resnet18', type=str,
                    help='network architecture')
parser.add_argument('--projection_size', type=int, default=128,
                    help=' project the representation to a 128-dimensional latent space')
parser.add_argument('--optimizer', default='Adam', type=str,
                    choices=['Adam', 'LARS'])
parser.add_argument('--lr', default='3e-4', type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=1e-6, type=float,
                    help='weight decay rate in LARS')
parser.add_argument('--temperature', default=0.5, type=float,
                    help='Temperature in contrastive loss')
parser.add_argument('--image_size', default=32, type=int,
                    help='images size')
parser.add_argument('--batch_size', default=512, type=int,
                    help='batch_size')
parser.add_argument('--epochs', default=200, type=int,
                    help='training epochs')
parser.add_argument('--resume', default='pretrain_results/resnet18_cifar10_model.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
np.random.seed(0)
torch.backends.cudnn.benchmark = True


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


def inference(data_loader, simclr_model):
    feature_vector = []
    label_vector = []
    for step, (x, y) in enumerate(data_loader):
        x = x.to(args.device)

        with torch.no_grad():
            h = simclr_model(x)[0].detach()

        feature_vector.extend(h.cpu().detach().numpy())
        label_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(label_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader):
    train_x, train_y = inference(train_loader, simclr_model)
    test_x, test_y = inference(test_loader, simclr_model)
    return train_x, train_y, test_x, test_y


def train(model, optimizer, train_loader, test_loader):
    best_acc = 0.0
    test_accs = []
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        p_bar = tqdm(range(len(train_loader)))
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(args.device)
            y = y.to(args.device)

            output = model(x)
            loss = F.cross_entropy(output, y)

            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs}. Iter: {batch}/{iter}. Batch: {bt:.3f}s. Loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(train_loader),
                    bt=batch_time.avg,
                    loss=losses.avg))
            p_bar.update()
        p_bar.close()

        test_loss, test_acc = test(test_loader, model)
        best_acc = max(test_acc, best_acc)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(
            np.mean(test_accs[-10:])))


def test(test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            test_loader.set_description(
                "Test Iter: {batch:}/{iter:}. Batch: {bt:.3f}s. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


def main():
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 4
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            './data',
            train=True,
            download=True,
            transform=TransformSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            './data',
            train=False,
            download=True,
            transform=TransformSimCLR(size=args.image_size).test_transform
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,num_workers=args.workers
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, sampler=RandomSampler(test_dataset),
        batch_size=args.batch_size, num_workers=args.workers
        )
    encoder = get_backbone(args.backbone, pretrained=False)
    n_features = encoder.fc.in_features

    simclr_model = SimCLR(encoder, args.projection_size, n_features)

    assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
    print("Loading Pre-Trained Model")
    checkpoint = torch.load(args.resume, map_location=args.device.type)
    simclr_model.load_state_dict(checkpoint['state_dict'])
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()

    clf_model = LogisticRegression(simclr_model.n_features, args.n_classes).to(args.device)

    optimizer = torch.optim.Adam(clf_model.parameters(), lr=args.lr)

    print("Generate Features with Pre-Trained Model")
    train_x, train_y, test_x, test_y = get_features(simclr_model, train_loader, test_loader)

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    print("Start Fine-Tuning Process")
    train(clf_model, optimizer, train_loader, test_loader)


if __name__ == '__main__':
    main()