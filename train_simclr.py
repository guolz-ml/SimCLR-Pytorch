import os
import numpy as np
import torch
import torchvision
import argparse
import time
from tqdm import tqdm

# SimCLR
from simclr.simclr import SimCLR
from simclr.contrastive_loss import ContrastiveLoss
from simclr.augmentation_simclr import TransformSimCLR
from utils.misc import *
from models.simclr_backbone import get_backbone
from simclr.lars import LARS
from torch.utils.data import DataLoader, RandomSampler


parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBE_DEVICES')
parser.add_argument('--workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'STL10'], help='dataset name')
parser.add_argument('--backbone', default='resnet18', type=str,
                    help='network architecture')
parser.add_argument('--projection_size', type=int, default=64,
                    help=' project the representation to a 128-dimensional latent space')
parser.add_argument('--optimizer', default='Adam', type=str,
                    choices=['Adam', 'LARS'])
parser.add_argument('--weight_decay', default=1e-6, type=float,
                    help='weight decay rate in LARS')
parser.add_argument('--temperature', default=0.5, type=float,
                    help='Temperature in contrastive loss')
parser.add_argument('--image_size', default=224, type=int,
                    help='images size')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch_size')
parser.add_argument('--epochs', default=100, type=int,
                    help='training epochs')
parser.add_argument('--out', default='pretrain_results',
                    help='directory to output the result')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
np.random.seed(0)
torch.backends.cudnn.benchmark = True


def save_checkpoint(state, checkpoint='args.out'):
    filename = args.backbone + '_' + args.dataset + '_model.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def load_optimizer(args, model):
    scheduler = None
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    elif args.optimizer == 'LARS':
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(model.parameters(), lr=learning_rate,
                         weight_decay=args.weight_decay,
                         exclude_from_weight_decay=['batch_normalization', 'bias'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def train(model, train_loader, criterion, optimizer, scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        p_bar = tqdm(range(len(train_loader)))
        for batch_idx, ((x_i, x_j), _) in enumerate(train_loader):
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)

            loss = criterion(z_i, z_j)

            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

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

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }, args.out)

def main():
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 4
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            './data',
            download=True,
            transform=TransformSimCLR(size=args.image_size)
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size, drop_last=True,
        num_workers=args.workers
        )

    encoder = get_backbone(args.backbone, pretrained=False)
    n_features = encoder.fc.in_features

    model = SimCLR(encoder, args.projection_size, n_features)
    model = model.to(args.device)

    optimizer, scheduler = load_optimizer(args, model)
    criterion = ContrastiveLoss(args.batch_size, args.temperature)
    train(model, train_loader, criterion, optimizer, scheduler)


if __name__ == '__main__':
    main()