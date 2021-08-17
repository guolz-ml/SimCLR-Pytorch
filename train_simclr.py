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
from torch.utils.data import DataLoader, RandomSampler
from datasets.load_open_data import *
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBE_DEVICES')
parser.add_argument('--workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--in_data', default='cifar10-animals', type=str,
                    choices=['cifar10-animals', 'cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--ood_data', default='cifar10-others', type=str,
                    choices=['cifar10-others', 'svhn', 'cifar100'], help='dataset name')
parser.add_argument('--num_labeled', type=int, default=1200,
                    help='number of labeled samples')
parser.add_argument('--num_val', type=int, default=0,
                    help='number of validation samples')
parser.add_argument('--num_ood', type=int, default=50000,
                    help='number of OOD samples')
parser.add_argument('--backbone', default='resnet18', type=str,
                    help='network architecture')
parser.add_argument('--projection_size', type=int, default=128,
                    help=' project the representation to a 128-dimensional latent space')
parser.add_argument('--optimizer', default='Adam', type=str,
                    choices=['Adam', 'SGD'])
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate in Adam')
parser.add_argument('--weight_decay', default=1e-6, type=float,
                    help='weight decay rate in Adam')
parser.add_argument('--temperature', default=0.5, type=float,
                    help='Temperature in contrastive loss')
parser.add_argument('--image_size', default=32, type=int,
                    help='resized images size')
parser.add_argument('--batch_size', default=512, type=int,
                    help='batch_size')
parser.add_argument('--epochs', default=1000, type=int,
                    help='training epochs')
parser.add_argument('--out', default='pretrain_results',
                    help='directory to output the result')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
np.random.seed(0)
torch.backends.cudnn.benchmark = True


def save_checkpoint(state, checkpoint='args.out'):
    filename = args.backbone + '_' + args.in_data + '_' + args.ood_data + '_model.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def train(model, train_loader, criterion, optimizer, scheduler=None):
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

        if epoch % 50 == 0:
            save_checkpoint(model.state_dict(), args.out)


def main():
    device = torch.device('cuda', args.gpu_id)
    args.world_size = 4
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    logger.warning(
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}")

    logger.info(dict(args._get_kwargs()))

    if args.in_data == 'cifar10-animals':
        args.num_classes = 6
        dataset = datasets.CIFAR10('./data', train=True, download=True, transform=TransformSimCLR(size=args.image_size))

        train_loader = torch.utils.data.DataLoader(
            dataset, shuffle=True, drop_last=True,
            batch_size=args.batch_size, num_workers=args.workers
            )
    else:
        if args.in_data == 'cifar10' and args.ood_data == 'svhn':
            args.num_classes = 10
            in_data = datasets.CIFAR10("./data", train=True, download=True)
            ood_data = datasets.SVHN("./data", split='train', download=True)
            ood_data.data = ood_data.data.transpose((0, 2, 3, 1))  # convert to HWC
            data = np.concatenate((in_data.data, ood_data.data[:args.num_ood]), axis=0)
            targets = np.concatenate((np.array(in_data.targets), ood_data.labels[:args.num_ood]), axis=0)

            print("# examples: {}, # OOD examples: {}".format(len(data), args.num_ood))
            train_dataset = OpensetSSL(data, targets, transform=TransformSimCLR(size=args.image_size))

            train_loader = torch.utils.data.DataLoader(
                train_dataset, sampler=RandomSampler(train_dataset),
                batch_size=args.batch_size, drop_last=True,
                num_workers=args.workers
            )
        else:
            raise NotImplementedError

    encoder = get_backbone(args.backbone, pretrained=False)
    n_features = encoder.fc.in_features
    print("n_features: ", n_features)

    model = SimCLR(encoder, args.projection_size, n_features)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = ContrastiveLoss(args.batch_size, args.temperature)
    train(model, train_loader, criterion, optimizer)


if __name__ == '__main__':
    main()
