import argparse
import os

import torch
import numpy as np

import torch.nn.parallel
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.transforms import ResizeImage
from utils.office_home_feature_preprocessor import OfficeHomeFeature
from utils.iterator import ForeverDataIterator

from model.DANN import DANN
from utils.lr_scheduler import StepwiseLR
from utils.accuracy import accuracy
from utils.logger import get_logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_source_iterator: ForeverDataIterator, train_target_iterator: ForeverDataIterator,
          model: DANN, optimizer, lr_scheduler: StepwiseLR, args: argparse.Namespace):

    model.train()
    domain_loss_list = list()
    label_loss_list = list()

    for i in range(1, args.iters_per_epoch+1):
        optimizer.zero_grad()
        lr_scheduler.step()

        x_s, label_s = next(train_source_iterator)
        x_t, _ = next(train_target_iterator)

        x_s = torch.tensor(x_s, requires_grad=False).float().to(device)
        x_t = torch.tensor(x_t, requires_grad=False).float().to(device)
        label_s = torch.tensor(label_s, requires_grad=False).long().to(device)

        src_domain = torch.tensor(np.tile([0., 1.], [x_s.size(0), 1]), requires_grad=False).float().cuda()
        tgt_domain = torch.tensor(np.tile([1., 0.], [x_t.size(0), 1]), requires_grad=False).float().cuda()

        x = torch.cat([x_s, x_t], dim=0)

        label_loss, domain_loss, domain_acc = model.forward(x=x, src_label=label_s,
                                                            src_domain_label=src_domain,
                                                            tgt_domain_label=tgt_domain)
        total_loss = label_loss + args.trade_off * domain_loss

        domain_loss_list.append(domain_loss.item())
        label_loss_list.append(label_loss.item())

        total_loss.backward()
        optimizer.step()

    logger.info("label loss %g" % np.mean(label_loss_list))
    logger.info("domain loss %g" % np.mean(domain_loss_list))


def valid(val_loader: DataLoader, model: DANN, epoch: int):
    model.eval()

    output_list = list()
    target_list = list()
    batch_size_list = list()
    label_loss_list = list()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = torch.tensor(images, requires_grad=False).float().to(device)
            target = torch.tensor(target, requires_grad=False).long().to(device)

            label_logits = model.inference(x=images)

            label_loss = F.cross_entropy(input=label_logits, target=target)

            output_list.append(label_logits)
            target_list.append(target)
            batch_size_list.append(target.size(0))
            label_loss_list.append(label_loss)

    total_output = torch.cat(output_list, dim=0)
    total_target = torch.cat(target_list, dim=0)
    label_acc = accuracy(output=total_output, target=total_target)[0]
    logger.info("epoch: %d\tlabel acc:%f" % (epoch, label_acc))
    return label_acc


def main(args: argparse.Namespace):
    train_source_dataset = OfficeHomeFeature(root=args.root,
                                             src_domain=args.source,
                                             tgt_domain=args.source)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
    train_target_dataset = OfficeHomeFeature(root=args.root,
                                             src_domain=args.source,
                                             tgt_domain=args.target)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)

    val_dataset = OfficeHomeFeature(root=args.root, src_domain=args.source, tgt_domain=args.target)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 12, shuffle=False, num_workers=args.workers)

    train_source_iterator = ForeverDataIterator(train_source_loader)
    train_target_iterator = ForeverDataIterator(train_target_loader)

    model = DANN(num_class=train_source_dataset.num_classes, bottleneck_dim=args.feature_dim).cuda()

    optimizer = SGD(params=model.get_parameters(), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)

    lr_scheduler = StepwiseLR(optimizer=optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    best_result = 0.0

    for epoch in range(args.epochs):
        train(train_source_iterator=train_source_iterator, train_target_iterator=train_target_iterator,
              model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, args=args)
        result = valid(val_loader=val_loader, model=model, epoch=epoch)
        if result > best_result:
            best_result = result
        logger.info("best result: %g" % best_result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="pytorch dann version, ResNet50 feature input")

    parser.add_argument("--root", default="../feature_dataset/Office-Home_resnet50", type=str)
    parser.add_argument("--data", type=str, default="Office-Home")
    parser.add_argument("--result", type=str, default="result")
    parser.add_argument("--source", default="Art", help="source domain")
    parser.add_argument("--target", default="Clipart", help="target domain")
    parser.add_argument("--feature_dim", default=256, type=int, help="Feature dim")
    parser.add_argument("--workers", default=8)
    parser.add_argument("--epochs", default=50, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=0.005, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.95, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--pring_freq", default=100, type=int, help="print frequency (default: 100)")
    parser.add_argument("--seed", default=0, type=float, help="the trade-off hyper-parameter for transfer loss")
    parser.add_argument("--trade_off", default=1.0, type=float,
                        help="the trade-off hyper-parameter for transfer loss")
    parser.add_argument("--iters_per_epoch", default=100, type=int,
                        help="the iteration number of a epoch")
    parser.add_argument("--batch_size", default=512, type=int, help="batch size")
    args = parser.parse_args()
    logger = get_logger(os.path.join(args.result,
                                     "standard_dann_sgd_lr_%g_seed_%d.log" %
                                     (args.lr, args.seed)))
    torch.manual_seed(args.seed)
    logger.info(args)
    main(args)
