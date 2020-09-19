import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import numpy as np

import torch.nn.parallel
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.office_home_feature_preprocessor_4_multisource import OfficeHomeFeature
from utils.iterator import ForeverDataIterator

from model.DANN import DANN, AdversarialNetwork, AdversarialLayer
from model.loss import PADA
from utils.lr_scheduler import StepwiseLR
from utils.accuracy import accuracy
from utils.logger import get_logger


def train(train_source_iterator: ForeverDataIterator, train_target_iterator: ForeverDataIterator,
          model: DANN, ad_net: AdversarialNetwork, grl: AdversarialLayer, optimizer: SGD,
          lr_scheduler: StepwiseLR, args: argparse.Namespace, criterion, class_weight):

    model.train()
    ad_net.train()
    domain_loss_list = list()
    label_loss_list = list()

    for i in range(args.iters_per_epoch):
        optimizer.zero_grad()
        lr_scheduler.step()

        x_s, label_s = next(train_source_iterator)
        x_t, _ = next(train_target_iterator)

        x_s = torch.tensor(x_s, requires_grad=False).float().cuda()
        x_t = torch.tensor(x_t, requires_grad=False).float().cuda()
        label_s = torch.tensor(label_s, requires_grad=False).long().cuda()

        # src_domain = torch.tensor(np.tile([0., 1.], [x_s.size(0), 1]), requires_grad=False).float().cuda()
        # tgt_domain = torch.tensor(np.tile([1., 0.], [x_t.size(0), 1]), requires_grad=False).float().cuda()

        x = torch.cat([x_s, x_t], dim=0)

        src_label_logits, f_s, f_t = model(x, label_s)
        weight_ad = torch.zeros(x.size(0))
        label_numpy = label_s.data.cpu().numpy()
        for j in range(int(x.size(0) / 2)):
            weight_ad[j] = class_weight[int(label_numpy[j])]

        weight_ad = weight_ad / torch.max(weight_ad[0:int(x.size(0) / 2)])
        for j in range(int(x.size(0) / 2), x.size(0)):
            weight_ad[j] = 1.0
        transfer_loss = PADA(features=torch.cat([f_s, f_t], dim=0), ad_net=ad_net, grl_layer=grl,
                             weight_ad=weight_ad)
        label_loss = criterion(src_label_logits, label_s)

        domain_loss_list.append(transfer_loss.item())
        label_loss_list.append(label_loss.item())

        total_loss = label_loss + transfer_loss
        total_loss.backward()
        optimizer.step()

    # print("label loss", np.mean(label_loss_list))
    # print("domain loss", np.mean(domain_loss_list))
    logger.info("label loss: %g" % np.mean(label_loss_list))
    logger.info("domain loss: %g" % np.mean(domain_loss_list))


def calculate_weight_ad(val_loader: DataLoader, model: DANN):
    model.eval()
    start_test = True
    with torch.no_grad():
        for x, label in val_loader:
            x = torch.tensor(x, requires_grad=False).float().cuda()
            logits = model.inference(x)
            softmax_outputs = torch.nn.Softmax(dim=1)(logits)
            if start_test:
                all_softmax_output = softmax_outputs.data.cpu().float()
                start_test = False
            else:
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
    return all_softmax_output


def valid(val_loader: DataLoader, model: DANN, epoch: int):
    model.eval()

    output_list = list()
    target_list = list()
    batch_size_list = list()

    with torch.no_grad():
        with torch.no_grad():
            for x, label in val_loader:
                x = torch.tensor(x, requires_grad=False).float().cuda()
                label = label.cuda()
                logits = model.inference(x)

                output_list.append(logits)
                target_list.append(label)
                batch_size_list.append(label.size(0))

    total_output = torch.cat(output_list, dim=0)
    total_target = torch.cat(target_list, dim=0)
    label_acc = accuracy(output=total_output, target=total_target)[0]
    # print("epoch: %d\tlabel acc:%f" % (epoch, label_acc))
    logger.info("epoch: %d\tlabel acc:%f" % (epoch, label_acc))


def main(args: argparse.Namespace):
    tgt_remove_list = open("utils/remove_label_record.txt", mode="r").readlines()[args.remove_list_idx].strip().split("|")
    tgt_remove_list = list(map(int, tgt_remove_list))

    train_source_dataset = OfficeHomeFeature(root=args.root, task=args.target, is_source=True, remove_list=[])
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
    train_target_dataset = OfficeHomeFeature(root=args.root, task=args.target, is_source=False,
                                             remove_list=tgt_remove_list)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)

    val_datatset = OfficeHomeFeature(root=args.root, task=args.target, is_source=False,
                                     remove_list=tgt_remove_list)
    val_loader = DataLoader(val_datatset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, drop_last=True)

    train_source_iterator = ForeverDataIterator(train_source_loader)
    train_target_iterator = ForeverDataIterator(train_target_loader)

    model = DANN(num_class=train_source_dataset.num_classes, bottleneck_dim=args.feature_dim).cuda()
    ad_net = AdversarialNetwork(in_feature=args.feature_dim).cuda()
    gradient_reverse_layer = AdversarialLayer()

    param_list = model.get_parameters() + [{"params": ad_net.parameters(), "lr_mult": 1.0}]

    optimizer = SGD(params=param_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = StepwiseLR(optimizer=optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    for epoch in range(args.epochs):
        target_fc8_out = calculate_weight_ad(val_loader=val_loader, model=model)
        class_weight = torch.mean(target_fc8_out, 0)
        class_weight = (class_weight / torch.mean(class_weight)).cuda().view(-1)
        class_criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

        train(train_source_iterator=train_source_iterator, train_target_iterator=train_target_iterator,
              model=model, ad_net=ad_net, grl=gradient_reverse_layer, optimizer=optimizer,
              lr_scheduler=lr_scheduler, args=args, criterion=class_criterion, class_weight=class_weight)
        valid(val_loader=val_loader, epoch=epoch, model=model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pytorch dann version, ResNet50 feature input")

    parser.add_argument("--root", default="feature_dataset/Office-Home_resnet50", type=str)
    parser.add_argument("--data", type=str, default="Office-Home")
    parser.add_argument("--result", type=str, default="result")
    parser.add_argument("--source", default="Art", help="source domain")
    parser.add_argument("--target", default="Clipart", help="target domain")
    parser.add_argument("--feature_dim", default=256, type=int, help="Feature dim")
    parser.add_argument("--remove_list_idx", default=0, type=int)
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
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    args = parser.parse_args()
    logger = get_logger(os.path.join(args.result,
                                     "pada_tgt_%s_random_seed_%d_drop_idx_%d.log" %
                                     (args.target, args.seed, args.remove_list_idx)))
    torch.manual_seed(args.seed)
    logger.info(args)
    main(args)

