import torch
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck
from utils.grl import WarmStartGradientReverseLayer
from typing import Optional, List, Dict
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class AdversarialLayer(torch.autograd.Function):
    def __init__(self, high_value=1.0):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high_value
        self.max_iter = 10000.0

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, gradOutput):
        self.coeff = np.float(
            2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                        self.high - self.low) + self.low)
        return -self.coeff * gradOutput


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 1024)
    self.ad_layer2 = nn.Linear(1024,1024)
    self.ad_layer3 = nn.Linear(1024, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.ad_layer2(x)
    x = self.ad_layer3(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1


class DANN(nn.Module):

    def __init__(self, num_class: int, bottleneck_dim: Optional[int] = -1,
                 domain_classifier_dim: Optional[int] = 1024,
                 feature_input_dim=2048, dropout_rate=0.7):
        super(DANN, self).__init__()

        self.num_class = num_class
        self.feature_dim = bottleneck_dim
        self.input_size = feature_input_dim

        self.bottleneck = nn.Sequential(
            nn.Linear(self.input_size, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
        )

        self.label_classifier = nn.Linear(self.feature_dim, self.num_class)

        self.grl = \
            WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True)

        self.backbone_bn = nn.BatchNorm1d(self.input_size)

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, domain_classifier_dim),
            nn.BatchNorm1d(domain_classifier_dim),
            nn.ReLU(inplace=True),
            nn.Linear(domain_classifier_dim, domain_classifier_dim),
            nn.BatchNorm1d(domain_classifier_dim),
            nn.ReLU(inplace=True),
            nn.Linear(domain_classifier_dim, 2)
        )

        self.mse_loss = nn.MSELoss()
        self.input_dropout = nn.Dropout(0.0)

    def calculate_label_loss(self, src_label_logits, src_label):
        label_loss = F.cross_entropy(input=src_label_logits, target=src_label)
        return label_loss

    def calculate_domain_loss(self, f_s: torch.Tensor, f_t: torch.Tensor,
                              src_domain_label: torch.Tensor,
                              tgt_domain_label: torch.Tensor):

        f = self.grl(torch.cat([f_s, f_t], dim=0))
        domain_logits = self.domain_classifier(f)
        src_domain_logits, tgt_domain_logits = domain_logits.chunk(2, dim=0)

        domain_acc = 0.5 * (binary_accuracy(output=src_domain_logits[:, 1], target=src_domain_label) +
                            binary_accuracy(output=tgt_domain_logits[:, 1], target=tgt_domain_label))

        domain_loss = 0.5 * (F.cross_entropy(input=src_domain_logits, target=src_domain_label) +
                             F.cross_entropy(input=tgt_domain_logits, target=tgt_domain_label))

        return domain_acc, domain_loss

    def inference(self, x):
        inner_code = self.bottleneck(x)
        label_logits = self.label_classifier(inner_code)

        return label_logits

    def forward(self, x, src_label):
        inner_code = self.bottleneck(x)
        label_logits = self.label_classifier(inner_code)

        f_s, f_t = inner_code.chunk(2, dim=0)
        src_label_logits, tgt_label_logits = label_logits.chunk(2, dim=0)

        return src_label_logits, f_s, f_t

    def get_parameters(self):
        params = [{"params": self.bottleneck.parameters(), "lr_mult": 1.0},
                  {"params": self.label_classifier.parameters(), "lr_mult": 1.0}]

        return params


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct




