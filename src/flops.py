import time
import torch
import random

from torch import nn
from thop import profile
from torchstat import stat

from models import PUNet

model = PUNet()
stat(model, (6, 256, 256))
