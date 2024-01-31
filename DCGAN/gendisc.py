from DCGAN.generator import Generator
from DCGAN.discriminator import Discriminator
from main.config import *
from torch import nn
# weight initialisation for netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Creating the Generator
netG = Generator(ngpu).to(device)
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, device_ids=list(range(ngpu)))
elif device.type == 'cuda' and ngpu == 1:
    netG = nn.DataParallel(netG, device_ids=[0])  # Use only one GPU
netG.apply(weights_init)
print(netG)

# Creating the Discriminator
netD = Discriminator(ngpu).to(device)
if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, device_ids=list(range(ngpu)))
elif device.type == 'cuda' and ngpu == 1:
    netD = nn.DataParallel(netD, device_ids=[0])  # Use only one GPU
netD.apply(weights_init)
print(netD)