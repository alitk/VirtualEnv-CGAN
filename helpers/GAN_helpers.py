# The base code is adopted from here:
# https://github.com/eriklindernoren/PyTorch-GAN
# However, a lot has changed

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)



cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self,latent_dim, output_shape):
        super(Generator, self).__init__()
        self.output_shape = output_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_shape),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        #img = img.view(img.size(0), *self.output_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity





class GAN:
    def __init__(self, output_shape, n_epochs = 200, batch_size= 64, lr= 0.0002, b1= 0.5, 
                b2= 0.999, img_size= 28, latent_dim= 100, channels= 1, 
                sample_interval= 400):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.sample_interval = sample_interval
        self.channels = channels
        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()
        # Initialize generator and discriminator
        #self.img_shape = (self.channels, self.img_size, self.img_size)
        self.output_shape = output_shape  
        self.generator = Generator(latent_dim, self.output_shape)
        self.discriminator = Discriminator(self.output_shape)
              

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))




# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
    def train(self,input_data):
        dataloader = torch.utils.data.DataLoader(
            input_data,
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.n_epochs):
            for i, batch_data in enumerate(dataloader):

                # Adversarial ground truths
                valid = torch.ones([batch_data.size(0),1], dtype=torch.float, requires_grad=False)
                fake = torch.zeros([batch_data.size(0),1], dtype=torch.float, requires_grad=False)
                #valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                #fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_output = torch.tensor(batch_data)
                
                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                #z = Variable(Tensor(np.random.normal(0, 1, (self.latent_dim))))
                z = torch.randn((batch_data.size(0),self.latent_dim))
                

                # Generate a batch of images
                gen_output = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                #print(valid.type())
                #print(self.discriminator(gen_output).type())
                g_loss = self.adversarial_loss(self.discriminator(gen_output), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(real_output), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_output.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )


                batches_done = epoch * len(dataloader) + i
                if batches_done % self.sample_interval == 0:
                    save_image(gen_output.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


