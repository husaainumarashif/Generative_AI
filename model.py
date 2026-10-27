# the VAE model

# torch is the main deep learning library
import torch

# nn gives us layers like Linear
import torch.nn as nn

# F gives us functions like relu and loss
import torch.nn.functional as F


# our VAE model class
class VAE(nn.Module):
    # latent_dim=2 so we can plot the latent space on a 2D graph
    def __init__(self, latent_dim=2):
        # call the parent class constructor
        super().__init__()

        # encoder layer: 784 pixels -> 400 numbers
        self.fc1 = nn.Linear(784, 400)

        # layer that gives the mean (mu) of the latent distribution
        self.fc_mu = nn.Linear(400, latent_dim)

        # layer that gives the log variance of the latent distribution
        self.fc_logvar = nn.Linear(400, latent_dim)

        # decoder layer: latent values -> 400 numbers
        self.fc3 = nn.Linear(latent_dim, 400)

        # decoder layer: 400 numbers -> 784 pixels
        self.fc4 = nn.Linear(400, 784)

    # encoder function
    def encode(self, x):
        # pass the image through the first layer and apply relu
        h = F.relu(self.fc1(x))

        # return the mean and log variance
        return self.fc_mu(h), self.fc_logvar(h)

    # latent sampling function (the reparameterization trick)
    def reparameterize(self, mu, logvar):
        # convert log variance to standard deviation
        std = torch.exp(0.5 * logvar)

        # random noise with the same shape as std
        eps = torch.randn_like(std)

        # sample z = mean + noise * std
        return mu + eps * std

    # decoder function
    def decode(self, z):
        # pass z through the first decoder layer with relu
        h = F.relu(self.fc3(z))

        # sigmoid keeps every pixel value between 0 and 1
        return torch.sigmoid(self.fc4(h))

    # forward pass: encode, sample, decode
    def forward(self, x):
        # flatten the image to 784 and encode it
        mu, logvar = self.encode(x.view(-1, 784))

        # sample a latent vector z
        z = self.reparameterize(mu, logvar)

        # decode z and also return mu and logvar for the loss
        return self.decode(z), mu, logvar


# loss function = reconstruction loss + KL divergence
def loss_function(recon, x, mu, logvar):
    # reconstruction loss: how different the output is from the input
    recon_loss = F.binary_cross_entropy(recon, x.view(-1, 784), reduction="sum")

    # KL divergence: how far the latent distribution is from N(0, 1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # return total loss and both parts separately
    return recon_loss + kl, recon_loss, kl
