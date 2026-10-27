# test cases for the VAE

# torch for creating test tensors
import torch

# import the model and loss function we want to test
from model import VAE, loss_function


# test 1: check the encoder-decoder shapes
def test_encoder_decoder_architecture():
    model = VAE(latent_dim=2)
    x = torch.rand(8, 784)
    mu, logvar = model.encode(x)
    assert mu.shape == (8, 2)
    assert logvar.shape == (8, 2)
    z = model.reparameterize(mu, logvar)
    out = model.decode(z)
    assert out.shape == (8, 784)
    assert out.min() >= 0 and out.max() <= 1


# test 2: check the latent distribution
def test_latent_distribution():
    model = VAE(latent_dim=2)
    mu = torch.zeros(10000, 2)
    logvar = torch.zeros(10000, 2)
    z = model.reparameterize(mu, logvar)
    assert abs(z.mean().item()) < 0.05
    assert abs(z.std().item() - 1.0) < 0.05


# test 3: check the reconstruction loss
def test_reconstruction_loss():
    x = torch.rand(4, 784)
    mu = torch.zeros(4, 2)
    logvar = torch.zeros(4, 2)
    _, rl_bad, _ = loss_function(torch.full((4, 784), 0.5), x, mu, logvar)
    _, rl_good, _ = loss_function(x.clone(), x, mu, logvar)
    assert rl_good < rl_bad


# test 4: check the KL divergence
def test_kl_divergence():
    x = torch.rand(4, 784)
    recon = torch.rand(4, 784)
    _, _, kl_zero = loss_function(recon, x, torch.zeros(4, 2), torch.zeros(4, 2))
    assert abs(kl_zero.item()) < 1e-5
    _, _, kl_pos = loss_function(recon, x, torch.ones(4, 2) * 3, torch.zeros(4, 2))
    assert kl_pos.item() > 0
