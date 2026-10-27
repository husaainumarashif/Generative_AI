# train the VAE on MNIST

# torch for training
import torch

# datasets and transforms to load MNIST
from torchvision import datasets, transforms

# DataLoader gives us data in small batches
from torch.utils.data import DataLoader

# import our model and loss from model.py
from model import VAE, loss_function

# use the GPU if available, otherwise use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# download MNIST training data and convert images to tensors
train_data = datasets.MNIST("data", train=True, download=True,
                            transform=transforms.ToTensor())

# split the data into batches of 128 and shuffle it
loader = DataLoader(train_data, batch_size=128, shuffle=True)

# create the model and move it to the device
model = VAE(latent_dim=2).to(device)

# Adam optimizer with learning rate 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# number of times we go through the whole dataset
epochs = 10

# loop over each epoch
for epoch in range(epochs):
    # variables to keep track of the losses
    total, total_recon, total_kl = 0, 0, 0

    # loop over each batch (we ignore the labels with _)
    for x, _ in loader:
        # move the batch to the device
        x = x.to(device)

        # forward pass through the VAE
        recon, mu, logvar = model(x)

        # calculate the loss
        loss, rl, kl = loss_function(recon, x, mu, logvar)

        # clear old gradients
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # update the weights
        optimizer.step()

        # add this batch loss to the totals
        total += loss.item()
        total_recon += rl.item()
        total_kl += kl.item()

    # number of images in the dataset
    n = len(loader.dataset)

    # print the average losses for this epoch
    print(f"Epoch {epoch+1}/{epochs}  loss={total/n:.2f}  "
          f"recon={total_recon/n:.2f}  kl={total_kl/n:.2f}")

# save the trained weights to a file
torch.save(model.state_dict(), "vae.pth")

# tell the user we are done
print("saved vae.pth")
