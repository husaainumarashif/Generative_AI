# Project: VAE Latent Space Visualizer (Task 8)
# Author: Ashif Raza

# io lets us work with data in memory like a file
import io

# base64 converts image bytes to text for the browser
import base64

# numpy for array work
import numpy as np

# torch to load the model
import torch

# Flask for the web server
from flask import Flask, request, jsonify

# PIL to open and resize images
from PIL import Image

# MNIST dataset for test images
from torchvision import datasets, transforms

# our VAE class from model.py
from model import VAE

# create the Flask app and tell it where the static folder is
app = Flask(__name__, static_folder="static", static_url_path="")

# create the model with a 2D latent space
model = VAE(latent_dim=2)

# load the weights we saved in train.py
model.load_state_dict(torch.load("vae.pth", map_location="cpu"))

# switch to evaluation mode
model.eval()

# load the MNIST test images (downloads the first time)
test_data = datasets.MNIST("data", train=False, download=True,
                           transform=transforms.ToTensor())


# helper: turn a tensor into a base64 image the browser can show
def tensor_to_b64(t):
    # reshape to 28x28, scale to 0-255 and convert to integers
    arr = (t.view(28, 28).detach().numpy() * 255).astype(np.uint8)

    # make an image and enlarge it to 140x140 so it is easier to see
    img = Image.fromarray(arr).resize((140, 140), Image.NEAREST)

    # create an in-memory buffer
    buf = io.BytesIO()

    # save the image into the buffer as PNG
    img.save(buf, format="PNG")

    # convert to base64 text and add the data prefix
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# route for the home page
@app.route("/")
def home():
    # send the index.html file
    return app.send_static_file("index.html")


# route that returns the latent space points
@app.route("/latent")
def latent():
    # take the first 2000 test images and stack them in one tensor
    xs = torch.stack([test_data[i][0] for i in range(2000)])

    # get the digit label for each image
    labels = [int(test_data[i][1]) for i in range(2000)]

    # no gradients needed here
    with torch.no_grad():
        # encode the images, we only need the mean
        mu, _ = model.encode(xs.view(-1, 784))

    # send points and labels as JSON
    return jsonify({"points": mu.tolist(), "labels": labels})


# route that reconstructs an image
@app.route("/reconstruct", methods=["POST"])
def reconstruct():
    # check if the user uploaded an image
    if "image" in request.files:
        # open it, make it grayscale and resize to 28x28
        img = Image.open(request.files["image"]).convert("L").resize((28, 28))

        # convert to a tensor with values between 0 and 1
        x = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    else:
        # otherwise pick a random test image
        idx = np.random.randint(len(test_data))

        # take only the image, not the label
        x = test_data[idx][0]

    # no gradients needed
    with torch.no_grad():
        # encode the image
        mu, logvar = model.encode(x.view(-1, 784))

        # decode using the mean for a clean reconstruction
        recon = model.decode(mu)

    # send original, reconstructed and latent position back
    return jsonify({
        "original": tensor_to_b64(x),
        "reconstructed": tensor_to_b64(recon),
        "z": mu[0].tolist(),
    })


# run the server only when this file is run directly
if __name__ == "__main__":
    # start Flask on port 5000
    app.run(port=5000)
