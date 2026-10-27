# VAE Latent Space Visualizer (Task 8)

Author: Ashif Raza

A Variational Autoencoder (VAE) trained on MNIST with a 2D latent space.
The web app shows the original image, the reconstructed image, and the latent space plot.

## Files
- `model.py` : VAE (encoder, latent sampling, decoder) and loss function
- `train.py` : trains the model and saves `vae.pth`
- `app.py` : Flask backend
- `static/index.html` : frontend
- `test_vae.py` : tests (architecture, latent distribution, reconstruction loss, KL divergence)

## How to run
```
pip install -r requirements.txt
python train.py      # skip if vae.pth already exists
python app.py
```
Open http://127.0.0.1:5000

## Run tests
```
python -m pytest test_vae.py -v
```
