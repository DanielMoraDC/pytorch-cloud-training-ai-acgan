import argparse
import os
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from google.cloud import storage

from acgan.discriminator import (
    Discriminator,
    initialize_weights as init_weights_d
)
from acgan.generator import (
    Generator,
    initialize_weights as init_weights_g
)
from acgan.data import fashion_mnist

# ACGAN
# Adapted from https://github.com/clvrai/ACGAN-PyTorch/blob/master/main.py

parser = argparse.ArgumentParser()
parser.add_argument("--job-dir",
                    type=str,
                    dest="job_dir",
                    default="gs://<some-bucket>",
                    help="""
                        Output path where model training files will be stored.
                        Required for Google AI Platform
                    """)
parser.add_argument("--n_epochs",
                    type=int,
                    default=25,
                    help="Number of training epochs")
parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="Size of the batches")
parser.add_argument("--lr",
                    type=float,
                    default=0.0002,
                    help="Adam learning rate")
parser.add_argument("--b1",
                    type=float,
                    default=0.5,
                    help="Adam beta1 parameter")
parser.add_argument("--b2",
                    type=float,
                    default=0.999,
                    help="Adam beta2 parameter")
parser.add_argument("--n_cpu",
                    type=int,
                    default=8,
                    help="Number of cpus used during batch generation")
parser.add_argument("--latent_dim",
                    type=int,
                    default=100,
                    help="Number of dimensions of the generator latent space")
parser.add_argument("--img_size",
                    type=int,
                    default=32,
                    help="Size of the image side for training")
parser.add_argument("--data_dir",
                    type=str,
                    default='data',
                    help="Folder where FashionMNIST data will be downloaded")
parser.add_argument("--stats_interval",
                    type=int,
                    default=100,
                    help="Interval between Tensorboard metrics tracking")
parser.add_argument("--sample_interval",
                    type=int,
                    default=400,
                    help="Interval between image sampling during training")
opt = parser.parse_args()
print(f'Run parameters: {opt}')

n_classes = 10
n_channels = 1

device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator(n_classes=n_classes,
                      n_channels=1,
                      img_size=opt.img_size,
                      latent_dimensions=opt.latent_dim,
                      base_feature_maps=64)

discriminator = Discriminator(n_classes=n_classes,
                              n_channels=1,
                              img_size=opt.img_size,
                              base_feature_maps=16)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(init_weights_g)
discriminator.apply(init_weights_d)

# Configure data loader
os.makedirs(opt.data_dir, exist_ok=True)
dataloader = fashion_mnist(opt.data_dir,
                           img_size=opt.img_size,
                           batch_size=opt.batch_size)
batches_per_epoch = len(dataloader)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr=opt.lr,
                               betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=opt.lr,
                               betas=(opt.b1, opt.b2))

# ------------
# Google Cloud storage configuration
# -------------

storage_client = storage.Client()
bucket_name = opt.job_dir[:-1] if opt.job_dir.endswith('/') else opt.job_dir
bucket_name = bucket_name.replace('gs://', '')
bucket = storage_client.bucket(bucket_name)


def upload_file(src: str):
    blob = bucket.blob(os.path.basename(src))
    blob.upload_from_filename(src)

# ------------
# Tensorboard configuration
# ------------


writer = SummaryWriter(opt.job_dir)

# ----------
#  Training
# ----------


def sample_image(writer: SummaryWriter,
                 samples_per_class: int,
                 iterations: int):
    z = torch.randn(samples_per_class * n_classes, opt.latent_dim).to(device)
    labels = np.array([
        num for num in range(samples_per_class) for _ in range(n_classes)])
    labels = torch.Tensor(labels).to(device)
    gen_imgs = generator(z, labels)
    writer.add_images('gan_grid', gen_imgs, iterations)


for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        iteration = epoch * batches_per_epoch + i
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = torch.full(size=(batch_size, 1), fill_value=1.0).to(device)
        fake = torch.full(size=(batch_size, 1), fill_value=0.0).to(device)

        # Configure batch data
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, opt.latent_dim).to(device)
        gen_labels = torch.randint(
            low=0, high=n_classes, size=(batch_size,)).to(device)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        if iteration % opt.stats_interval == 0:
            writer.add_scalar('Discriminator loss', d_loss.item(), iteration)
            writer.add_scalar(
                'Discriminator real label loss', d_real_loss.item(), iteration)
            writer.add_scalar(
                'Discriminator fake label loss', d_real_loss.item(), iteration)
            writer.add_scalar('Generator loss', g_loss.item(), iteration)
            writer.add_scalar(
                'Discriminator accuracy', d_acc.item(), iteration)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )

        if iteration % opt.sample_interval == 0:
            sample_image(writer, samples_per_class=10, iterations=iteration)

# Save resulting model, locally
torch.save(discriminator, 'discriminator.pt')
torch.save(generator,  'generator.pt')

# Update results to GCP
upload_file('discriminator.pt')
upload_file('generator.pt')

writer.close()
