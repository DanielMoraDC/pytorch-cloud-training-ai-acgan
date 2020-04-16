import argparse
import os

from google.cloud import storage

from acgan.data import fashion_mnist
from acgan.model import ACGAN

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

opt = parser.parse_args()
print(f'Run parameters: {opt}')

# Configure data loader
os.makedirs(opt.data_dir, exist_ok=True)
data_loader = fashion_mnist(opt.data_dir,
                            img_size=opt.img_size,
                            batch_size=opt.batch_size)

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

# ----------
#  Training
# ----------


model = ACGAN(data_loader,
              image_size=opt.img_size,
              n_channels=1,
              n_classes=10,
              latent_dimension=opt.latent_dim)

model.train(n_epochs=opt.n_epochs,
            logs_dir=opt.job_dir,
            lr=opt.lr,
            b1=opt.b1,
            b2=opt.b2,
            stats_interval=opt.stats_interval)

# Update models to GCP
model.save_discriminator('discriminator.pt')
upload_file('discriminator.pt')

model.save_generator('generator.pt')
upload_file('generator.pt')
