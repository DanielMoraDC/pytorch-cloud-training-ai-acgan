from torch.utils.data import DataLoader

import numpy as np

import torch

from torch.utils.tensorboard import SummaryWriter

from discriminator import (
    Discriminator,
    initialize_weights as init_weights_d
)

from generator import (
    Generator,
    initialize_weights as init_weights_g
)


DEFAULT_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
CUDA_ENABLED = True if torch.cuda.is_available() else False


class ACGAN(object):

    # Adapted from https://github.com/clvrai/ACGAN-PyTorch/blob/master/main.py

    def __init__(self,
                 dataset: DataLoader,
                 image_size: int,
                 n_channels: int,
                 n_classes: int,
                 latent_dimension: int,
                 g_base_maps: int = 64,
                 d_base_maps: int = 64):
        self._data_loader = dataset
        self._image_size = image_size
        self._n_channels = n_channels
        self._n_classes = n_classes
        self._latent_dimension = latent_dimension
        self._g_base_maps = g_base_maps
        self._d_base_maps = d_base_maps

        self._discriminator = self._build_discriminator()
        self._generator = self._build_generator()

    def _build_generator(self):
        g = Generator(n_classes=self._n_classes,
                      n_channels=self._n_channels,
                      img_size=self._image_size,
                      latent_dimensions=self._latent_dimension,
                      base_feature_maps=self._g_base_maps)

        if CUDA_ENABLED:
            g.cuda()

        return g

    def _build_discriminator(self):
        d = Discriminator(n_classes=self._n_classes,
                          n_channels=self._n_channels,
                          img_size=self._image_size,
                          base_feature_maps=self._d_base_maps)
        if CUDA_ENABLED:
            d.cuda()

        return d

    @staticmethod
    def _losses():
        adversarial_loss = torch.nn.BCELoss()
        auxiliary_loss = torch.nn.CrossEntropyLoss()

        if CUDA_ENABLED:
            adversarial_loss.cuda()
            auxiliary_loss.cuda()

        return adversarial_loss, auxiliary_loss

    def _optimizers(self, lr: float, b1: float, b2: float):
        optimizer_g = torch.optim.Adam(self._generator.parameters(),
                                       lr=lr,
                                       betas=(b1, b2))
        optimizer_d = torch.optim.Adam(self._discriminator.parameters(),
                                       lr=lr,
                                       betas=(b1, b2))
        return optimizer_g, optimizer_d

    def _train_generator(self,
                         batch_size,
                         optimizer,
                         adversarial_loss,
                         auxiliary_loss):
        # Make sure we do not accumulate gradients
        optimizer.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, self._latent_dimension).to(DEFAULT_DEVICE)
        generated_labels = torch.randint(
            low=0, high=self._n_classes, size=(batch_size,)).to(DEFAULT_DEVICE)

        # Generate a batch of images
        generated_images = self._generator(z, generated_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = self._discriminator(generated_images)
        valid = torch.full(
            size=(batch_size, 1), fill_value=1.0).to(DEFAULT_DEVICE)
        loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, generated_labels))  # noqa

        # Compute loss and apply gradients
        loss.backward()
        optimizer.step()

        return loss, generated_images, generated_labels

    def _train_discriminator(self,
                             images,
                             labels,
                             generated_images,
                             generated_labels,
                             optimizer,
                             adversarial_loss,
                             auxiliary_loss):
        batch_size = images.shape[0]
        optimizer.zero_grad()

        # Loss for real images
        real_pred, real_class = self._discriminator(images)
        valid = torch.full(
            size=(batch_size, 1), fill_value=1.0).to(DEFAULT_DEVICE)
        real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_class, labels)) / 2  # noqa

        # Loss for fake images
        fake_pred, fake_class = self._discriminator(generated_images.detach())
        fake = torch.full(
            size=(batch_size, 1), fill_value=0.0).to(DEFAULT_DEVICE)
        fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_class, generated_labels)) / 2  # noqa

        # Total discriminator loss
        loss = (real_loss + fake_loss) / 2

        # Calculate discriminator accuracy
        predicted = np.concatenate(
            [real_class.data.cpu().numpy(), fake_class.data.cpu().numpy()],
            axis=0
        )
        groundtruth = np.concatenate(
            [labels.data.cpu().numpy(), generated_labels.data.cpu().numpy()],
            axis=0
        )
        accuracy = np.mean(np.argmax(predicted, axis=1) == groundtruth)

        # Compute loss and apply gradients
        loss.backward()
        optimizer.step()

        return loss, real_loss, fake_loss, accuracy

    @staticmethod
    def _track_metrics(writer,
                       g_loss,
                       d_loss,
                       d_real_loss,
                       d_fake_loss,
                       acc,
                       iteration):
        # Write scalar stats
        writer.add_scalar('Discriminator loss', d_loss.item(), iteration)
        writer.add_scalar(
            'Discriminator real label loss', d_real_loss.item(), iteration)
        writer.add_scalar(
            'Discriminator fake label loss', d_fake_loss.item(), iteration)
        writer.add_scalar('Generator loss', g_loss.item(), iteration)
        writer.add_scalar(
            'Discriminator accuracy', acc.item(), iteration)

    def _track_generated_images(self, writer, iteration):
        samples_per_class = 10
        # Generated hidden vector
        z = torch.randn(
            samples_per_class * self._n_classes, self._latent_dimension)
        z = z.to(DEFAULT_DEVICE)
        # Generate fake labels
        labels = np.array([
            num for num in range(samples_per_class) for _ in
            range(self._n_classes)])
        labels = torch.Tensor(labels).to(DEFAULT_DEVICE)
        # Store images
        gen_imgs = self._generator(z, labels)
        writer.add_images('gan_grid', gen_imgs, iteration)

    def train(self,
              n_epochs: int,
              logs_dir: str,
              lr: float,
              b1: float,
              b2: float,
              stats_interval: int):
        self._generator.apply(init_weights_g)
        self._discriminator.apply(init_weights_d)

        adversarial_loss, auxiliary_loss = ACGAN._losses()

        optimizer_g, optimizer_d = self._optimizers(lr, b1, b2)

        batches_per_epoch = len(self._data_loader)

        writer = SummaryWriter(logs_dir)

        for epoch in range(n_epochs):
            for i, (imgs, labels) in enumerate(self._data_loader):

                real_imgs = imgs.to(DEFAULT_DEVICE)
                real_labels = labels.to(DEFAULT_DEVICE)

                g_loss, gen_images, gen_labels = \
                        self._train_generator(batch_size=imgs.shape[0],
                                              optimizer=optimizer_g,
                                              adversarial_loss=adversarial_loss,  # noqa
                                              auxiliary_loss=auxiliary_loss)

                d_loss, d_real_loss, d_fake_loss, acc = \
                    self._train_discriminator(images=real_imgs,
                                              labels=real_labels,
                                              generated_images=gen_images,
                                              generated_labels=gen_labels,
                                              optimizer=optimizer_d,
                                              adversarial_loss=adversarial_loss,  # noqa
                                              auxiliary_loss=auxiliary_loss)

                iteration = epoch * batches_per_epoch + i

                if iteration % stats_interval == 0:
                    print('Entering tracking section')
                    ACGAN._track_metrics(
                        writer, g_loss, d_loss, d_real_loss, d_fake_loss, acc, iteration)  # noqa
                    self._track_generated_images(writer, iteration)

                iteration_info = "[Epoch %d/%d] [Batch %d/%d]" \
                                 % (epoch, n_epochs, i, batches_per_epoch)
                metrics_info = "[D loss: %f, acc: %d%%] [G loss: %f]" \
                               % (d_loss.item(), 100 * acc, g_loss.item())
                print(" ".join([iteration_info, metrics_info]))

        writer.close()

    def save_discriminator(self, dst: str):
        torch.save(self._discriminator, dst)

    def save_generator(self, dst):
        torch.save(self._generator,  dst)
