import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def fashion_mnist(path: str,
                  img_size: int,
                  batch_size: int,
                  training: bool = True):
    """ Returns the dataset object for FashionMNIST

    Args:
        path: Local path where to store the data (will be downloaded if it
            does not exist).
        img_size: Side of the shape image will be resized to.
        batch_size: Size of batches to produce.
        training: Whether to retrieve training (i.e. True) or test (i.e. False).

    Returns:
        FashionMNIST dataset
    """
    norm_means = [0.5]
    norm_stds = norm_means
    return DataLoader(
        dataset=datasets.FashionMNIST(
            root=path,
            train=training,
            download=True,
            transform=transforms.Compose([
                # Resize into [img_size, img_size]
                transforms.Resize(img_size),
                # Move into [0, 1]
                transforms.ToTensor(),
                # Move into [-1, 1]
                transforms.Normalize(norm_means, norm_stds)])),
        batch_size=batch_size,
        shuffle=True,
    )

