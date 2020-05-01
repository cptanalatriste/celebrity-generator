from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor


def create_image_dataloader(batch_size, target_image_size, data_location):
    target_dimensions = (target_image_size, target_image_size)
    transformations = Compose([Resize(target_dimensions), ToTensor()])
    image_folder = ImageFolder(root=data_location, transform=transformations)
    return DataLoader(image_folder, batch_size=batch_size, shuffle=True)
