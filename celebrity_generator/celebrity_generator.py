from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor


def create_image_dataloader(batch_size, target_image_size, data_location):
    target_dimensions = (target_image_size, target_image_size)
    transformations = Compose([Resize(target_dimensions), ToTensor()])
    image_folder = ImageFolder(root=data_location, transform=transformations)
    return DataLoader(image_folder, batch_size=batch_size, shuffle=True)


def scale_tensor(input_tensor, new_scale):
    new_scale_min, new_scale_max = new_scale
    input_tensor = input_tensor * (new_scale_max - new_scale_min) + new_scale_min
    return input_tensor


class Discriminator(nn.Module):

    def __init__(self, conv_dim, in_channels=3, negative_slope=0.2):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # complete init function
        self.first_conv_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                        out_channels=conv_dim,
                                                        kernel_size=4,
                                                        stride=2,
                                                        padding=1),
                                              nn.LeakyReLU(negative_slope=negative_slope))

        sec_conv_output = conv_dim * 2
        self.sec_conv_layer = nn.Sequential(nn.Conv2d(in_channels=conv_dim,
                                                      out_channels=sec_conv_output,
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding=1),
                                            nn.BatchNorm2d(num_features=sec_conv_output),
                                            nn.LeakyReLU(negative_slope=negative_slope))

        third_conv_output = sec_conv_output * 2
        self.third_conv_layer = nn.Sequential(nn.Conv2d(in_channels=sec_conv_output,
                                                        out_channels=third_conv_output,
                                                        kernel_size=4,
                                                        stride=2,
                                                        padding=1),
                                              nn.BatchNorm2d(num_features=third_conv_output),
                                              nn.LeakyReLU(negative_slope=negative_slope))

        self.linear_layer_input = third_conv_output * 4 * 4
        self.final_linear_layer = nn.Linear(in_features=self.linear_layer_input, out_features=1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior

        x = self.first_conv_layer(x)
        x = self.sec_conv_layer(x)
        x = self.third_conv_layer(x)

        x = x.view(-1, self.linear_layer_input)
        x = self.final_linear_layer(x)

        return x
