import torch
import numpy as np
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


def initialise_weights(submodule, mean, std_dev):
    if isinstance(submodule, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.normal_(submodule.weight.data, mean, std_dev)

        if hasattr(submodule, 'bias') and submodule.bias is not None:
            nn.init.constant_(submodule.bias.data, 0.0)


def loss_against_real_labels(discriminator_output):
    num_images = discriminator_output.size(0)
    real_labels = torch.ones(num_images)
    if torch.cuda.is_available():
        real_labels = real_labels.cuda()

    criterion = nn.BCEWithLogitsLoss()

    return criterion(discriminator_output.squeeze(), real_labels)


def loss_against_fake_labels(discriminator_output):
    num_images = discriminator_output.size(0)
    fake_labels = torch.zeros(num_images)
    if torch.cuda.is_available():
        fake_labels = fake_labels.cuda()

    criterion = nn.BCEWithLogitsLoss()

    return criterion(discriminator_output.squeeze(), fake_labels)


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
                                                        padding=1,
                                                        bias=False),
                                              nn.LeakyReLU(negative_slope=negative_slope))

        sec_conv_output = conv_dim * 2
        self.sec_conv_layer = nn.Sequential(nn.Conv2d(in_channels=conv_dim,
                                                      out_channels=sec_conv_output,
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding=1,
                                                      bias=False),
                                            nn.BatchNorm2d(num_features=sec_conv_output),
                                            nn.LeakyReLU(negative_slope=negative_slope))

        third_conv_output = sec_conv_output * 2
        self.third_conv_layer = nn.Sequential(nn.Conv2d(in_channels=sec_conv_output,
                                                        out_channels=third_conv_output,
                                                        kernel_size=4,
                                                        stride=2,
                                                        padding=1,
                                                        bias=False),
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

    def train_discriminator(self, real_images, generator_network, discriminator_optimiser):
        self.zero_grad()

        if torch.cuda.is_available():
            real_images = real_images.cuda()

        output_from_real = self.forward(real_images)
        loss_from_real = loss_against_real_labels(discriminator_output=output_from_real)

        fake_images = generator_network.generate_images(batch_size=real_images.size()[0])
        output_from_fake = self.forward(fake_images)
        loss_from_fake = loss_against_fake_labels(discriminator_output=output_from_fake)

        discriminator_loss = loss_from_real + loss_from_fake
        discriminator_loss.backward()
        discriminator_optimiser.step()

        return discriminator_loss


class Generator(nn.Module):

    def __init__(self, z_size, conv_dim, final_output_channels=3):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        # complete init function

        self.input_layer_input = z_size
        self.first_transconv_input = 128
        self.init_image_size = 4
        first_transconv_output = int(self.first_transconv_input / 2)

        self.input_layer = nn.Linear(in_features=z_size,
                                     out_features=self.first_transconv_input * self.init_image_size * self.init_image_size)

        self.first_transcov_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=self.first_transconv_input,
                                                                     out_channels=first_transconv_output,
                                                                     kernel_size=4,
                                                                     stride=2,
                                                                     padding=1,
                                                                     bias=False),
                                                  nn.BatchNorm2d(num_features=first_transconv_output),
                                                  nn.ReLU())

        second_transconv_output = conv_dim
        self.sec_transcov_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=first_transconv_output,
                                                                   out_channels=second_transconv_output,
                                                                   kernel_size=4,
                                                                   stride=2,
                                                                   padding=1,
                                                                   bias=False),
                                                nn.BatchNorm2d(num_features=second_transconv_output),
                                                nn.ReLU())

        self.final_transcov_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=second_transconv_output,
                                                                     out_channels=final_output_channels,
                                                                     kernel_size=4,
                                                                     stride=2,
                                                                     padding=1,
                                                                     bias=False),
                                                  nn.Tanh())

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior

        x = self.input_layer(x)
        x = x.view(-1, self.first_transconv_input, self.init_image_size, self.init_image_size)

        x = self.first_transcov_layer(x)
        x = self.sec_transcov_layer(x)
        x = self.final_transcov_layer(x)

        return x

    def generate_images(self, batch_size):
        latent_vector = np.random.uniform(-1, 1, size=(batch_size, self.input_layer_input))
        net_input = torch.from_numpy(latent_vector).float()

        if torch.cuda.is_available():
            net_input = net_input.cuda()

        return self.forward(net_input)

    def train_generator(self, real_images, discriminator_network, generator_optimiser):
        self.zero_grad()

        if torch.cuda.is_available():
            real_images = real_images.cuda()

        fake_images = self.generate_images(batch_size=real_images.size()[0])
        output_from_fake = discriminator_network.forward(fake_images)
        loss_from_fake = loss_against_real_labels(discriminator_output=output_from_fake)

        loss_from_fake.backward()
        generator_optimiser.step()

        return loss_from_fake
