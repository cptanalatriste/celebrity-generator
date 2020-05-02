import copy
from unittest import TestCase

import torch
import numpy as np
from torch import optim

from celebrity_generator import create_image_dataloader, scale_tensor, Discriminator, Generator, initialise_weights, \
    loss_against_real_labels, loss_against_fake_labels
from problem_unittests import test_discriminator, test_generator


def compare_model_parameters(parameters, more_parameters):
    """
    Taken from: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    :param parameters:
    :param more_parameters:
    :return:
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(parameters.items(), more_parameters.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismtach found at', key_item_1[0])
                return False
    if models_differ == 0:
        return True


class CelebrityGeneratorTest(TestCase):

    def setUp(self):
        self.batch_size = 128
        data_location = '../processed_celeba_small'
        self.image_dataloader = create_image_dataloader(batch_size=self.batch_size, target_image_size=32,
                                                        data_location=data_location)

    def test_create_image_dataloader(self):
        target_image_size = 32

        image_samples, _ = next(iter(self.image_dataloader))
        self.assertEqual(len(image_samples), self.batch_size)

        an_image = image_samples[0]
        self.assertEqual(an_image.size(), (3, target_image_size, target_image_size))

    def test_scale_tensor(self):
        input_tensor = torch.FloatTensor([0.0, 0.5, 1.0])
        new_scale = (-1, 1)
        rescaled_tensor = scale_tensor(input_tensor, new_scale)

        new_scale_min, new_scale_max = new_scale
        self.assertEqual(rescaled_tensor[0], new_scale_min)
        self.assertEqual(rescaled_tensor[1], (new_scale_min + new_scale_max) / 2)
        self.assertEqual(rescaled_tensor[2], new_scale_max)

    def test_training_process(self):
        image_samples, _ = next(iter(self.image_dataloader))
        new_scale = (-1, 1)
        real_images = scale_tensor(image_samples, new_scale)

        d_conv_dim = 64
        g_conv_dim = 32
        z_size = 100
        D = Discriminator(d_conv_dim)
        G = Generator(z_size=z_size, conv_dim=g_conv_dim)
        d_optimizer = optim.Adam(params=D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        g_optimizer = optim.Adam(params=G.parameters(), lr=0.0002, betas=(0.5, 0.999))

        parameters_before_training = copy.deepcopy(D.state_dict())
        d_loss = D.train(real_images=real_images, generator_network=G, discriminator_optimiser=d_optimizer)
        self.assertIsInstance(d_loss.item(), float)

        parameters_after_training = copy.deepcopy(D.state_dict())
        self.assertFalse(
            compare_model_parameters(parameters_before_training, parameters_after_training))

        g_loss = G.train(real_images=real_images, discriminator_network=D, generator_optimiser=g_optimizer)
        self.assertIsInstance(g_loss.item(), float)


class TestDiscriminator(TestCase):

    def setUp(self):
        self.discriminator_network = Discriminator(conv_dim=64)

    def test_forward_pass(self):
        sample_input = torch.FloatTensor(50, 3, 32, 32).uniform_(-1, 1)

        output = self.discriminator_network.forward(sample_input)

        real_loss = loss_against_real_labels(discriminator_output=output)
        self.assertIsNotNone(real_loss)
        fake_loss = loss_against_fake_labels(discriminator_output=output)
        self.assertIsNotNone(fake_loss)

        output = output.detach()
        self.assertEqual((50, 1), output.size())

    def test_initialise_weights(self):
        convolutional_layer = self.discriminator_network.first_conv_layer[0]
        conv_before_init = convolutional_layer.weight.clone()
        initialise_weights(submodule=convolutional_layer, mean=0.0, std_dev=0.02)

        self.assertFalse(torch.all(torch.eq(conv_before_init, convolutional_layer.weight)))

        normalisation_layer = self.discriminator_network.sec_conv_layer[1]
        norm_before_init = normalisation_layer.weight.clone()
        initialise_weights(submodule=normalisation_layer, mean=0.0, std_dev=0.02)

        self.assertTrue(torch.all(torch.eq(norm_before_init, normalisation_layer.weight)))

    def test_bias_clearance(self):
        linear_layer = self.discriminator_network.final_linear_layer
        initialise_weights(submodule=linear_layer, mean=0.0, std_dev=0.02)

        self.assertEqual(0, linear_layer.bias.data.sum())

    def test_structure(self):
        test_discriminator(Discriminator)


class TestGenerator(TestCase):
    def test_forward(self):
        latent_vector_size = 100
        generator_network = Generator(z_size=latent_vector_size, conv_dim=32)

        print(generator_network)

        sample_size = 16
        fixed_z = np.random.uniform(-1, 1, size=(sample_size, latent_vector_size))
        sample_input = torch.from_numpy(fixed_z).float()

        output = generator_network.forward(sample_input)
        output = output.detach()
        self.assertEqual((sample_size, 3, 32, 32), output.size())

    def test_structure(self):
        test_generator(Generator)
