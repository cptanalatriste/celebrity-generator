# celebrity-generator

![love_island_new_roster](https://github.com/cptanalatriste/celebrity-generator/blob/master/img/new_roster.png?raw=true)


A [deep convolutional generative adversarial network (DCGAN)](https://arxiv.org/abs/1511.06434) for generating faces,
trained over a dataset of celebrity photos.

## Getting started
To train the network, be sure to do the following first:

1. Clone this repository.
2. Download a pre-processed version of the  [CelebFaces Attributes Dataset](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip). 
3. Place the dataset files in your cloned copy of the repository.
4. Make sure you have installed all the Python packages defined in `requirements.txt`.

## Instructions
To explore the training process, you can take a look at the `dlnd_face_generation.ipynb` jupyter notebook.
The network code is contained in the `celebrity_generator` module.