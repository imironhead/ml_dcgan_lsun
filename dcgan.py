"""
"""
import numpy as np
import os
import scipy.misc
import sys
import time

from lsun import Lsun
from model import Dcgan


def make_dir(dir_path):
    """
    Helper function to make a directory if it doesn`t exist.
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def next_real_batch(params):
    """
    Get next batch from LSUN. A bunch of RGB images. The images are resized to
    [batch_size, 64, 64, 3].
    """
    if 'lsun' not in params:
        # lazy loading.
        path_dir_lsun = params.get('path_dir_lsun', None)

        params['lsun'] = Lsun(path_dir_lsun)

    data = params['lsun']

    discriminator_batch_size = params.get('discriminator_batch_size', 128)

    raw_batch = data.next_batch(discriminator_batch_size)

    # crop to 64 x 64 x 3 and copy to numpy array
    discriminator_batch = np.zeros((discriminator_batch_size, 64, 64, 3))

    for i in xrange(len(raw_batch)):
        image = raw_batch[i]

        w, h = image.shape[:2]

        x, y = (w / 2) - 32, (h / 2) - 32

        discriminator_batch[i] = image[x:x + 64, y:y + 64, :]

    return discriminator_batch


def next_fake_batch(params):
    """
    Return random seeds for the generator.
    """
    generator_batch_size = params.get('generator_batch_size', 128)

    generator_seed_size = params.get('generator_seed_size', 128)

    batch = np.random.uniform(
        -1.0,
        1.0,
        size=[generator_batch_size, generator_seed_size])

    return batch.astype(np.float32)


def save_merged_results(params, results, path_merged_results):
    """
    Save the generated images into a sprite sheet.
    """
    width, height = results.shape[1:3]

    count_x = params.get('results_image_x_count', 4)
    count_y = params.get('results_image_y_count', 4)

    image = np.zeros((height * count_y, width * count_x, 3))

    results = 0.5 * (results + 1.0)

    # make the sheet.
    for idx, result in enumerate(results[:count_x * count_y]):
        x = (idx % count_x) * width
        y = (idx / count_x) * height

        image[y:y + height, x:x + width, :] = result

    scipy.misc.imsave(path_merged_results, image)


def train(params):
    """
    """
    print 'training'

    # create model
    dcgan = Dcgan(params)

    fixed_fake_sources = next_fake_batch(params)[:16]

    # train
    for iteration in xrange(params['training_iterations']):
        real_sources = next_real_batch(params)
        fake_sources = next_fake_batch(params)

        dcgan.train_discriminator(fake_sources, real_sources)
        dcgan.train_generator(fake_sources)

        print 'iteration: {}'.format(iteration)

        if iteration % 100 == 0:
            dcgan.save_checkpoint()

        # peek the generator.
        if iteration % 100 == 0:
            fixed_fake_sources[:8] = next_fake_batch(params)[:8]

            fake_results = dcgan.generate(fixed_fake_sources)

            path_dir_results = params.get('path_dir_results', './results/')

            path_results = os.path.join(
                path_dir_results, 'training_{:08}.png'.format(iteration))

            save_merged_results(params, fake_results, path_results)


def generate(params):
    """
    32 x 8 images
    """
    name_results = 'generate_{}.png'.format(time.strftime('%Y%m%d_%H%M%S'))
    path_results = os.path.join(params['path_dir_results'], name_results)

    head, diff = next_fake_batch(params)[:2]
    diff /= 256.0

    seeds = [head + diff * i for i in xrange(256)]

    dcgan = Dcgan(params)

    results = dcgan.generate(seeds)

    save_merged_results(params, results, path_results)


if __name__ == '__main__':
    # training parameters
    params = {}
    params['path_dir_results'] = './results/'
    params['training_iterations'] = 20000000
    params['discriminator_batch_size'] = 128
    params['generator_batch_size'] = 128
    params['generator_seed_size'] = 100

    make_dir(params['path_dir_results'])

    if len(sys.argv) < 2:
        raise Exception('give me parameters')

    if sys.argv[1] == 'g':
        params['results_image_x_count'] = 32
        params['results_image_y_count'] = 8

        generate(params)
    else:
        params['path_dir_lsun'] = sys.argv[1]
        params['results_image_x_count'] = 4
        params['results_image_y_count'] = 4

        train(params)
