#! /usr/bin/python
import argparse
import numpy as np
# import scipy.ndimage as scimg
from scipy import misc
from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
import os.path
import sys


def preprocess_grasps(grasp_images):
    """
        Make binary images of the grasp images and extract position of origin.
        ---------
        Arguments
        ---------
        grasp_images, dict (int->np.array) - Dictionary mapping integer grasp ids to grasp images.
            Image arrays are expected to be of shape (n, m, 4).

        -------
        Returns
        -------
        grasp_masks, dict (int -> np.array) - maps grasp id to binary image of shape (n, m)
        origins, dict (int -> np.array) - maps grasp id to an array of shape (2,) indicating the center position 
            of the respective grasp mask.
    """
    grasp_masks = {}
    origins = {}
    for grasp_id, rgb_image in grasp_images.iteritems():
        rgb_image = np.flip(rgb_image, axis=(0, 1))
        xs, ys = np.where(np.logical_and.reduce(rgb_image == (255, 0, 0, 255), axis=2))
        if len(xs) != 1 or len(ys) != 1:
            print "Error: There is not only a single red pixel in the image for grasp ", grasp_id
            sys.exit(1)
        origins[grasp_id] = np.array([xs[0], ys[0]], dtype=int) - (np.array(rgb_image.shape[:2], dtype=int) / 2)
        # origins[grasp_id] = np.array([0, 0], dtype=int)
        grasp_masks[grasp_id] = make_binary(rgb_image)
    return grasp_masks, origins


def compute_collision_spaces(world_image, grasp_masks, origins):
    """
        Compute collision spaces using convolution.
        ---------
        Arguments
        ---------
        world_image - np array representing world
        grasp_masks - dict (int -> np array) - binary image for each grasp
        origins - dict (int -> np.array) - robot center positions
        -------
        Returns
        -------
        collision_spaces - dict (int -> np.array) - binary collision spaces
    """
    # make world image binary
    binary_world_image = make_binary(world_image)
    collision_spaces = {}
    for grasp_id, grasp_image in grasp_masks.iteritems():
        collision_spaces[grasp_id] = make_binary(convolve(binary_world_image, grasp_image,
                                                          mode='constant', cval=1.0, origin=origins[grasp_id]))
    return collision_spaces


def compute_cost_spaces(collision_spaces):
    """
        Compute cost spaces.
        ---------
        Arguments
        ---------
        collision_spaces - dict (int -> np.array) - collision spaces
        -------
        Returns
        -------
        cost_spaces - dict (int -> np.array) - floating point cost spaces
    """
    # return collision_spaces
    cost_spaces = {}
    for gid, image in collision_spaces.iteritems():
        inner_distances = distance_transform_edt(image)
        outer_distances = distance_transform_edt(1.0 - image)
        cost_spaces[gid] = outer_distances - inner_distances
    return cost_spaces
    # return {gid: for gid, image in collision_spaces.iteritems()}


def make_binary(image, dtype=float):
    """
        Make the given array binary. Any non-black pixel is mapped to 1.0, any black one to 0.0
    """
    if len(image.shape) == 3:  # we have a color image
        bg_color = (0, 0, 0, 255) if image.shape[2] == 4 else (0, 0, 0)
        binary_image = np.logical_not(np.logical_and.reduce(image == bg_color, axis=2)).astype(float)
    else:
        # grayscale
        binary_image = (image > 0).astype(float)
    return binary_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'grasp_images', help='Path to a folder containing pictures of the robot with the grasped object.' +
                             'The files in the folder should be named <id>.png, where <id> is an integer.' +
                             'The image 0.png should just show the robot. In all picture there should be ' +
                             'a single red pixel (255, 0, 0) indicating the refernce position of the robot.',
                             type=str)
    parser.add_argument(
        'world_image', help='Path to a single image of the world.', type=str)
    parser.add_argument(
        'output_path', help='Path to a folder of where to store the generated cost spaces.', type=str)
    args = parser.parse_args()
    # load grasp images
    if not os.path.isdir(args.grasp_images):
        print "Error: The given grasp images path is not a folder."
        sys.exit(1)
    grasp_images = {}
    for filename in os.listdir(args.grasp_images):
        grasp_id_str = os.path.basename(filename).split('.')[0]
        try:
            grasp_id = int(grasp_id_str)
            grasp_images[grasp_id] = misc.imread(args.grasp_images + '/' + filename)
        except ValueError as e:
            print "Could not parse grasp id from: ", filename
            continue
    # load world image
    world_image = misc.imread(args.world_image)
    # extract origins and make binary masks
    grasp_masks, origins = preprocess_grasps(grasp_images)
    # compute collision spaces through convolution
    collision_spaces = compute_collision_spaces(world_image, grasp_masks, origins)
    # compute cost spaces
    cost_spaces = compute_cost_spaces(collision_spaces)
    # save cost spaces
    for grasp_id, space_image in cost_spaces.iteritems():
        # np.save(args.output_path + '/' + str(grasp_id) + '.npy', space_image)
        space_image -= np.min(space_image)
        space_image /= np.max(space_image) - np.min(space_image)
        misc.imsave(args.output_path + '/' + str(grasp_id) + '.bmp', space_image.astype(float))
    print "Done"
    sys.exit(0)
