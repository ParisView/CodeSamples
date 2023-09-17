from os.path import join
import shutil
import numpy as np
from matplotlib import image as mpimg

from widthHeightClassTargetMarking import whc_target_file_name_suffix, whc_target_directory_name
from imageLoading import create_lists_of_image_and_target_files


INVERTED_PIXELS_RATIO = 0.1
AUGMENTED_IMAGE_MIN_INDEX = 0
AUGMENTED_IMAGE_MAX_INDEX = 9
augmented_image_index_range = range(AUGMENTED_IMAGE_MIN_INDEX, AUGMENTED_IMAGE_MAX_INDEX + 1)
original_image_designator = 'o'
augmented_image_designator = 'a'
augmented_image_index_width = 3


def create_images_and_targets_with_random_pixel_inversion(image_directory_path):
    target_directory_path = join(image_directory_path, whc_target_directory_name)

    # creating a list of image files and a list of target files
    list_of_image_files, list_of_target_files = create_lists_of_image_and_target_files(
        image_directory_path, target_directory_path, whc_target_file_name_suffix
    )

    print("\ncreating files with random pixel inversion\n")

    # one by one loading files as an original file from specified image directory
    for i in range(len(list_of_image_files)):
        # show progress
        print(f'\rprogress: {i + 1} of {len(list_of_image_files)}', end='')

        # check if the current file is an original image and not an augmented one, and check if there are no
        # augmented images created for this original image in the specified augmented image index range
        if (list_of_image_files[i][2] == original_image_designator) and (
                not check_existing_augmented_images(list_of_image_files, list_of_image_files[i])):

            # load the content of the original image file into an array [height (~31), width(~700), colors(4)]
            original_image_file = ''.join([image_directory_path, list_of_image_files[i]])
            original_image = np.array(mpimg.imread(original_image_file))
            image_width = original_image.shape[1]
            image_height = original_image.shape[0]

            for augmented_image_index in augmented_image_index_range:
                # create a list of randomly selected pixel indices
                random_pixel_indices_list = np.random.choice(
                    image_width * image_height,
                    size=int(INVERTED_PIXELS_RATIO * image_width * image_height),
                    replace=False
                )

                image_with_inverted_pixels = np.copy(original_image)

                # invert the values of the pixels with selected indices
                for pixel_index in random_pixel_indices_list:
                    index_h, index_w = divmod(pixel_index, image_width)
                    if image_with_inverted_pixels[index_h, index_w, 0] < 0.5:
                        image_with_inverted_pixels[index_h, index_w, 0] = 1
                        image_with_inverted_pixels[index_h, index_w, 1] = 1
                        image_with_inverted_pixels[index_h, index_w, 2] = 1
                    else:
                        image_with_inverted_pixels[index_h, index_w, 0] = 0
                        image_with_inverted_pixels[index_h, index_w, 1] = 0
                        image_with_inverted_pixels[index_h, index_w, 2] = 0

                # create augmented image file name and augmented target file name
                augmented_image_name = construct_augmented_image_name(
                    list_of_image_files[i], augmented_image_index)
                augmented_target_name = construct_augmented_image_name(
                    list_of_target_files[i], augmented_image_index)
                augmented_image_name = join(image_directory_path, augmented_image_name)
                augmented_target_name = join(target_directory_path, augmented_target_name)

                # save augmented image
                mpimg.imsave(augmented_image_name, image_with_inverted_pixels)

                # copy target file to augmented target file
                shutil.copyfile(join(target_directory_path, list_of_target_files[i]), augmented_target_name)

    print('\nDone')


def check_existing_augmented_images(list_of_image_names, image_name):
    for augmented_image_index in augmented_image_index_range:
        augmented_image_name = construct_augmented_image_name(image_name, augmented_image_index)
        if augmented_image_name in list_of_image_names:
            print(f"\naugmented image '{augmented_image_name}' already exists")
            return True
    return False


def construct_augmented_image_name(original_image_name, augmentation_index):
    return ''.join([original_image_name[0:2], augmented_image_designator, str(augmentation_index).
                   zfill(augmented_image_index_width), original_image_name[3:]])


if __name__ == '__main__':
    IMAGE_DIRECTORY_PATH = "Dataset/"
    create_images_and_targets_with_random_pixel_inversion(IMAGE_DIRECTORY_PATH)
