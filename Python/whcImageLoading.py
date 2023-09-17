import random
import sys
from os import listdir
from os.path import isfile, join

import torch
from torch import nn as nn
from matplotlib import image as mpimg
from widthHeightClassTargetMarking import whc_target_file_name_suffix, whc_target_directory_name, \
    PRESSED_KEY_VALUE_SPACE, PRESSED_KEY_VALUE_CHARACTER_START, PRESSED_KEY_VALUE_CHARACTER_MIDDLE, \
    PRESSED_KEY_VALUE_CHARACTER_END, END_OF_CHARACTER_START_GROUP, whc_targets
from imageLoading import create_lists_of_image_and_target_files

# target values
TARGET_VALUE_SPACE = PRESSED_KEY_VALUE_SPACE
TARGET_VALUE_CHARACTER_START = PRESSED_KEY_VALUE_CHARACTER_START
TARGET_VALUE_CHARACTER_MIDDLE = PRESSED_KEY_VALUE_CHARACTER_MIDDLE
TARGET_VALUE_CHARACTER_END = PRESSED_KEY_VALUE_CHARACTER_END
TARGET_END_OF_CHARACTER_START_GROUP = END_OF_CHARACTER_START_GROUP

# character classes to index dictionary
classes_dict = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25,

    '0': 26,
    '1': 27,
    '2': 28,
    '3': 29,
    '4': 30,
    '5': 31,
    '6': 32,
    '7': 33,
    '8': 34,
    '9': 35,

    '<': 36
}

# number of cross-validation datasets to create, also defines test dataset ratio as 1/N_OF_CROSS_VALIDATION_DATASETS
N_OF_CROSS_VALIDATION_DATASETS = 5

# number of images in batch
IMAGES_IN_BATCH = 5

# leading and trailing spaces limit
MAX_LEADING_SPACES = 10
MAX_TRAILING_SPACES = MAX_LEADING_SPACES

# image mean and standard deviation calculated on whole recognition model dataset
IMAGE_MEAN = -0.02598962001502514
IMAGE_STD = 0.3347407579421997


class WHCSegmenterExtractedFeaturesLoader:
    """
    WHCSegmenterExtractedFeaturesLoader class loads from a specified directory into memory
    all image files, that have corresponding target files in the corresponding target
    directory, shuffles their indices, divides them into 5 train and test datasets for
    cross-validation.
    Inside the next_train_batch and next_test_batch methods before being output the
    images are passed through a feature_extractor, that must be given to the methods as
    a parameter.
    """
    def __init__(
            self, image_directory_path
    ):
        target_directory_path = join(image_directory_path, whc_target_directory_name)

        # creating a list of image files and a list of target files
        list_of_image_files, list_of_target_files = create_lists_of_image_and_target_files(
            image_directory_path, target_directory_path, whc_target_file_name_suffix
        )

        self.dataset_list = []
        self.indices_list = []
        self.targets_list = []
        self.train_dataset_index = [0] * N_OF_CROSS_VALIDATION_DATASETS
        self.test_dataset_index = [0] * N_OF_CROSS_VALIDATION_DATASETS
        self.total_images_in_train_dataset = [0] * N_OF_CROSS_VALIDATION_DATASETS
        self.total_images_in_test_dataset = [0] * N_OF_CROSS_VALIDATION_DATASETS
        self.n_of_cross_validation_datasets = N_OF_CROSS_VALIDATION_DATASETS
        self.images_in_loaded_test_batch = 0
        self.images_in_loaded_train_batch = 0

        # statistics parameters
        total_targets_by_class = [0] * len(whc_targets.keys())
        total_targets = 0
        total_trailing_spaces = 0
        total_leading_spaces = 0
        total_image_lengths = 0
        total_targets_taken_to_dataset_by_class = [0] * len(whc_targets.keys())
        total_targets_taken_to_dataset = 0

        print(f"\n\nfound {len(list_of_image_files)} image files with marked targets "
              f"in directory: {image_directory_path}")
        print(f"\ndataset mean: {IMAGE_MEAN}, "
              f"dataset standard deviation: {IMAGE_STD}")
        print(f"dataset normalized by: (x - dataset mean) / dataset standard deviation")

        # load one by one text strip images and their corresponding sets of targets
        for i in range(len(list_of_image_files)):
            # load image
            text_strip_image = torch.tensor(mpimg.imread(join(
                image_directory_path, list_of_image_files[i]))[:, :, 0])
            # preliminary scale and shift of loaded image from (0, 1) range to (-1, 1) range
            text_strip_image = text_strip_image * 2.0 - 1.0

            # normalize image with mean and standard deviation, calculated on whole
            # recognition model dataset
            text_strip_image -= IMAGE_MEAN
            text_strip_image /= IMAGE_STD

            # calculate image length statistics
            total_image_lengths += text_strip_image.shape[1]

            # load set of targets
            # one target corresponds to a column of pixels of the loaded image, that belongs to one
            # of the classes: space, character start, character middle, character end
            loaded_target_file = open(join(
                target_directory_path, list_of_target_files[i]), 'r', encoding='utf8')
            loaded_targets_list = [string.strip() for string in loaded_target_file]
            loaded_target_file.close()

            # limit leading and trailing spaces
            temp_index = 0
            while loaded_targets_list[temp_index] != TARGET_VALUE_CHARACTER_START:
                temp_index += 1
            if temp_index < MAX_LEADING_SPACES:
                targets_list_index = 0
                image_position_x = targets_list_index
            else:
                targets_list_index = temp_index - MAX_LEADING_SPACES
                image_position_x = targets_list_index
                total_leading_spaces += targets_list_index
            temp_index = len(loaded_targets_list) - 1
            while loaded_targets_list[temp_index] != TARGET_VALUE_CHARACTER_END:
                temp_index -= 1
            if len(loaded_targets_list) - 1 - temp_index < MAX_TRAILING_SPACES:
                image_length = len(loaded_targets_list)
            else:
                image_length = temp_index + MAX_TRAILING_SPACES
                total_trailing_spaces += len(loaded_targets_list) - image_length

            # fill targets and indices lists
            equal_classes_targets_list = []
            targets_indices_list = []
            before_start_spaces = []
            after_end_spaces = []
            other_spaces = []
            before_end_middles = []
            after_start_middles = []
            other_middles = []

            while targets_list_index < image_length:
                key = loaded_targets_list[targets_list_index]
                total_targets_by_class[whc_targets[key]] += 1
                total_targets += 1

                if key == TARGET_VALUE_CHARACTER_START:
                    equal_classes_targets_list.append(whc_targets[key])
                    targets_indices_list.append(image_position_x)
                    total_targets_taken_to_dataset_by_class[whc_targets[key]] += 1
                    total_targets_taken_to_dataset += 1
                    targets_list_index += 1
                    while loaded_targets_list[targets_list_index] != TARGET_END_OF_CHARACTER_START_GROUP:
                        targets_list_index += 1
                elif key == TARGET_VALUE_CHARACTER_END:
                    equal_classes_targets_list.append(whc_targets[key])
                    targets_indices_list.append(image_position_x)
                    total_targets_taken_to_dataset_by_class[whc_targets[key]] += 1
                    total_targets_taken_to_dataset += 1
                elif key == TARGET_VALUE_SPACE:
                    if targets_list_index > 0 and (
                            loaded_targets_list[targets_list_index - 1] ==
                            TARGET_VALUE_CHARACTER_END):
                        after_end_spaces.append(image_position_x)
                    elif targets_list_index + 1 < image_length and (
                            loaded_targets_list[targets_list_index + 1] ==
                            TARGET_VALUE_CHARACTER_START):
                        before_start_spaces.append(image_position_x)
                    else:
                        other_spaces.append(image_position_x)
                elif key == TARGET_VALUE_CHARACTER_MIDDLE:
                    if (loaded_targets_list[targets_list_index - 1] ==
                            TARGET_END_OF_CHARACTER_START_GROUP):
                        after_start_middles.append(image_position_x)
                    elif (loaded_targets_list[targets_list_index + 1] ==
                            TARGET_VALUE_CHARACTER_END):
                        before_end_middles.append(image_position_x)
                    else:
                        other_middles.append(image_position_x)

                targets_list_index += 1
                image_position_x += 1

            n_characters_in_image = len(equal_classes_targets_list) // 2

            # shuffle all lists with spaces and character middles
            random.shuffle(before_start_spaces)
            random.shuffle(after_end_spaces)
            random.shuffle(other_spaces)
            random.shuffle(before_end_middles)
            random.shuffle(after_start_middles)
            random.shuffle(other_middles)

            # append spaces to targets_indices_list and equal_classes_targets_list
            result_list = []
            n_samples_to_append = min(n_characters_in_image, len(before_start_spaces) +
                                      len(after_end_spaces) + len(other_spaces))
            appended_samples = 0
            appended_before_start_spaces = 0
            appended_after_end_spaces = 0
            appended_other_spaces = 0
            while appended_samples < n_samples_to_append:
                if appended_before_start_spaces < len(before_start_spaces):
                    result_list.append(before_start_spaces[appended_before_start_spaces])
                    appended_samples += 1
                    appended_before_start_spaces += 1
                if appended_samples < n_samples_to_append:
                    if appended_after_end_spaces < len(after_end_spaces):
                        result_list.append(after_end_spaces[appended_after_end_spaces])
                        appended_samples += 1
                        appended_after_end_spaces += 1
                    if appended_samples < n_samples_to_append:
                        if appended_other_spaces < len(other_spaces):
                            result_list.append(other_spaces[appended_other_spaces])
                            appended_samples += 1
                            appended_other_spaces += 1
            targets_indices_list += result_list
            equal_classes_targets_list += \
                [whc_targets[TARGET_VALUE_SPACE]] * len(result_list)
            total_targets_taken_to_dataset_by_class[
                whc_targets[TARGET_VALUE_SPACE]] += len(result_list)
            total_targets_taken_to_dataset += len(result_list)

            # append character middles to targets_indices_list and equal_classes_targets_list
            result_list = []
            n_samples_to_append = min(n_characters_in_image, len(before_end_middles) +
                                      len(after_start_middles) + len(other_middles))
            appended_samples = 0
            appended_before_end_middles = 0
            appended_after_start_middles = 0
            appended_other_middles = 0
            while appended_samples < n_samples_to_append:
                if appended_before_end_middles < len(before_end_middles):
                    result_list.append(before_end_middles[appended_before_end_middles])
                    appended_samples += 1
                    appended_before_end_middles += 1
                if appended_samples < n_samples_to_append:
                    if appended_after_start_middles < len(after_start_middles):
                        result_list.append(after_start_middles[appended_after_start_middles])
                        appended_samples += 1
                        appended_after_start_middles += 1
                    if appended_samples < n_samples_to_append:
                        if appended_other_middles < len(other_middles):
                            result_list.append(other_middles[appended_other_middles])
                            appended_samples += 1
                            appended_other_middles += 1
            targets_indices_list += result_list
            equal_classes_targets_list += \
                [whc_targets[TARGET_VALUE_CHARACTER_MIDDLE]] * len(result_list)
            total_targets_taken_to_dataset_by_class[
                whc_targets[TARGET_VALUE_CHARACTER_MIDDLE]] += len(result_list)
            total_targets_taken_to_dataset += len(result_list)

            targets_tensor = torch.tensor(equal_classes_targets_list)
            indices_tensor = torch.tensor(targets_indices_list)

            # add loaded image and set of targets to corresponding lists
            self.dataset_list.append(text_strip_image)
            self.indices_list.append(indices_tensor)
            self.targets_list.append(targets_tensor)

        # print statistics of loaded targets
        print(f"total length of all images {total_image_lengths}")
        print(f"total leading spaces {total_leading_spaces}")
        print(f"total trailing spaces {total_trailing_spaces}")
        print(f"loaded {total_targets} targets")
        total_targets_by_class_to_print = []
        for j in range(len(total_targets_by_class)):
            whc_targets_key = list(whc_targets.keys())[list(whc_targets.values()).index(j)]
            total_targets_by_class_to_print.append([whc_targets_key, total_targets_by_class[j]])
        print(f"targets (not images) distribution by class: {total_targets_by_class_to_print}")
        print(f"added {total_targets_taken_to_dataset} targets to dataset")
        total_targets_taken_to_dataset_by_class_to_print = []
        for j in range(len(total_targets_taken_to_dataset_by_class)):
            whc_targets_key = list(whc_targets.keys())[list(whc_targets.values()).index(j)]
            total_targets_taken_to_dataset_by_class_to_print.append(
                [whc_targets_key, total_targets_taken_to_dataset_by_class[j]])
        print(f"targets taken to dataset (not images) distribution by class:"
              f" {total_targets_taken_to_dataset_by_class_to_print}")

        # checking if statistics numbers add up
        if not ((sum(total_targets_by_class) == total_targets) and
                (sum(total_targets_taken_to_dataset_by_class) == total_targets_taken_to_dataset) and
                (total_image_lengths == total_leading_spaces + total_trailing_spaces + total_targets)):
            print(f"\n\n-------------------  WARNING: statistics don't add up  ------------------\n\n")

        # shuffle indices and create cross validation train and test lists of indices
        n_of_datasets = N_OF_CROSS_VALIDATION_DATASETS
        shuffled_indices = list(range(len(self.dataset_list)))
        random.shuffle(shuffled_indices)
        self.train_dataset_list = []
        self.test_dataset_list = [shuffled_indices[i::n_of_datasets] for i in range(n_of_datasets)]
        for n in range(n_of_datasets):
            train_list = []
            for k in range(n_of_datasets):
                if k != n:
                    train_list += self.test_dataset_list[k]
            self.train_dataset_list.append(train_list)
            self.total_images_in_test_dataset[n] = len(self.test_dataset_list[n])
            self.total_images_in_train_dataset[n] = len(train_list)
        print(f"images in test dataset: {self.total_images_in_test_dataset}")
        print(f"images in train dataset: {self.total_images_in_train_dataset}")

    def next_train_batch(self, feature_extractor, images_in_batch=IMAGES_IN_BATCH,
                         cross_validation_set=0, output_device="cpu"):
        cross_validation_set = cross_validation_set % N_OF_CROSS_VALIDATION_DATASETS

        # if reached the end of dataset list, then start from the beginning
        if self.reached_train_dataset_end(cross_validation_set):
            self.train_dataset_index[cross_validation_set] = 0

        features_list = []
        targets_list = []
        sample_count = 0
        while (
                sample_count < images_in_batch and
                (not self.reached_train_dataset_end(cross_validation_set))
        ):
            sample_index = self.train_dataset_list[cross_validation_set][
                self.train_dataset_index[cross_validation_set]]
            image_output = self.dataset_list[sample_index][None, None, :]
            features_output = feature_extractor(image_output.to(output_device))
            indices_tensor = self.indices_list[sample_index].to(output_device)
            features_output = torch.index_select(features_output, 0, indices_tensor)
            target_output = self.targets_list[sample_index].to(output_device)
            features_list.append(features_output)
            targets_list.append(target_output)
            self.train_dataset_index[cross_validation_set] += 1
            sample_count += 1

        self.images_in_loaded_train_batch = sample_count

        features_output = torch.cat(features_list, 0).detach().clone()
        target_output = torch.cat(targets_list, 0).detach().clone()
        return features_output, target_output

    def next_test_batch(self, feature_extractor,  images_in_batch=IMAGES_IN_BATCH,
                        cross_validation_set=0, output_device="cpu"):
        cross_validation_set = cross_validation_set % N_OF_CROSS_VALIDATION_DATASETS

        # if reached the end of dataset list, then start from the beginning
        if self.reached_test_dataset_end(cross_validation_set):
            self.test_dataset_index[cross_validation_set] = 0

        features_list = []
        targets_list = []
        sample_count = 0
        while (
                sample_count < images_in_batch and
                (not self.reached_test_dataset_end(cross_validation_set))
        ):
            sample_index = self.test_dataset_list[cross_validation_set][
                self.test_dataset_index[cross_validation_set]]
            image_output = self.dataset_list[sample_index][None, None, :]
            features_output = feature_extractor(image_output.to(output_device))
            indices_tensor = self.indices_list[sample_index].to(output_device)
            features_output = torch.index_select(features_output, 0, indices_tensor)
            target_output = self.targets_list[sample_index].to(output_device)
            features_list.append(features_output)
            targets_list.append(target_output)
            self.test_dataset_index[cross_validation_set] += 1
            sample_count += 1

        self.images_in_loaded_test_batch = sample_count

        features_output = torch.cat(features_list, 0).detach().clone()
        target_output = torch.cat(targets_list, 0).detach().clone()
        return features_output, target_output

    def reset_train_dataset_index(self, cross_validation_set=0):
        self.train_dataset_index[cross_validation_set] = 0

    def reset_test_dataset_index(self, cross_validation_set=0):
        self.test_dataset_index[cross_validation_set] = 0

    def reached_train_dataset_end(self, cross_validation_set=0):
        return self.train_dataset_index[cross_validation_set] >= len(self.train_dataset_list[cross_validation_set])

    def reached_test_dataset_end(self, cross_validation_set=0):
        return self.test_dataset_index[cross_validation_set] >= len(self.test_dataset_list[cross_validation_set])
