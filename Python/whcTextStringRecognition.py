import random
from os.path import join
from threading import Lock
from matplotlib import image as mpimg, pyplot as plt
import torch
import numpy as np
import copy


from widthHeightClassTargetMarking import whc_target_file_name_suffix, whc_target_directory_name, \
    PRESSED_KEY_VALUE_SPACE, PRESSED_KEY_VALUE_CHARACTER_START, PRESSED_KEY_VALUE_CHARACTER_MIDDLE, \
    PRESSED_KEY_VALUE_CHARACTER_END, END_OF_CHARACTER_START_GROUP, whc_targets
from imageLoading import create_lists_of_image_and_target_files
from whcSegmenterDataPreparation import load_feature_extractor_checkpoint
from whcCharSegmentation import WHCCharSegmentationConvolutionalModel
from whcCharRecognition import WHCCharRecognitionModel
from whcImageLoading import classes_dict


# colors
COLOR_GREEN_A700_STRING = '#00C853'
COLOR_PINK_A400_STRING = '#F50057'


# target values
TARGET_VALUE_SPACE = PRESSED_KEY_VALUE_SPACE
TARGET_VALUE_CHARACTER_START = PRESSED_KEY_VALUE_CHARACTER_START
TARGET_VALUE_CHARACTER_MIDDLE = PRESSED_KEY_VALUE_CHARACTER_MIDDLE
TARGET_VALUE_CHARACTER_END = PRESSED_KEY_VALUE_CHARACTER_END
TARGET_END_OF_CHARACTER_START_GROUP = END_OF_CHARACTER_START_GROUP


def no_extremums_mean(data_array, n_delete_min_and_max_loops=1):
    """
    This function calculates a mean value of all elements of input data array except those several
    elements, that have min and max values. Number of excluded elements with min values is the same
    as the number of excluded elements with max values and is equal to the value of the parameter
    n_delete_min_and_max_loops.
    :param data_array: the input data array, that contains all elements, on which the calculation
    of the mean value should be performed.
    :param n_delete_min_and_max_loops: number of elements with min values to be removed from input data
    array before calculation of the mean value, and simultaneously number of elements with max values
    to be removed from input data array before calculation of the mean value.
    :return: mean value, calculated on all elements of the input data array after the corresponding
    number of elements with min and max values has been removed.
    """
    data_array = data_array.copy()
    for i in range(n_delete_min_and_max_loops):
        max_index = data_array.argmax(0)
        data_array = np.delete(data_array, max_index, 0)
        min_index = data_array.argmin(0)
        data_array = np.delete(data_array, min_index, 0)
    mean_value = data_array.sum(0) // (data_array.shape[0])

    return mean_value


class CharacterItem:
    """
    Data holder class. Holds data for one character. Start, end, lowest and highest character points
    correspond to the character's position in an image of a string of characters.
    """
    def __init__(self, character_class_index=-1, character_start=-1,
                 character_end=-1, character_lowest_point=-1, character_highest_point=-1):
        self.character_class_index = character_class_index
        self.character_start = character_start
        self.character_end = character_end
        self.character_lowest_point = character_lowest_point
        self.character_highest_point = character_highest_point


class CharacterGroupsDetectorOutput:
    """
    Data holder class. Holds output data of the detect_next_character_group method of
    a CharacterGroupsDetector.
    """
    def __init__(self, detected_character_group=False, character_start=-1, character_end=-1,
                 new_array_index=-1):
        self.detected_character_group = detected_character_group
        self.character_start = character_start
        self.character_end = character_end
        self.new_array_index = new_array_index


class CharacterGroupsDetector:
    """
    The CharacterGroupsDetector class detects groups of targets, that form a single character.
    Input for the __call__ method of the CharacterGroupsDetector class is an array of
    targets, that is taken from a prediction of a segmentation model, that marks each column
    of pixels of an image of a text string as one of the following classes: character start,
    character middle, character end, space.
    """
    def __init__(self):
        # setting constants
        self.MAX_ALLOWED_ERRORS_IN_GROUP = 2
        self.MAX_NEXT_TARGETS_TO_CHECK = self.MAX_ALLOWED_ERRORS_IN_GROUP + 1
        self.MAX_END_SEARCH_ROUNDS = 3
        self.END_OF_SEQUENCE = - 1
        self.MIN_GROUP_WIDTH = 5

    def __call__(self, prediction_array):
        array_index = 0
        predicted_characters_list = []
        while array_index < prediction_array.shape[0]:
            detector_output = self.detect_next_character_group(prediction_array, array_index)
            array_index = detector_output.new_array_index
            if detector_output.detected_character_group:
                # add character item to the predicted_characters_list
                character_item = CharacterItem(
                    character_start=detector_output.character_start,
                    character_end=detector_output.character_end
                )
                predicted_characters_list.append(character_item)

        predicted_characters_list = self.clean_up_characters_list(predicted_characters_list)

        return predicted_characters_list

    def clean_up_characters_list(self, characters_list):
        # mean_calc_length is the length of the array on which mean distance between centers is
        # calculated for end characters. For too wide and too narrow characters this length is doubled.
        mean_calc_length = 5

        # if characters list is long enough, perform the clean up
        min_characters_in_list = 2 * mean_calc_length + 4
        if len(characters_list) >= min_characters_in_list:
            # calculate arrays of coordinates and distances
            distances_between_centers = np.zeros(len(characters_list) - 1, dtype=int)
            character_starts = np.zeros(len(characters_list), dtype=int)
            character_ends = np.zeros(len(characters_list), dtype=int)
            for i in range(len(characters_list)):
                character_starts[i] = characters_list[i].character_start
                character_ends[i] = characters_list[i].character_end
            character_centers = (character_starts + character_ends) // 2
            for i in range(distances_between_centers.shape[0]):
                distances_between_centers[i] = character_centers[i + 1] - character_centers[i]

            # perform the clean up of end characters
            characters_list = self.clean_up_end_characters(
                characters_list, character_starts, character_ends, distances_between_centers,
                mean_calc_length
            )

            # recalculate arrays of coordinates and distances after cleaning up end characters
            distances_between_centers = np.zeros(len(characters_list) - 1, dtype=int)
            character_starts = np.zeros(len(characters_list), dtype=int)
            character_ends = np.zeros(len(characters_list), dtype=int)
            for i in range(len(characters_list)):
                character_starts[i] = characters_list[i].character_start
                character_ends[i] = characters_list[i].character_end
            character_widths = character_ends - character_starts + 1
            character_centers = (character_starts + character_ends) // 2
            for i in range(distances_between_centers.shape[0]):
                distances_between_centers[i] = character_centers[i + 1] - character_centers[i]

            # perform the clean up of too wide and too narrow characters
            characters_list = self.clean_up_too_wide_or_narrow_characters(
                characters_list, character_starts, character_ends, character_widths,
                distances_between_centers, mean_calc_length
            )

        return characters_list

    def clean_up_end_characters(self, characters_list, starts_array, ends_array,
                                c_distance_array, mean_calc_length):
        # calculate mean distance between centers without extremums
        start_distances_mean = no_extremums_mean(c_distance_array[2:2 + mean_calc_length])
        end_distances_mean = no_extremums_mean(c_distance_array[-2 - mean_calc_length:-2])

        # delete start and end characters if they are too far from the rest of the text
        if starts_array[2] - ends_array[1] - 1 > start_distances_mean:
            del characters_list[0:2]
        else:
            if starts_array[1] - ends_array[0] - 1 > start_distances_mean:
                del characters_list[0]
        if starts_array[-2] - ends_array[-3] - 1 > end_distances_mean:
            del characters_list[-2:]
        else:
            if starts_array[-1] - ends_array[-2] - 1 > end_distances_mean:
                del characters_list[-1]
        return characters_list

    def clean_up_too_wide_or_narrow_characters(
            self, characters_list, starts_array, ends_array, widths_array, c_distance_array,
            mean_calc_length
    ):
        too_wide_character_factor = 1.2
        too_narrow_character_factor = 1.5
        doubled_mean_calc_length = 2 * mean_calc_length
        distances_mean = no_extremums_mean(c_distance_array[0:doubled_mean_calc_length],
                                           n_delete_min_and_max_loops=3)

        add_characters_dict = {}
        unite_characters_dict = {}
        # for each character in characters list perform the clean up
        for i in range(len(characters_list)):
            # recalculate distances_mean for characters in the middle of the text string, that
            # are at least mean_calc_length positions away from ends of the string
            if mean_calc_length < i <= c_distance_array.shape[0] - mean_calc_length:
                distances_mean = no_extremums_mean(
                    c_distance_array[i - mean_calc_length:i + mean_calc_length],
                    n_delete_min_and_max_loops=3
                )

            # handle too wide characters
            if widths_array[i] > distances_mean * too_wide_character_factor:
                divider = widths_array[i] // distances_mean
                if i < c_distance_array.shape[0]:
                    padding = (starts_array[i + 1] - ends_array[i]) // 2
                else:
                    padding = (starts_array[i] - ends_array[i - 1]) // 2
                recovered_width = widths_array[i] // (divider + 1)
                add_list = []

                for j in range(divider):
                    add_list.append(CharacterItem())
                characters_list[i].character_start = ends_array[i] + padding - recovered_width
                add_list[-1].character_end = characters_list[i].character_start - 1
                add_list[-1].character_start = \
                    characters_list[i].character_start - recovered_width

                for j in range(-2, -1 - len(add_list), -1):
                    add_list[j].character_end = add_list[j + 1].character_start - 1
                    add_list[j].character_start = \
                        add_list[j + 1].character_start - recovered_width
                add_list[0].character_start = starts_array[i]
                add_characters_dict[i] = add_list

            # handle too narrow characters
            elif i < c_distance_array.shape[0]:
                if c_distance_array[i] * too_narrow_character_factor < distances_mean:
                    unite_characters_dict[i] = i + 1

        # reconstruct characters list
        last_index = len(characters_list) - 1
        for i in range(last_index, -1, -1):
            if i in add_characters_dict:
                add_list = add_characters_dict[i]
                characters_list[i:i] = add_list
            elif i in unite_characters_dict:
                characters_list[i].character_end = characters_list[i + 1].character_end
                del characters_list[i + 1]

        return characters_list

    def detect_next_character_group(self, prediction_array, array_index):
        next_target = self.read_target(prediction_array, array_index)

        if next_target == whc_targets[TARGET_VALUE_SPACE]:
            array_index += self.sort_multiple_encounters(
                prediction_array, array_index, whc_targets[TARGET_VALUE_SPACE])
            next_target = self.read_target(prediction_array, array_index)

        if next_target == whc_targets[TARGET_VALUE_CHARACTER_END]:
            array_index += self.sort_multiple_encounters(
                prediction_array, array_index, whc_targets[TARGET_VALUE_CHARACTER_END])
            next_target = self.read_target(prediction_array, array_index)

        output = CharacterGroupsDetectorOutput(
            new_array_index=array_index
        )

        if next_target == whc_targets[TARGET_VALUE_CHARACTER_START]:
            # write character start position
            output.character_start = array_index

            middle_found, middle_search_index_increment = self.find_next(
                prediction_array, array_index, whc_targets[TARGET_VALUE_CHARACTER_MIDDLE])
            if middle_found:
                array_index += middle_search_index_increment
                error_count = middle_search_index_increment - 1
                array_index += self.sort_multiple_encounters(
                    prediction_array, array_index, whc_targets[TARGET_VALUE_CHARACTER_MIDDLE]
                )
                self.find_group_end(prediction_array, array_index, error_count, output)
            else:
                output.new_array_index += 1

        elif next_target == whc_targets[TARGET_VALUE_CHARACTER_MIDDLE]:
            # write character start position
            output.character_start = array_index
            error_count = 1
            array_index += self.sort_multiple_encounters(
                prediction_array, array_index, whc_targets[TARGET_VALUE_CHARACTER_MIDDLE]
            )
            self.find_group_end(prediction_array, array_index, error_count, output)

        return output

    def find_group_end(self, prediction_array, array_index, error_count, output):
        end_search_round = 0
        while end_search_round < self.MAX_END_SEARCH_ROUNDS:
            next_target = self.read_target(prediction_array, array_index + end_search_round)
            if next_target == whc_targets[TARGET_VALUE_CHARACTER_END]:
                # character group is detected
                # handle double end
                if (self.read_target(prediction_array, array_index + 1 + end_search_round) ==
                        whc_targets[TARGET_VALUE_CHARACTER_END]):
                    array_index += 1
                # write character end position
                output.character_end = array_index + end_search_round
                output.new_array_index = array_index + end_search_round + 1
                if output.character_end - output.character_start + 1 >= self.MIN_GROUP_WIDTH:
                    output.detected_character_group = True
                # stop search
                end_search_round = self.MAX_END_SEARCH_ROUNDS
            else:
                error_count += 1
                if error_count > self.MAX_ALLOWED_ERRORS_IN_GROUP:
                    if end_search_round > 0:
                        # character group is detected
                        # write character end position
                        output.character_end = array_index - 1
                        output.new_array_index = array_index
                        if output.character_end - output.character_start + 1 >= self.MIN_GROUP_WIDTH:
                            output.detected_character_group = True
                        # stop search
                        end_search_round = self.MAX_END_SEARCH_ROUNDS
                    else:
                        end_search_round = self.MAX_END_SEARCH_ROUNDS
                        output.new_array_index += 1
                else:
                    end_search_round += 1

    def find_next(self, array, array_index, target_value):
        """
        This function searches for the specified target in the next MAX_NEXT_TARGETS_TO_CHECK
        positions of the target_array.
        """

        # index_increment is returned as the position at which the specified target was
        # found counting from the position of the array_index
        index_increment = 1
        continue_search = True
        found_specified_target = False

        while continue_search:
            if array_index + index_increment < array.shape[0]:  # array index within array length
                if array[array_index + index_increment] == target_value:  # found the specified target
                    continue_search = False
                    found_specified_target = True
                else:  # did not find the specified target
                    # if checked less than MAX_NEXT_TARGETS_TO_CHECK targets
                    if index_increment < self.MAX_NEXT_TARGETS_TO_CHECK - 1:
                        index_increment += 1
                    else:
                        continue_search = False
            else:
                continue_search = False

        return found_specified_target, index_increment

    def read_target(self, array, array_index):
        return array[array_index] if array_index < array.shape[0] \
            else self.END_OF_SEQUENCE

    def sort_multiple_encounters(self, array, array_index, target_value, index_increment=0):
        while array_index + index_increment < array.shape[0] and \
                array[array_index + index_increment] == target_value:
            index_increment += 1
        return index_increment


class CharacterHeightPositionDetector:
    def __call__(self, v_strip_image):
        signal_array = v_strip_image.numpy()
        signal_array = signal_array.min(1)
        signal_array -= signal_array.mean()

        index_range = np.arange(signal_array.shape[0])

        lowest_point_correlation_function = np.zeros(signal_array.shape[0])
        highest_point_correlation_function = np.zeros(signal_array.shape[0])

        for threshold_index in range(signal_array.shape[0]):
            lowest_point_threshold_function = np.where(
                index_range <= threshold_index, 1.0, -1.0)

            lowest_point_correlation_function[threshold_index] = (
                    signal_array * lowest_point_threshold_function).sum()

            highest_point_threshold_function = np.where(
                index_range < threshold_index, -1.0, 1.0)

            highest_point_correlation_function[threshold_index] = (
                    signal_array * highest_point_threshold_function).sum()

        lowest = lowest_point_correlation_function.argmax()
        highest = highest_point_correlation_function.argmax()
        half_length = signal_array.shape[0] // 2

        # if highest character point is detected lower than lowest character point, then
        # only one of them is correct, the other point should be changed to the opposite edge
        # of the image
        if highest <= lowest:
            if np.abs(lowest - half_length) <= np.abs(highest - half_length):
                # lowest is closer to middle so it is for sure correct
                highest = signal_array.shape[0] - 1
            else:
                # highest is closer to middle so it is for sure correct
                lowest = 0

        return lowest, highest


class WHCTargetListToListOfCharacterItemsConverter:
    """
    The WHCTargetListToListOfCharacterItemsConverter class converts a targets list (which is
    produced during target marking of images of text strings for the segmenter model training)
    to a list of items of the CharacterItem class.
    """

    def __call__(self, whc_targets_list):
        list_of_character_items = []
        targets_list_index = 0
        target_index = 0

        while targets_list_index < len(whc_targets_list):
            whc_target = whc_targets_list[targets_list_index]

            if whc_target == TARGET_VALUE_CHARACTER_START:
                character_start = target_index
                targets_list_index += 1
                character_lowest_point = int(whc_targets_list[targets_list_index])
                targets_list_index += 1
                character_highest_point = int(whc_targets_list[targets_list_index])
                targets_list_index += 1
                character_class_index = classes_dict[whc_targets_list[targets_list_index]]
                targets_list_index += 1
                while whc_targets_list[targets_list_index] != TARGET_END_OF_CHARACTER_START_GROUP:
                    targets_list_index += 1
                targets_list_index += 1
                target_index += 1
                while whc_targets_list[targets_list_index] != TARGET_VALUE_CHARACTER_END:
                    targets_list_index += 1
                    target_index += 1
                character_end = target_index

                character_item = CharacterItem(
                    character_class_index, character_start, character_end,
                    character_lowest_point, character_highest_point
                )
                list_of_character_items.append(character_item)

            targets_list_index += 1
            target_index += 1

        return list_of_character_items


class WHCTextStringRecognizer:
    """
    The WHCTextStringRecognizer class demonstrates the result of character recognition on
    an image of a text string. The demonstration image must be marked and have a
    corresponding target file.
    """

    def __init__(self, feature_extractor, image_segmenter, character_recognizer,
                 char_groups_detector, char_h_pos_detector, target_list_converter):
        self.feature_extractor = feature_extractor
        self.image_segmenter = image_segmenter
        self.character_recognizer = character_recognizer
        self.char_groups_detector = char_groups_detector
        self.char_h_pos_detector = char_h_pos_detector
        self.target_list_converter = target_list_converter
        self.recognition_statistics = 0

        # character cutout height and width calculation parameters
        self.H_W_CUTOUT_PADDING = 2
        self.H_W_THRESHOLD_1X1 = 18  # 18x18 pixels for 1x1 result of nn calculations
        self.H_W_STEP = 4

        # image mean and reciprocal to standard deviation calculated on whole
        # recognition model dataset
        self.IMAGE_MEAN = -0.02598962001502514
        self.IMAGE_STD_RECIPROCAL = 2.9873864364394844

        # image mean and reciprocal to standard deviation calculated on whole
        # segmentation model dataset
        self.STRIP_IMAGE_MEAN = 0.0031279860995709896
        self.STRIP_IMAGE_STD_RECIPROCAL = 3.5935380458831787

        self.lock = Lock()
        plt.ion()
        self.figure = plt.figure(figsize=(15.0, 8.0), constrained_layout=True)
        self.figure.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
        spec = self.figure.add_gridspec(8, 3)
        self.image_ax = self.figure.add_subplot(spec[0: 3, :])
        self.cut_char_ax = self.figure.add_subplot(spec[3: 8, :])
        self.TEXT_FONT_SIZE_FACTOR = 0.5
        self.TEXT_SHIFT_FACTOR = 0.1

    def __call__(self, image_directory_path, show_images=True, shuffle_images=False,
                 print_segmentation=False):
        target_directory_path = join(image_directory_path, whc_target_directory_name)

        # creating a list of image files and a list of target files
        list_of_image_files, list_of_target_files = create_lists_of_image_and_target_files(
            image_directory_path, target_directory_path, whc_target_file_name_suffix
        )

        print(f"\n\nfound {len(list_of_image_files)} image files with marked targets "
              f"in directory: {image_directory_path}")
        connection_id = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press_event)
        total_correctly_recognized_images = 0
        total_processed_images = 0
        self.recognition_statistics = np.zeros(len(list_of_image_files))

        # load one by one in shuffled order text strip images and their
        # corresponding sets of targets
        shuffled_indices = list(range(len(list_of_image_files)))

        # shuffle images order if corresponding parameter is set
        if shuffle_images:
            random.shuffle(shuffled_indices)

        for image_index in shuffled_indices:
            # load image
            text_strip_image_to_show = torch.tensor(mpimg.imread(join(
                image_directory_path, list_of_image_files[image_index])))
            text_strip_image_to_process = text_strip_image_to_show[:, :, 0]
            text_strip_image_to_process = text_strip_image_to_process * 2.0 - 1.0

            # load set of targets and convert them into list of character class indices
            loaded_target_file = open(join(
                target_directory_path, list_of_target_files[image_index]), 'r', encoding='utf8')
            loaded_targets_list = [string.strip() for string in loaded_target_file]
            loaded_target_file.close()
            target_characters_list = self.target_list_converter(loaded_targets_list)
            target_characters_list = [char_item.character_class_index for char_item in
                                      target_characters_list]

            # add two dimensions for compatibility with torch modules
            text_strip_image_to_process = text_strip_image_to_process[None, None, :]

            # normalize image
            normalized_image_to_process = (text_strip_image_to_process - self.STRIP_IMAGE_MEAN
                                           ).detach().clone()
            normalized_image_to_process *= self.STRIP_IMAGE_STD_RECIPROCAL

            # segment image
            segmenter_output = self.image_segmenter(normalized_image_to_process)
            segmentation_prediction = np.array(segmenter_output.max(1)[1].data)

            # detect character groups and create list of predicted characters
            predicted_characters_list = self.char_groups_detector(segmentation_prediction)

            # if print_segmentation is on, then calculate recovered segmentation and print
            # prediction vs recovered vs targets
            if print_segmentation:
                print(f'\n\n===============================================\nN: {image_index}')
                print(f'segmentation prediction: {segmentation_prediction}')
                groups_detector_recovered_array = np.zeros(
                    predicted_characters_list[-1].character_end + 1, dtype=int)
                for predicted_character in predicted_characters_list:
                    groups_detector_recovered_array[predicted_character.character_start + 1:
                                                    predicted_character.character_end] = \
                        whc_targets[TARGET_VALUE_CHARACTER_MIDDLE]
                    groups_detector_recovered_array[predicted_character.character_end] = \
                        whc_targets[TARGET_VALUE_CHARACTER_END]
                    groups_detector_recovered_array[predicted_character.character_start] = \
                        whc_targets[TARGET_VALUE_CHARACTER_START]
                print(f'groups recovered array : {groups_detector_recovered_array}')

                start_group_removed_targets_list = []
                targets_list_index = 0
                while targets_list_index < len(loaded_targets_list):
                    key = loaded_targets_list[targets_list_index]
                    start_group_removed_targets_list.append(whc_targets[key])

                    if key == TARGET_VALUE_CHARACTER_START:
                        targets_list_index += 1
                        while loaded_targets_list[targets_list_index] != TARGET_END_OF_CHARACTER_START_GROUP:
                            targets_list_index += 1
                    targets_list_index += 1
                start_group_removed_targets_array = np.array(start_group_removed_targets_list)
                print(f'targets array          : {start_group_removed_targets_array}')

            # fill predicted characters list with character lowest and highest point
            for predicted_character in predicted_characters_list:
                w_start = predicted_character.character_start - self.H_W_CUTOUT_PADDING
                if w_start < 0:
                    w_start = 0
                w_end = predicted_character.character_end + 1 + self.H_W_CUTOUT_PADDING
                if w_end > text_strip_image_to_process.shape[3]:
                    w_end = text_strip_image_to_process.shape[3]

                (predicted_character.character_lowest_point,
                 predicted_character.character_highest_point) = \
                    self.char_h_pos_detector(normalized_image_to_process[0, 0, :, w_start: w_end])

            # one by one cut out characters from strip image, run them through character
            # recognizer, fill predicted characters list with character class
            # and create image with cut out characters
            image_with_cut_out_characters = torch.nn.functional.pad(
                input=text_strip_image_to_show,
                pad=(0, 0, 0, 0, text_strip_image_to_show.shape[0], 0),
                mode='constant', value=1.0
            )

            for predicted_character in predicted_characters_list:
                w_cut_start = predicted_character.character_start - self.H_W_CUTOUT_PADDING
                if w_cut_start < 0:
                    w_cut_start = 0
                w_cut_end = predicted_character.character_end + 1 + self.H_W_CUTOUT_PADDING
                if w_cut_end > text_strip_image_to_process.shape[3]:
                    w_cut_end = text_strip_image_to_process.shape[3]
                h_cut_start = predicted_character.character_lowest_point - self.H_W_CUTOUT_PADDING
                if h_cut_start < 0:
                    h_cut_start = 0
                h_cut_end = predicted_character.character_highest_point + 1 + self.H_W_CUTOUT_PADDING
                if h_cut_end > text_strip_image_to_process.shape[2]:
                    h_cut_end = text_strip_image_to_process.shape[2]
                cut_width = w_cut_end - w_cut_start
                cut_height = h_cut_end - h_cut_start

                processed_image_h, processed_image_w = \
                    self.calculate_size_of_image_to_process(cut_height, cut_width)
                character_image_to_process = torch.zeros(
                    [1, 1, processed_image_h, processed_image_w])

                h_paste_start = (processed_image_h - cut_height) // 2
                h_paste_end = h_paste_start + cut_height
                w_paste_start = (processed_image_w - cut_width) // 2
                w_paste_end = w_paste_start + cut_width

                character_image_to_process[
                    0, 0, h_paste_start:h_paste_end, w_paste_start:w_paste_end
                ] = text_strip_image_to_process[
                    0, 0, h_cut_start:h_cut_end, w_cut_start:w_cut_end
                    ]

                # normalize image with mean and standard deviation, calculated on whole
                # recognition model dataset
                character_image_to_process -= self.IMAGE_MEAN
                character_image_to_process *= self.IMAGE_STD_RECIPROCAL

                recognizer_output = self.character_recognizer(character_image_to_process)

                character_index_prediction = recognizer_output.max(1)[1].item()
                predicted_character.character_class_index = character_index_prediction

                image_with_cut_out_characters[
                    predicted_character.character_lowest_point:
                    predicted_character.character_highest_point + 1,
                    predicted_character.character_start:
                    predicted_character.character_end + 1] = \
                    text_strip_image_to_show[
                        predicted_character.character_lowest_point:
                        predicted_character.character_highest_point + 1,
                        predicted_character.character_start:
                        predicted_character.character_end + 1]
                image_with_cut_out_characters[
                    predicted_character.character_lowest_point +
                        text_strip_image_to_show.shape[0]:
                    predicted_character.character_highest_point + 1 +
                        text_strip_image_to_show.shape[0],
                    predicted_character.character_start:
                    predicted_character.character_end + 1] = 1.0

            # calculate statistics
            total_processed_images += 1
            recognized_correctly = True
            if len(predicted_characters_list) == len(target_characters_list):
                for i in range(len(predicted_characters_list)):
                    if (predicted_characters_list[i].character_class_index !=
                            target_characters_list[i]):
                        recognized_correctly = False
                        break
            else:
                recognized_correctly = False
            if recognized_correctly:
                total_correctly_recognized_images += 1
                self.recognition_statistics[image_index] = 1
            print(f'\rprocessed {total_processed_images} images, recognized '
                  f'correctly {total_correctly_recognized_images}', end='')

            # show recognition result
            if show_images:
                self.image_ax.cla()
                self.image_ax.imshow(text_strip_image_to_show)

                self.cut_char_ax.cla()
                self.cut_char_ax.imshow(image_with_cut_out_characters)

                # add character class letters
                cut_char_ax_text_font_size = text_strip_image_to_show.shape[0] * \
                                             self.TEXT_FONT_SIZE_FACTOR
                cut_char_ax_text_shift = text_strip_image_to_show.shape[0] * \
                                         self.TEXT_SHIFT_FACTOR
                if recognized_correctly:
                    character_class_color = COLOR_GREEN_A700_STRING
                else:
                    character_class_color = COLOR_PINK_A400_STRING
                for predicted_character in predicted_characters_list:
                    self.cut_char_ax.text(
                        predicted_character.character_start,
                        - cut_char_ax_text_shift,
                        list(classes_dict.keys())[list(classes_dict.values()).index(
                            predicted_character.character_class_index)],
                        fontsize=cut_char_ax_text_font_size,
                        color=character_class_color
                    )

                # wait for user to press a key
                self.block_until_key_is_pressed()

                # release lock
                if self.lock.locked():
                    self.lock.release()

        correctly_recognized_images_part = \
            total_correctly_recognized_images / total_processed_images
        print(f'\npart of correctly recognized images: {correctly_recognized_images_part:.3f}')

        self.figure.canvas.mpl_disconnect(connection_id)

    def block_until_key_is_pressed(self):
        if self.lock.locked():
            self.lock.release()
        self.figure.canvas.start_event_loop()

        # flush all excessive pressed button events while the lock is still on
        self.figure.canvas.flush_events()

    def on_key_press_event(self, event):
        if not self.lock.locked():
            self.lock.acquire()

            if event.key == 'enter':
                self.figure.canvas.stop_event_loop()
            else:
                self.lock.release()

    def calculate_size_of_image_to_process(self, character_height, character_width):
        threshold_1x1 = self.H_W_THRESHOLD_1X1
        step_h_w = self.H_W_STEP
        if character_height <= threshold_1x1:
            tensor_h = threshold_1x1
        else:
            tensor_h = threshold_1x1 + (
                    (character_height - threshold_1x1) // step_h_w) * step_h_w
            if (character_height - threshold_1x1) % step_h_w > 0:
                tensor_h += step_h_w
        if character_width <= threshold_1x1:
            tensor_w = threshold_1x1
        else:
            tensor_w = threshold_1x1 + (
                    (character_width - threshold_1x1) // step_h_w) * step_h_w
            if (character_width - threshold_1x1) % step_h_w > 0:
                tensor_w += step_h_w
        return tensor_h, tensor_w


if __name__ == '__main__':
    np.set_printoptions(linewidth=10000)

    IMAGE_DIRECTORY_PATH = "Dataset/"

    FEATURE_EXTRACTOR_CHECKPOINT_PATH = "Checkpoints/"
    FEATURE_EXTRACTOR_CHECKPOINT_NAME = "recognizer_checkpoint.pth"

    SEGMENTER_CHECKPOINT_PATH = "Checkpoints/"
    SEGMENTER_CHECKPOINT_NAME = "segmenter_checkpoint.pth"

    image_segmenter = WHCCharSegmentationConvolutionalModel()
    cp_path_to_load = join(SEGMENTER_CHECKPOINT_PATH, SEGMENTER_CHECKPOINT_NAME)
    load_feature_extractor_checkpoint(cp_path_to_load, image_segmenter)
    image_segmenter.eval()

    character_recognizer = WHCCharRecognitionModel()
    cp_path_to_load = join(FEATURE_EXTRACTOR_CHECKPOINT_PATH, FEATURE_EXTRACTOR_CHECKPOINT_NAME)
    load_feature_extractor_checkpoint(cp_path_to_load, character_recognizer)
    character_recognizer.eval()

    target_list_converter = WHCTargetListToListOfCharacterItemsConverter()
    character_groups_detector = CharacterGroupsDetector()
    character_h_pos_detector = CharacterHeightPositionDetector()
    whc_feature_extractor = 0

    text_string_recognizer = WHCTextStringRecognizer(
        whc_feature_extractor, image_segmenter, character_recognizer,
        character_groups_detector, character_h_pos_detector, target_list_converter)

    text_string_recognizer(IMAGE_DIRECTORY_PATH, show_images=True,  shuffle_images=False,
                           print_segmentation=True)

