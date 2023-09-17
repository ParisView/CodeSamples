from os.path import join
import numpy as np
from matplotlib import image as mpimg, pyplot as plt
from threading import Lock

from imageLoading import create_lists_of_image_and_target_files

segmentation_targets = {
    "space": 0,
    "character_start": 1,
    "character_middle": 2,
    "character_end": 3
}

target_file_name_suffix = "_target.txt"
target_directory_name = "targets/"


class CharacterGroupsDetector:
    """
    This class searches through a string of targets and detects groups of targets, that correspond
    to a character. The string of targets must be an output of a classification neural network,
    that takes a picture of a string of characters and for each column of pixels of this picture
    outputs a classification value , classifying it as a space, character_start, character_middle
    or character_end.
    """

    def __init__(self):
        # creating a figure
        plt.ion()
        self.figure = plt.figure(figsize=(15.0, 8.0))

        # connecting a callback for a key press event
        self.connection_id = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press_event)

        # creating a lock
        self.lock = Lock()

        # creating targets to values conversion dictionary
        self.targets_to_values_dict = {
            "space": 1,
            "character_start": 3,
            "character_middle": 2,
            "character_end": 4
        }

        # setting constants
        self.MAX_ALLOWED_ERRORS_IN_GROUP = 2
        self.MAX_NEXT_TARGETS_TO_CHECK = self.MAX_ALLOWED_ERRORS_IN_GROUP + 1
        self.MAX_END_SEARCH_ROUNDS = 3
        self.START_SHIFT_DUE_TO_CONVOLUTION = 2  # shift due to convolution kernel half width
        self.SPACE = segmentation_targets["space"]
        self.CHAR_START = segmentation_targets["character_start"]
        self.CHAR_MIDDLE = segmentation_targets["character_middle"]
        self.CHAR_END = segmentation_targets["character_end"]
        self.END_OF_SEQUENCE = self.find_available_int(np.array([
            self.SPACE, self.CHAR_START, self.CHAR_MIDDLE, self.CHAR_END]))

        # creating variables
        self.target_index = 0
        self.error_count = 0
        self.start_position_of_detected_group = 0
        self.end_position_of_detected_group = 0
        self.target_array = np.zeros(1)

    def detect_character_groups_for_images_in_directory(self, image_directory_path):
        target_directory_path = join(image_directory_path, target_directory_name)

        # creating a list of image files and a list of target files
        list_of_image_files, list_of_target_files = create_lists_of_image_and_target_files(
            image_directory_path, target_directory_path, target_file_name_suffix
        )

        # one by one loading target data to a numpy array from the list of targets
        for image_index in range(len(list_of_target_files)):
            # load the content of the image file
            image_file = ''.join([image_directory_path, list_of_image_files[image_index]])
            print(f'processing: {image_file}')
            image_array = np.array(mpimg.imread(image_file))

            # load the content of the target file into an array [height (1), width(~700)]
            target_file = ''.join([target_directory_path, list_of_target_files[image_index]])
            target_file = open(target_file, 'r', encoding='utf8')
            target_content_list = [string.strip() for string in target_file]
            target_file.close()

            # convert strings of target names to values
            target_array = np.array([segmentation_targets[item] for item in target_content_list])

            # detect character groups in target_array and display result in self.figure
            self.detect_character_groups(image_array, target_array)

            # start figure event loop and wait for user to press a key to stop the event loop in
            # key press callback
            self.start_event_loop_and_wait_for_key_press()

    def detect_character_groups(self, image_array, target_array,
                                padding=self.START_SHIFT_DUE_TO_CONVOLUTION):
        self.target_array = target_array
        list_of_values = list(segmentation_targets.values())
        list_of_keys = list(segmentation_targets.keys())

        # for plotting purposes add self.START_SHIFT_DUE_TO_CONVOLUTION elements at the beginning
        # and at the end of the array due to the shift, produced by convolution kernel half width
        padding_list = [0] * padding
        target_plot_array = np.array(padding_list + [self.targets_to_values_dict[list_of_keys[
            list_of_values.index(item)]] for item in target_array] + padding_list)
        image_height = image_array.shape[0]
        detected_char_array = np.zeros((image_height * 2, image_array.shape[1], image_array.shape[2]))

        # set parameters of the figure
        self.figure.clf()
        spec = self.figure.add_gridspec(4, 1)
        ax1 = self.figure.add_subplot(spec[0, 0])
        ax2 = self.figure.add_subplot(spec[1, 0])
        ax3 = self.figure.add_subplot(spec[2, 0])
        ax2_y_margin = 0.5

        # plot the image content
        ax1.imshow(image_array)

        # plot the target content
        ax2.stairs(target_plot_array, fill=True, color='#ff8f80')
        ax2.axis([0, len(target_plot_array), - ax2_y_margin,
                  max(self.targets_to_values_dict.values()) + ax2_y_margin])

        # detect character groups in array of targets
        character_positions_list = self.detect_character_groups_in_target_array()

        # if character_positions_list is not empty display detected characters on the figure
        if character_positions_list:
            self.copy_image_content_to_detected_char_array(
                detected_char_array, image_array, 0, 0,
                character_positions_list[0][0] + self.START_SHIFT_DUE_TO_CONVOLUTION)
            for character_position_index in range(len(character_positions_list)):
                self.copy_image_content_to_detected_char_array(
                    detected_char_array, image_array, image_height,
                    character_positions_list[character_position_index][0] + self.START_SHIFT_DUE_TO_CONVOLUTION,
                    character_positions_list[character_position_index][1] + 1 + self.START_SHIFT_DUE_TO_CONVOLUTION)
                x_stop = (character_positions_list[character_position_index + 1][0] +
                          self.START_SHIFT_DUE_TO_CONVOLUTION
                          if character_position_index + 1 < len(character_positions_list)
                          else image_array.shape[1])
                self.copy_image_content_to_detected_char_array(
                    detected_char_array, image_array, 0,
                    character_positions_list[character_position_index][1] + 1 + self.START_SHIFT_DUE_TO_CONVOLUTION,
                    x_stop)

        # plot the detected characters array
        ax3.imshow(detected_char_array)

    def copy_image_content_to_detected_char_array(self, detected_char_array, image_content, y_shift,
                                                  x_start, x_stop):
        y_start = y_shift
        y_stop = y_start + image_content.shape[0]
        detected_char_array[y_start: y_stop, x_start: x_stop, :] = image_content[:, x_start: x_stop, :]

    def detect_character_groups_in_target_array(self):
        self.target_index = 0
        character_positions_list = list()
        while self.target_index < self.target_array.shape[0]:
            character_group_detected = self.detect_next_character_group()
            if character_group_detected:
                # add character group start and end to the character_positions_list
                character_positions_list.append((self.start_position_of_detected_group,
                                                 self.end_position_of_detected_group))
        return character_positions_list

    def detect_next_character_group(self):
        next_target = self.read_target(self.target_index)
        index_increment = 0
        self.error_count = 0
        character_group_detected = False

        if next_target == self.SPACE:
            self.target_index += self.sort_multiple_encounters(self.SPACE)
            next_target = self.read_target(self.target_index)

        if next_target == self.CHAR_END:
            self.target_index += self.sort_multiple_encounters(self.CHAR_END)
            next_target = self.read_target(self.target_index)

        if next_target == self.CHAR_START:
            middle_found, target_index_increment = self.find_next(self.CHAR_MIDDLE)

            if middle_found:
                self.error_count += target_index_increment - 1
                index_increment += self.sort_multiple_encounters(self.CHAR_MIDDLE, target_index_increment)
                character_group_detected = self.find_group_end(index_increment)
            else:
                self.target_index += 1

        elif next_target == self.CHAR_MIDDLE:
            self.error_count += 1
            index_increment += self.sort_multiple_encounters(self.CHAR_MIDDLE)
            character_group_detected = self.find_group_end(index_increment)

        return character_group_detected

    def find_group_end(self, index_increment):
        found_group_end = False
        end_search_round = 0
        while end_search_round < self.MAX_END_SEARCH_ROUNDS:
            next_target = self.read_target(self.target_index + index_increment + end_search_round)
            if next_target == self.CHAR_END:
                # at this point a character group is detected,
                # write character start and end positions to corresponding parameters
                self.start_position_of_detected_group = self.target_index
                self.end_position_of_detected_group = self.target_index + index_increment + end_search_round
                self.target_index += index_increment + end_search_round + 1
                # stop the search
                end_search_round = self.MAX_END_SEARCH_ROUNDS
                found_group_end = True
            else:
                # increment error counter
                self.error_count += 1
                if self.error_count > self.MAX_ALLOWED_ERRORS_IN_GROUP:
                    # too many errors by this point, deciding either group is detected or
                    # move to the next target index
                    if end_search_round > 0:
                        # at this point a character group is detected,
                        # write character start and end positions to corresponding parameters
                        self.start_position_of_detected_group = self.target_index
                        self.end_position_of_detected_group = self.target_index + index_increment - 1
                        self.target_index += index_increment
                        # stop the search
                        end_search_round = self.MAX_END_SEARCH_ROUNDS
                        found_group_end = True
                    else:
                        # stop the search and increment target index
                        end_search_round = self.MAX_END_SEARCH_ROUNDS
                        self.target_index += 1
                else:
                    # increment end search round and continue the search
                    end_search_round += 1
        return found_group_end

    def find_next(self, target):
        """
        this function searches for the specified target in the next max_next_targets_to_check positions
        of the target_array
        """
        # index_increment is returned as the position at which the specified target was
        # found (equal to the number of errors to add)
        index_increment = 1
        continue_search = True
        found_specified_target = False

        while continue_search:
            if self.target_index + index_increment < self.target_array.shape[0]:  # array index within array length
                if self.target_array[self.target_index + index_increment] == target:  # found the specified target
                    continue_search = False
                    found_specified_target = True
                else:  # did not find the specified target
                    # if checked less than max_next_targets_to_check targets, then continue the search
                    if index_increment < self.MAX_NEXT_TARGETS_TO_CHECK - 1:
                        index_increment += 1
                    else:
                        continue_search = False
            else:
                continue_search = False

        return found_specified_target, index_increment

    def read_target(self, target_index):
        return self.target_array[target_index] if target_index < self.target_array.shape[0] \
            else self.END_OF_SEQUENCE

    def sort_multiple_encounters(self, target, index_increment=0):
        while self.target_index + index_increment < self.target_array.shape[0] and \
                self.target_array[self.target_index + index_increment] == target:
            index_increment += 1
        return index_increment

    def find_available_int(self, int_array):
        int_array_index = 0
        available_number = 0
        continue_search = True
        while continue_search:
            continue_search = False
            available_number = int_array[int_array_index] + 1
            for i in range(int_array.shape[0]):
                if available_number == int_array[i]:
                    continue_search = True
                    break
            int_array_index += 1
        return available_number

    def start_event_loop_and_wait_for_key_press(self):
        if self.lock.locked():
            self.lock.release()
        self.figure.canvas.start_event_loop()
        # this flush_events aims to flush all excessive pressed button
        # events while the lock is still on
        self.figure.canvas.flush_events()

    def on_key_press_event(self, event):
        if not self.lock.locked():
            self.lock.acquire()
            self.figure.canvas.stop_event_loop()


if __name__ == '__main__':
    IMAGE_DIRECTORY_PATH = "Dataset/"
    character_groups_detector = CharacterGroupsDetector()
    character_groups_detector.detect_character_groups_for_images_in_directory(IMAGE_DIRECTORY_PATH)
