from os import listdir
from os.path import isfile, join
from threading import Lock

import torch
from matplotlib import image as mpimg, pyplot as plt
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.patches import Rectangle
import matplotlib as mpl

# pressed key values
PRESSED_KEY_VALUE_SPACE = "space"
PRESSED_KEY_VALUE_CHARACTER_START = "character_start"
PRESSED_KEY_VALUE_CHARACTER_MIDDLE = "character_middle"
PRESSED_KEY_VALUE_CHARACTER_END = "character_end"
PRESSED_KEY_VALUE_WRONG_BUTTON = "wrong button"
PRESSED_KEY_VALUE_SAVING_FILE = "saving file"
PRESSED_KEY_VALUE_DELETE_PREVIOUS = "delete previous"
PRESSED_KEY_VALUE_DELETE_PREVIOUS_GROUP = "delete previous group"
PRESSED_KEY_VALUE_DELETE_MULTIPLE_GROUPS = "delete multiple groups"
PRESSED_KEY_VALUE_MOVE_UP = "move up"
PRESSED_KEY_VALUE_MOVE_DOWN = "move down"
PRESSED_KEY_VALUE_ENTER = "enter"
PRESSED_KEY_VALUE_ESCAPE = 'escape'
END_OF_CHARACTER_START_GROUP = 'end_of_character_start_group'
PRESSED_KEY_VALUE_AUTO_MARK_CHARACTER = "auto mark character"
PRESSED_KEY_VALUE_CHARACTER_CLASS = "entered character CLASS"

# whc (Width-Height-Class) targets dictionary
whc_targets = {
    PRESSED_KEY_VALUE_SPACE: 0,
    PRESSED_KEY_VALUE_CHARACTER_START: 1,
    PRESSED_KEY_VALUE_CHARACTER_MIDDLE: 2,
    PRESSED_KEY_VALUE_CHARACTER_END: 3
}

# character classes dictionary
character_classes_dict = {
    'a': 'A',
    'b': 'B',
    'c': 'C',
    'd': 'D',
    'e': 'E',
    'f': 'F',
    'g': 'G',
    'h': 'H',
    'i': 'I',
    'j': 'J',
    'k': 'K',
    'l': 'L',
    'm': 'M',
    'n': 'N',
    'o': 'O',
    'p': 'P',
    'q': 'Q',
    'r': 'R',
    's': 'S',
    't': 'T',
    'u': 'U',
    'v': 'V',
    'w': 'W',
    'x': 'X',
    'y': 'Y',
    'z': 'Z',

    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',

    ',': '<',
}

# character start group parameters
CHARACTER_LOWEST_POINT = 'lowest_point'
CHARACTER_HIGHEST_POINT = 'highest_point'
CHARACTER_CLASS = 'character_class'

# list of character start group parameters
start_group_parameters_list = [CHARACTER_LOWEST_POINT, CHARACTER_HIGHEST_POINT, CHARACTER_CLASS]

# whc targets directory name and file name suffix
whc_target_file_name_suffix = "_whctgt.txt"
whc_target_directory_name = "whc-targets/"

# colors
COLOR_BLUE_A400_STRING = '#2979FF'
COLOR_TEAL_A200_STRING = '#64FFDA'
COLOR_ORANGE_800_STRING = '#EF6C00'
COLOR_GREEN_800_STRING = '#2E7D32'
COLOR_GREEN_A700_STRING = '#00C853'
COLOR_LIGHT_GREEN_A400_STRING = '#76FF03'
COLOR_PINK_A400_STRING = '#F50057'
COLOR_PLAIN_BLACK_STRING = '#000000'
COLOR_PLAIN_GRAY_STRING = '#808080'
COLOR_PLAIN_RED_STRING = '#800000'
COLOR_PLAIN_WHITE_TENSOR = torch.tensor([1, 1, 1, 1])
COLOR_PLAIN_GRAY_TENSOR = torch.tensor([0.5, 0.5, 0.5, 1])
COLOR_PLAIN_RED_TENSOR = torch.tensor([1, 0, 0, 1])
COLOR_PLAIN_GREEN_TENSOR = torch.tensor([0, 1, 0, 1])
COLOR_PLAIN_BLUE_TENSOR = torch.tensor([0, 0, 1, 1])
COLOR_PINK_200_TENSOR = torch.tensor([0.953, 0.559, 0.691, 1.0])  # #F48FB1
COLOR_PINK_400_TENSOR = torch.tensor([0.922, 0.250, 0.477, 1.0])  # #EC407A
COLOR_LIME_200_TENSOR = torch.tensor([0.898, 0.930, 0.609, 1.0])  # #E6EE9C
COLOR_LIME_500_TENSOR = torch.tensor([0.801, 0.859, 0.223, 1.0])  # #CDDC39
COLOR_LIME_A200_TENSOR = torch.tensor([0.929, 1.0, 0.254, 1.0])  # #EEFF41
COLOR_GREEN_800_TENSOR = torch.tensor([0.180, 0.488, 0.195, 1.0])  # #2E7D32
COLOR_LIGHT_GREEN_400_TENSOR = torch.tensor([0.609, 0.797, 0.395, 1.0])  # #9CCC65
COLOR_BLUE_100_TENSOR = torch.tensor([0.730, 0.867, 0.980, 1.0])  # #BBDEFB
COLOR_BLUE_200_TENSOR = torch.tensor([0.563, 0.789, 0.973, 1.0])  # #90CAF9
COLOR_BLUE_A400_TENSOR = torch.tensor([0.16, 0.473, 1.0, 1.0])  # #2979FF
COLOR_LIGHT_GREEN_50_TENSOR = torch.tensor([0.941, 0.969, 0.910, 1.0])  # #F1F8E9

# verification colors factor dictionary, [Red, Green, Blue, Alpha]
BACKGROUND_COLOR = "background color"
HIGHEST_AND_LOWEST_LEVEL_COLOR = "highest and lowest level color"
NEGATIVE_LOWEST_AND_HIGHEST_LEVEL_COLOR_TENSOR = "negative highest and lowest level color"
verification_image_and_mark_color_factors = {
    PRESSED_KEY_VALUE_SPACE: [COLOR_PLAIN_WHITE_TENSOR, COLOR_PLAIN_GRAY_TENSOR],
    PRESSED_KEY_VALUE_CHARACTER_START: [COLOR_LIGHT_GREEN_400_TENSOR, COLOR_GREEN_800_TENSOR],
    PRESSED_KEY_VALUE_CHARACTER_MIDDLE: [COLOR_BLUE_100_TENSOR, COLOR_BLUE_A400_TENSOR],
    PRESSED_KEY_VALUE_CHARACTER_END: [COLOR_PINK_200_TENSOR, COLOR_PLAIN_RED_TENSOR],
    PRESSED_KEY_VALUE_DELETE_PREVIOUS: [COLOR_LIME_200_TENSOR, COLOR_LIME_500_TENSOR],
    BACKGROUND_COLOR: [COLOR_LIGHT_GREEN_50_TENSOR, COLOR_LIGHT_GREEN_50_TENSOR],
    NEGATIVE_LOWEST_AND_HIGHEST_LEVEL_COLOR_TENSOR: [torch.tensor([1, 1, 1, 2]) - COLOR_LIME_A200_TENSOR,
                                                     torch.tensor([1, 1, 1, 2]) - COLOR_LIME_A200_TENSOR]
}

# verification shift and height dictionary
verification_start_and_end = {
    PRESSED_KEY_VALUE_SPACE: [0.2, 0.3],
    PRESSED_KEY_VALUE_CHARACTER_START: [0.0, 1.0],
    PRESSED_KEY_VALUE_CHARACTER_MIDDLE: [0.1, 0.7],
    PRESSED_KEY_VALUE_CHARACTER_END: [0.0, 0.9]
}

# character class verification parameters (fraction of image height)
CHARACTER_CLASS_VERIFICATION_TEXT_FONT_SIZE_FACTOR = 0.5
CHARACTER_CLASS_VERIFICATION_TEXT_SHIFT_FACTOR = 0.1
COORDINATE_X_KEY = 'coordinate_x'
CHARACTER_CLASS_KEY = 'character_class'

# automatic marking parameters
AUTO_MARKING_MIN_THRESHOLD = 0.0
AUTO_MARKING_MAX_THRESHOLD = 1.0
AUTO_MARKING_DEFAULT_THRESHOLD = 0.46
AUTO_MARKING_THRESHOLD_INCREMENT = 0.01
BRIGHTNESS_VIEW_HEIGHT = 100
MIN_N_OF_MIDDLE_TARGETS_IN_CHARACTER = 2
ROOT_SEGMENT_ID = 11
MAX_GAP_BETWEEN_SEGMENTS = 2
ENTER_CHARACTER_CLASS_WARNING = "Enter Character Class"
NO_CLASSES_ENTERED_WARNING_START_COORDINATE_X = 0
NO_CLASSES_ENTERED_WARNING = "No Character Classes Entered Yet"
NEXT_CHARACTER_TEXT_COORDINATE_X = 0
NEXT_CHARACTER_TEXT = "Next Character"
NO_NEXT_CHARACTER_WARNING_COORDINATE_X = 0
NO_NEXT_CHARACTER_WARNING = "No Next Character Available"
DETECTED_TOO_NARROW_CHARACTER = "Detected Too Narrow Character"
CHARACTER_IS_OUT_OF_CURRENT_SCOPE = "Character Is Out Of Current Scope"
CANT_MARK_CHARACTER_TOO_WIDE = "Can't Mark. Character Is Too Wide"
REACHED_END_OF_IMAGE = "Reached End Of Image"


class Segment:
    def __init__(self, segment_id):
        self.id = segment_id
        self.start = 0
        self.end = 0
        self.width = 0
        self.min_h = 0
        self.max_h = 0
        self.connections = []


class WHCTargetMarker:
    """
    WHCTargetMarker is a character width, height and class target marker.
    It helps manually create target files for images of text strips for Machine Learning applications of character
    segmentation and recognition. It helps to mark positions of characters in an image of a text strip, their widths,
    heights and classes, to which these characters belong (the class, to which a character belongs, is the character,
    that is presented in the marked area of the image)
    """

    def __init__(self, image_directory_path):
        self.image_directory_path = image_directory_path
        self.target_directory_path = join(self.image_directory_path, whc_target_directory_name)
        self.list_of_target_files = [f[:-11] for f in listdir(self.target_directory_path) if
                                     isfile(join(self.target_directory_path, f))]
        self.list_of_image_files = [f for f in listdir(self.image_directory_path) if
                                    (isfile(join(self.image_directory_path, f)) and
                                     (f[:-4] not in self.list_of_target_files))]
        self.strip_of_targets_list = []
        self.image_sector_width = 61  # must be an odd integer
        self.marker_position_in_image_sector = 10
        self.draft_sector_text_shift = 0
        self.draft_sector_text_font_size = 0
        self.mark_position = 0
        self.mark_verification = torch.zeros([0])
        self.mark_verification_padding_start = 0
        self.mark_verification_padding_width = 0
        self.mark_verification_text_font_size = 0
        self.mark_verification_text_shift = 0
        self.string_of_character_classes_list = []
        self.character_class_index = 0

        self.verification_samples_dict = {
            PRESSED_KEY_VALUE_SPACE: torch.tensor([0]),
            PRESSED_KEY_VALUE_CHARACTER_START: torch.tensor([0]),
            PRESSED_KEY_VALUE_CHARACTER_MIDDLE: torch.tensor([0]),
            PRESSED_KEY_VALUE_CHARACTER_END: torch.tensor([0]),
            PRESSED_KEY_VALUE_DELETE_PREVIOUS: torch.tensor([0])
        }
        self.image = torch.zeros([0])
        self.lock = Lock()
        self.save_file = False
        self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
        self.start_group_parameters_list_index = 0
        self.height_marker_index_low_limit = 0
        self.height_marker_index_high_limit = 0
        self.character_lowest_point_index = 0
        self.character_highest_point_index = 0
        self.mark_lowest_and_highest_level = False
        self.height_marker_index = 0
        self.auto_marking_threshold = AUTO_MARKING_DEFAULT_THRESHOLD
        self.pressed_escape_in_auto_mark_mode = False
        self.auto_marked_spaces_list = []
        self.auto_marked_start_group_list = []
        self.auto_marked_character_middle_list = []
        self.auto_marked_character_end_list = []

        plt.ion()
        self.figure = plt.figure(figsize=(15.0, 9.0), constrained_layout=True)
        self.figure.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
        spec = self.figure.add_gridspec(29, 3)
        self.image_ax = self.figure.add_subplot(spec[0: 5, :])
        self.verification_ax = self.figure.add_subplot(spec[6: 13, :])
        self.mark_draft_sector_ax = self.figure.add_subplot(spec[14: 19, :])
        self.brightness_sector_ax = self.figure.add_subplot(spec[19: 24, :])
        self.image_sector_ax = self.figure.add_subplot(spec[24: 29, :])
        self.image_aspect_ratio = 1
        self.verification_aspect_ratio = 1
        self.image_sector_aspect_ratio = 0.2
        self.image_sector_brightness_view_aspect_ratio = 1

        # delete all matplotlib keyboard shortcuts to be able to use keys for target marking without
        # interference of other matplotlib functions
        mpl.rcParams["keymap.home"] = ''
        mpl.rcParams["keymap.back"] = ''
        mpl.rcParams["keymap.forward"] = ''
        mpl.rcParams["keymap.pan"] = ''
        mpl.rcParams["keymap.zoom"] = ''
        mpl.rcParams["keymap.save"] = ''
        mpl.rcParams["keymap.fullscreen"] = ''
        mpl.rcParams["keymap.grid"] = ''
        mpl.rcParams["keymap.grid_minor"] = ''
        mpl.rcParams["keymap.xscale"] = ''
        mpl.rcParams["keymap.yscale"] = ''
        mpl.rcParams["keymap.quit"] = ''

        # print all used keys
        # 'backspace' 'enter' 'delete' 'end' 'pagedown' 'insert' 'pageup'
        # 'up' 'down' '-' 'escape' '='
        print()
        print("to mark target as a SPACE press:                    'space'")
        print("to mark target as a CHARACTER START press:          'delete'")
        print("to mark target as a CHARACTER MIDDLE press:         'end'")
        print("to mark target as a CHARACTER END press:            'pagedown'")
        print("to mark LOWEST and HIGHEST character points use:    'up', 'down' and 'enter'")
        print("to mark a character CLASS press:                     letter or digit or ','( for <)")
        print("to DELETE PREVIOUS marked target press:             'backspace'")
        print("to automatically mark SINGLE CHARACTER press:       'insert'")
        print("to automatically mark MULTIPLE CHARACTERS press:    'pageup'")
        print("to DELETE previous SINGLE character press:          '='")
        print("to DELETE previous MULTIPLE characters press:       '-'")
        print("to SWITCH from single characters TO TARGETS press:  'escape'")
        print('\n')

    def mark_targets(self):
        print(f"found {len(self.list_of_image_files)} files to mark targets in")
        print(self.image_directory_path)
        connection_id = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press_event)

        for file_name in self.list_of_image_files:
            # load image
            full_file_path = join(self.image_directory_path, file_name)
            self.image = torch.tensor(mpimg.imread(full_file_path))

            # create mark_verification image
            self.mark_verification_padding_start = self.image.shape[0]
            self.mark_verification_padding_width = self.image.shape[0]
            self.mark_verification_text_font_size = \
                self.image.shape[0] * CHARACTER_CLASS_VERIFICATION_TEXT_FONT_SIZE_FACTOR
            self.mark_verification_text_shift = \
                self.image.shape[0] * CHARACTER_CLASS_VERIFICATION_TEXT_SHIFT_FACTOR
            self.draft_sector_text_shift = self.mark_verification_text_shift
            self.draft_sector_text_font_size = self.mark_verification_text_font_size
            self.mark_verification = torch.nn.functional.pad(input=self.image, pad=(
                0, 0, 0, 0, 0, self.mark_verification_padding_width), mode='constant', value=0)

            # create verification samples for mark_verification image
            for key in verification_start_and_end.keys():
                mark_start = int(self.mark_verification_padding_width * verification_start_and_end[key][0])
                mark_end = int(self.mark_verification_padding_width * verification_start_and_end[key][1])
                self.verification_samples_dict[key] = torch.ones([self.mark_verification_padding_width, 4])
                if mark_start > 0:
                    self.verification_samples_dict[key][0: mark_start] *= \
                        verification_image_and_mark_color_factors[BACKGROUND_COLOR][1]
                elif mark_start < 0:
                    mark_start = 0
                if mark_end < self.mark_verification_padding_width:
                    self.verification_samples_dict[key][mark_end: self.mark_verification_padding_width] *= \
                        verification_image_and_mark_color_factors[BACKGROUND_COLOR][1]
                elif mark_end > self.mark_verification_padding_width:
                    mark_end = self.mark_verification_padding_width
                self.verification_samples_dict[key][mark_start: mark_end] *= \
                    verification_image_and_mark_color_factors[key][1]

            self.verification_samples_dict[PRESSED_KEY_VALUE_DELETE_PREVIOUS] = torch.ones(
                [self.mark_verification_padding_width, 4]) * verification_image_and_mark_color_factors[
                PRESSED_KEY_VALUE_DELETE_PREVIOUS][1]

            # fill mark_verification image with default samples
            self.mark_verification[self.mark_verification_padding_start:, :] = \
                self.verification_samples_dict[PRESSED_KEY_VALUE_DELETE_PREVIOUS][None, :].permute(1, 0, 2)

            # show image and set up parameters
            print(f"\nloaded image: {full_file_path} with size {self.image.shape}")
            self.image_ax.cla()
            self.image_ax.imshow(self.image, aspect=self.image_aspect_ratio)
            self.strip_of_targets_list = []
            self.mark_position = 0
            self.character_class_index = 0
            if self.string_of_character_classes_list:
                self.string_of_character_classes_list[self.character_class_index][COORDINATE_X_KEY] = 0
                self.rearrange_characters_in_string_of_character_classes_list()
            self.save_file = False
            self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON

            # one by one mark targets from the beginning of the loaded image to its end
            while not self.mark_position_reached_image_end():
                image_sector, rectangle_marker = self.marking_loop_step()
                print(f"mark position: {self.mark_position}")

                if self.pressed_key_value == PRESSED_KEY_VALUE_CHARACTER_START:
                    self.figure.canvas.mpl_disconnect(connection_id)
                    connection_id = self.figure.canvas.mpl_connect(
                        'key_press_event', self.on_key_press_event_when_marking_start_group)
                    self.mark_start_group(image_sector, rectangle_marker)
                    self.figure.canvas.mpl_disconnect(connection_id)
                    connection_id = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press_event)
                elif self.pressed_key_value == PRESSED_KEY_VALUE_AUTO_MARK_CHARACTER:
                    self.figure.canvas.mpl_disconnect(connection_id)
                    connection_id = self.figure.canvas.mpl_connect(
                        'key_press_event', self.on_key_press_event_when_auto_marking_character)
                    self.auto_mark_character()
                    self.figure.canvas.mpl_disconnect(connection_id)
                    connection_id = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press_event)

                # when the last target is marked, show the result and wait for a key to be pressed to
                # select if the target file should be saved,
                # or alternatively a "delete previous" or "delete previous group" could be pressed to
                # delete previously assigned targets
                if self.mark_position_reached_image_end():
                    self.figure.canvas.mpl_disconnect(connection_id)
                    connection_id = self.figure.canvas.mpl_connect(
                        'key_press_event', self.on_key_press_event_when_reached_image_end)
                    print('all targets are marked')
                    self.marking_loop_step()
                    self.figure.canvas.mpl_disconnect(connection_id)
                    connection_id = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press_event)

            # release lock
            if self.lock.locked():
                self.lock.release()

            # save the list of targets to a file
            if self.save_file:
                # save file
                file_path_components_list = [self.target_directory_path, file_name[:-4], whc_target_file_name_suffix]
                file_path = ''.join(file_path_components_list)

                output_file = open(file_path, 'w', encoding='utf8')
                for target in self.strip_of_targets_list:
                    print(target, file=output_file)
                output_file.flush()
                output_file.close()
                print(f"saved target file: {file_path}")

        self.figure.canvas.mpl_disconnect(connection_id)
        print("all image files are marked with their strips of targets")

    def auto_mark_character(self):
        while not self.exit_auto_mark_character_mode():
            self.auto_mark_character_loop_step()

    def auto_mark_character_loop_step(self):
        image_sector, image_sector_start = self.calculate_image_sector()
        self.update_mark_verification_on_verification_ax()
        self.image_sector_ax.cla()
        self.image_sector_ax.imshow(image_sector, aspect=self.image_sector_aspect_ratio)
        self.mark_draft_sector_ax.cla()
        self.mark_draft_sector_ax.imshow(image_sector, aspect=self.image_sector_aspect_ratio)
        image_sector_brightness_view = self.calculate_image_sector_brightness_view(image_sector)
        self.update_image_sector_brightness_view_on_brightness_sector_ax(image_sector_brightness_view)

        self.calculate_auto_marked_lists(image_sector, image_sector_start, show_text_and_markers=True)

        # waiting for user to input a selection of the next action in the process of automatic marking
        self.block_until_key_is_pressed()

    def auto_mark_multiple_characters_and_show_verification(self):
        mark_next_character = True
        while self.mark_position < self.image.shape[1] and mark_next_character:
            # calculate image sector
            image_sector, image_sector_start = self.calculate_image_sector()
            # auto mark next character
            self.calculate_auto_marked_lists(image_sector, image_sector_start)

            if self.auto_marked_start_group_list:
                if self.character_class_index < len(self.string_of_character_classes_list):
                    # accept all marking suggestions for the current character
                    self.accept_auto_marking_calculations()
                    self.add_auto_marked_targets()
                else:
                    # if character is marked but character class is not entered, stop marking
                    mark_next_character = False
            else:
                # accept all marking suggestions for the current character
                self.add_auto_marked_targets()
        self.update_mark_verification_on_verification_ax()

    def accept_auto_marking_calculations(self):
        self.string_of_character_classes_list[self.character_class_index][COORDINATE_X_KEY] = \
            len(self.auto_marked_spaces_list) + self.mark_position
        self.rearrange_characters_in_string_of_character_classes_list()
        self.auto_marked_start_group_list.append(
            self.string_of_character_classes_list[self.character_class_index][CHARACTER_CLASS_KEY])
        self.auto_marked_start_group_list.append(END_OF_CHARACTER_START_GROUP)
        self.character_class_index += 1

    def on_key_press_event_when_auto_marking_character(self, event):
        if not self.lock.locked():
            self.lock.acquire()
            print_threshold = False
            print_class = False
            print_no_class_needed = False
            print_added_target_group = False
            print_enter_class = False
            if event.key == 'insert':
                self.pressed_key_value = PRESSED_KEY_VALUE_AUTO_MARK_CHARACTER
                if self.auto_marked_start_group_list:
                    if self.character_class_index < len(self.string_of_character_classes_list):
                        # accept all marking suggestions for the current character and auto mark next character
                        self.accept_auto_marking_calculations()
                        self.add_auto_marked_targets()
                        print_added_target_group = True
                    else:
                        # if character is marked but character class is not entered, request entering character class
                        self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
                        print_enter_class = True
                else:
                    # accept all marking suggestions for the current character and auto mark next character
                    self.add_auto_marked_targets()
                    print_added_target_group = True
                self.figure.canvas.stop_event_loop()
            elif event.key in character_classes_dict:
                if self.auto_marked_start_group_list:
                    self.pressed_key_value = PRESSED_KEY_VALUE_CHARACTER_CLASS
                    print_class = True
                    character_classes_list_item = {
                        COORDINATE_X_KEY: len(self.auto_marked_spaces_list) + self.mark_position,
                        CHARACTER_CLASS_KEY: character_classes_dict[event.key]
                    }
                    if self.character_class_index < len(self.string_of_character_classes_list):
                        self.string_of_character_classes_list[self.character_class_index] = \
                            character_classes_list_item
                        self.rearrange_characters_in_string_of_character_classes_list()
                    else:
                        self.string_of_character_classes_list.append(character_classes_list_item)
                    self.figure.canvas.stop_event_loop()
                else:
                    self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
                    print_no_class_needed = True
                    self.lock.release()
            elif event.key == 'up':
                self.pressed_key_value = PRESSED_KEY_VALUE_MOVE_UP
                if self.auto_marking_threshold + AUTO_MARKING_THRESHOLD_INCREMENT <= AUTO_MARKING_MAX_THRESHOLD:
                    self.auto_marking_threshold += AUTO_MARKING_THRESHOLD_INCREMENT
                print_threshold = True
                self.figure.canvas.stop_event_loop()
            elif event.key == 'down':
                self.pressed_key_value = PRESSED_KEY_VALUE_MOVE_DOWN
                if self.auto_marking_threshold - AUTO_MARKING_THRESHOLD_INCREMENT >= AUTO_MARKING_MIN_THRESHOLD:
                    self.auto_marking_threshold -= AUTO_MARKING_THRESHOLD_INCREMENT
                print_threshold = True
                self.figure.canvas.stop_event_loop()
            elif event.key == 'enter':
                self.pressed_key_value = PRESSED_KEY_VALUE_ENTER
                self.auto_marked_start_group_list = []
                self.auto_marked_character_middle_list = []
                self.auto_marked_character_end_list = []
                self.add_auto_marked_targets()
                print_added_target_group = True
                self.pressed_escape_in_auto_mark_mode = True
                self.figure.canvas.stop_event_loop()
            elif event.key == 'escape':
                self.pressed_key_value = PRESSED_KEY_VALUE_ESCAPE
                self.pressed_escape_in_auto_mark_mode = True
                self.figure.canvas.stop_event_loop()
            else:
                self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
                self.lock.release()
            print(f"pressed key: '{event.key}', interpretation: {self.pressed_key_value}", end='')
            if print_threshold:
                print(f", auto marking threshold: {self.auto_marking_threshold:.2f}")
            elif print_class:
                print(f", class: "
                      f"{self.string_of_character_classes_list[self.character_class_index][CHARACTER_CLASS_KEY]}")
            elif print_added_target_group:
                print(f", added target group: "
                      f"spaces: {len(self.auto_marked_spaces_list)}, "
                      f"start group: {len(self.auto_marked_start_group_list)}, "
                      f"middles: {len(self.auto_marked_character_middle_list)}, "
                      f"end: {len(self.auto_marked_character_end_list)}", end='')
                if self.auto_marked_start_group_list:
                    print(f", character class: {self.auto_marked_start_group_list[3]}, ")
                else:
                    print('')
            elif print_enter_class:
                print(". No class entered. Enter character class first")
            elif print_no_class_needed:
                print(". No character group currently marked. No class needed.")
            else:
                print('')

    def calculate_auto_marked_lists(self, image_sector, image_sector_start, show_text_and_markers=False):
        space_marker_y_start = image_sector.shape[0] * 0.4
        space_marker_y_length = image_sector.shape[0] * 0.2

        auto_mark_start_position_in_sector = self.mark_position - image_sector_start
        auto_mark_position_in_sector = auto_mark_start_position_in_sector
        self.auto_marked_spaces_list = []
        self.auto_marked_start_group_list = []
        self.auto_marked_character_middle_list = []
        self.auto_marked_character_end_list = []
        character_segments_list = []
        character_min_height_index = image_sector.shape[0] - 1
        character_max_height_index = 0

        # mark all spaces
        while (
                not self.reached_sector_end(auto_mark_position_in_sector, image_sector.shape[1])
        ) and (
                image_sector[:, auto_mark_position_in_sector, 0].min(0)[0] >= self.auto_marking_threshold
        ):
            self.auto_marked_spaces_list.append(PRESSED_KEY_VALUE_SPACE)
            auto_mark_position_in_sector += 1

        # mark character start
        if not self.reached_sector_end(auto_mark_position_in_sector, image_sector.shape[1]):
            self.auto_marked_start_group_list.append(PRESSED_KEY_VALUE_CHARACTER_START)
            height_index = 0
            while height_index < image_sector.shape[0]:
                if image_sector[height_index, auto_mark_position_in_sector, 0] < self.auto_marking_threshold:
                    segment = Segment(ROOT_SEGMENT_ID)
                    segment.start = height_index
                    segment.min_h = segment.start
                    if character_min_height_index > segment.start:
                        character_min_height_index = segment.start
                    height_index += 1
                    while (
                            height_index < image_sector.shape[0]
                    ) and (
                            image_sector[height_index, auto_mark_position_in_sector, 0] < self.auto_marking_threshold
                    ):
                        height_index += 1
                    height_index -= 1
                    segment.end = height_index
                    segment.max_h = segment.end
                    if character_max_height_index < segment.end:
                        character_max_height_index = segment.end
                    segment.width = 1
                    character_segments_list.append(segment)
                height_index += 1
            auto_mark_position_in_sector += 1

        # mark character middle
        root_segment_end_detected = False
        new_segment_id = ROOT_SEGMENT_ID + 1
        while (not root_segment_end_detected) and (
                not self.reached_sector_end(auto_mark_position_in_sector, image_sector.shape[1])):

            character_middle_segments_list = []
            height_index = 0
            while height_index < image_sector.shape[0]:
                if image_sector[height_index, auto_mark_position_in_sector, 0] < self.auto_marking_threshold:
                    segment = Segment(new_segment_id)
                    new_segment_id += 1
                    segment.start = height_index
                    segment.min_h = segment.start
                    height_index += 1
                    while (
                            height_index < image_sector.shape[0]
                    ) and (
                            image_sector[
                                height_index, auto_mark_position_in_sector, 0] < self.auto_marking_threshold
                    ):
                        height_index += 1
                    height_index -= 1
                    segment.end = height_index
                    segment.max_h = segment.end
                    segment.width = 1

                    for character_segment in character_segments_list:
                        # if pixels of current and previous segments cross
                        if (character_segment.end + MAX_GAP_BETWEEN_SEGMENTS >= segment.start) and (
                                character_segment.start - MAX_GAP_BETWEEN_SEGMENTS <= segment.end):
                            segment.connections.append(character_segment.id)
                            if character_segment.id < segment.id:
                                segment.id = character_segment.id
                            if character_segment.width + 1 > segment.width:
                                segment.width = character_segment.width + 1
                            if character_segment.min_h < segment.min_h:
                                segment.min_h = character_segment.min_h
                            if character_segment.max_h > segment.max_h:
                                segment.max_h = character_segment.max_h

                    character_middle_segments_list.append(segment)
                height_index += 1

            # analyzing interconnections between segments in character_middle_segments_list
            for segment_1 in character_middle_segments_list:
                # connections list may be empty, but in that case the 'for' loop will not be executed
                for connection_1 in segment_1.connections:
                    if connection_1 != segment_1.id:
                        for segment_2 in character_middle_segments_list:
                            if (character_middle_segments_list.index(segment_1) !=
                                    character_middle_segments_list.index(segment_2)):
                                for i in range(len(segment_2.connections)):
                                    if segment_2.connections[i] == connection_1:
                                        segment_2.connections[i] = segment_1.id
                                        if segment_2.id > segment_1.id:
                                            segment_2.id = segment_1.id

            # if a segment is connected to a root segment of the character, then
            # transfer min and max height of the segment to the character min and max height
            for segment in character_middle_segments_list:
                if segment.id == ROOT_SEGMENT_ID:
                    if character_min_height_index > segment.min_h:
                        character_min_height_index = segment.min_h
                    if character_max_height_index < segment.max_h:
                        character_max_height_index = segment.max_h

            # detecting if a root segment is not present any more in character_middle_segments_list meaning
            # that the end of the character is reached
            root_segment_end_detected = True
            for middle_segment in character_middle_segments_list:
                if middle_segment.id == ROOT_SEGMENT_ID:
                    self.auto_marked_character_middle_list.append(PRESSED_KEY_VALUE_CHARACTER_MIDDLE)
                    character_segments_list = character_middle_segments_list
                    root_segment_end_detected = False
                    auto_mark_position_in_sector += 1
                    break

        # calculate max width of a next character segment present in the current character rectangle of pixels
        unconnected_segment_width = 0
        for character_segment in character_segments_list:
            if character_segment.id != ROOT_SEGMENT_ID:
                if unconnected_segment_width < character_segment.width:
                    unconnected_segment_width = character_segment.width
        character_end_shift = unconnected_segment_width // 2

        # if root segment end detected: at least start is appended and one target after start
        # (or after one or several middles) is either a space or has no segments, connected to the root segment
        if root_segment_end_detected:
            if (len(self.auto_marked_character_middle_list) - character_end_shift >
                    MIN_N_OF_MIDDLE_TARGETS_IN_CHARACTER):
                del self.auto_marked_character_middle_list[-1 - character_end_shift:]
                self.auto_marked_character_end_list.append(PRESSED_KEY_VALUE_CHARACTER_END)
                self.auto_marked_start_group_list.append(character_min_height_index)
                self.auto_marked_start_group_list.append(character_max_height_index)

                if show_text_and_markers:
                    if self.character_class_index < len(self.string_of_character_classes_list):
                        self.mark_draft_sector_ax.text(
                            auto_mark_start_position_in_sector + len(self.auto_marked_spaces_list),
                            - self.draft_sector_text_shift,
                            self.string_of_character_classes_list[self.character_class_index][CHARACTER_CLASS_KEY],
                            fontsize=self.draft_sector_text_font_size,
                            color=COLOR_GREEN_A700_STRING
                        )
                    else:
                        self.mark_draft_sector_ax.text(
                            auto_mark_start_position_in_sector,
                            - self.draft_sector_text_shift,
                            ENTER_CHARACTER_CLASS_WARNING,
                            fontsize=self.draft_sector_text_font_size,
                            color=COLOR_PINK_A400_STRING
                        )
            else:
                # mark as spaces the start and all middles up to the start of the longest unconnected segment
                # that may be the start of the next character
                for i in range(1 + len(self.auto_marked_character_middle_list) - unconnected_segment_width):
                    self.auto_marked_spaces_list.append(PRESSED_KEY_VALUE_SPACE)

                # empty the unused lists
                self.auto_marked_start_group_list = []
                self.auto_marked_character_middle_list = []

                if show_text_and_markers:
                    self.mark_draft_sector_ax.text(
                        auto_mark_start_position_in_sector,
                        - self.draft_sector_text_shift,
                        DETECTED_TOO_NARROW_CHARACTER,
                        fontsize=self.draft_sector_text_font_size,
                        color=COLOR_PINK_A400_STRING
                    )
        # if root segment end is NOT detected, then the end of the sector is reached, which is either after one
        # or several spaces, after a start or after one or several middles
        else:
            # if end of image is reached
            if self.mark_position + auto_mark_position_in_sector >= self.image.shape[1]:
                if len(self.auto_marked_character_middle_list) > MIN_N_OF_MIDDLE_TARGETS_IN_CHARACTER:
                    del self.auto_marked_character_middle_list[-1]
                    self.auto_marked_character_end_list.append(PRESSED_KEY_VALUE_CHARACTER_END)
                    self.auto_marked_start_group_list.append(character_min_height_index)
                    self.auto_marked_start_group_list.append(character_max_height_index)

                    if show_text_and_markers:
                        if self.character_class_index < len(self.string_of_character_classes_list):
                            self.mark_draft_sector_ax.text(
                                auto_mark_start_position_in_sector + len(self.auto_marked_spaces_list),
                                - self.draft_sector_text_shift,
                                self.string_of_character_classes_list[self.character_class_index][CHARACTER_CLASS_KEY],
                                fontsize=self.draft_sector_text_font_size,
                                color=COLOR_GREEN_A700_STRING
                            )
                        else:
                            self.mark_draft_sector_ax.text(
                                auto_mark_start_position_in_sector,
                                - self.draft_sector_text_shift,
                                ENTER_CHARACTER_CLASS_WARNING,
                                fontsize=self.draft_sector_text_font_size,
                                color=COLOR_PINK_A400_STRING
                            )
                else:
                    # mark as spaces the start and all middles
                    if self.auto_marked_start_group_list:
                        message = DETECTED_TOO_NARROW_CHARACTER
                        for i in range(1 + len(self.auto_marked_character_middle_list)):
                            self.auto_marked_spaces_list.append(PRESSED_KEY_VALUE_SPACE)
                    else:
                        message = REACHED_END_OF_IMAGE

                    # empty the unused lists
                    self.auto_marked_start_group_list = []
                    self.auto_marked_character_middle_list = []

                    if show_text_and_markers:
                        self.mark_draft_sector_ax.text(
                            auto_mark_start_position_in_sector,
                            - self.draft_sector_text_shift,
                            message,
                            fontsize=self.draft_sector_text_font_size,
                            color=COLOR_PINK_A400_STRING
                        )
            else:
                # empty remaining lists
                self.auto_marked_start_group_list = []
                self.auto_marked_character_middle_list = []
                message = CHARACTER_IS_OUT_OF_CURRENT_SCOPE if self.auto_marked_spaces_list \
                    else CANT_MARK_CHARACTER_TOO_WIDE
                if show_text_and_markers:
                    self.mark_draft_sector_ax.text(
                        auto_mark_start_position_in_sector,
                        - self.draft_sector_text_shift,
                        message,
                        fontsize=self.draft_sector_text_font_size,
                        color=COLOR_PINK_A400_STRING
                    )

        # set character lowest and highest levels
        self.character_lowest_point_index = character_min_height_index
        self.character_highest_point_index = character_max_height_index

        # placing rectangle markers on the mark draft sector ax
        if show_text_and_markers:
            rectangle_center_x_position = auto_mark_start_position_in_sector
            for item in self.auto_marked_spaces_list:
                rectangle_marker = Rectangle(
                    (rectangle_center_x_position - 0.4, space_marker_y_start), 0.8, space_marker_y_length,
                    linewidth=1, edgecolor=COLOR_LIGHT_GREEN_A400_STRING, facecolor='none')
                self.mark_draft_sector_ax.add_patch(rectangle_marker)
                rectangle_center_x_position += 1
            if self.auto_marked_start_group_list:
                h_marker_start = rectangle_center_x_position
                v_marker_y_start = character_min_height_index
                v_marker_y_length = character_max_height_index - character_min_height_index
                rectangle_marker = Rectangle(
                    (rectangle_center_x_position - 0.4, v_marker_y_start), 0.8, v_marker_y_length,
                    linewidth=1, edgecolor=COLOR_LIGHT_GREEN_A400_STRING, facecolor='none')
                self.mark_draft_sector_ax.add_patch(rectangle_marker)
                rectangle_center_x_position += 1

                rectangle_center_x_position += len(self.auto_marked_character_middle_list)

                rectangle_marker = Rectangle(
                    (rectangle_center_x_position - 0.4, v_marker_y_start), 0.8, v_marker_y_length,
                    linewidth=1, edgecolor=COLOR_LIGHT_GREEN_A400_STRING, facecolor='none')
                self.mark_draft_sector_ax.add_patch(rectangle_marker)
                rectangle_center_x_position += 1

                h_marker_length = rectangle_center_x_position - h_marker_start
                self.add_horizontal_marker_to_mark_draft_sector_ax(
                    h_marker_start, character_min_height_index, h_marker_length, COLOR_TEAL_A200_STRING)
                self.add_horizontal_marker_to_mark_draft_sector_ax(
                    h_marker_start, character_max_height_index, h_marker_length, COLOR_TEAL_A200_STRING)

    def reached_sector_end(self, auto_mark_position_in_sector, sector_length):
        return auto_mark_position_in_sector >= sector_length

    def add_auto_marked_targets(self):
        for item in self.auto_marked_spaces_list:
            self.mark_single_target_and_show_verification(PRESSED_KEY_VALUE_SPACE)

        if self.auto_marked_start_group_list:
            self.mark_single_target_and_show_verification(PRESSED_KEY_VALUE_CHARACTER_START)
            self.strip_of_targets_list += self.auto_marked_start_group_list[1:]

        for item in self.auto_marked_character_middle_list:
            self.mark_single_target_and_show_verification(PRESSED_KEY_VALUE_CHARACTER_MIDDLE)

        for item in self.auto_marked_character_end_list:
            self.mark_single_target_and_show_verification(PRESSED_KEY_VALUE_CHARACTER_END)

    def exit_auto_mark_character_mode(self):
        if self.pressed_escape_in_auto_mark_mode:
            self.pressed_escape_in_auto_mark_mode = False
            return True
        elif self.mark_position >= self.image.shape[1]:
            return True
        else:
            return False

    def mark_start_group(self, image_sector, rectangle_marker):
        self.start_group_parameters_list_index = 0
        self.height_marker_index_low_limit = - 1
        self.height_marker_index_high_limit = image_sector.shape[0] - 1
        self.height_marker_index = self.height_marker_index_high_limit // 3
        self.character_lowest_point_index = 0
        self.character_highest_point_index = 0

        self.update_mark_verification_on_verification_ax()

        while not self.marked_all_start_group_parameters():
            self.start_group_marking_loop_step(image_sector, rectangle_marker)

    def on_key_press_event(self, event):
        if not self.lock.locked():
            self.lock.acquire()

            if event.key == 'pageup':
                self.auto_mark_multiple_characters_and_show_verification()
                self.pressed_key_value = PRESSED_KEY_VALUE_AUTO_MARK_CHARACTER
                self.figure.canvas.stop_event_loop()
            elif event.key == 'insert':
                self.pressed_key_value = PRESSED_KEY_VALUE_AUTO_MARK_CHARACTER
                self.figure.canvas.stop_event_loop()
            elif event.key == ' ':
                self.pressed_key_value = PRESSED_KEY_VALUE_SPACE
                self.mark_single_target_and_show_verification(self.pressed_key_value)
                self.figure.canvas.stop_event_loop()
            elif event.key == 'delete':
                self.pressed_key_value = PRESSED_KEY_VALUE_CHARACTER_START
                self.mark_single_target_and_show_verification(self.pressed_key_value)
                self.figure.canvas.stop_event_loop()
            elif event.key == 'end':
                self.pressed_key_value = PRESSED_KEY_VALUE_CHARACTER_MIDDLE
                self.mark_single_target_and_show_verification(self.pressed_key_value)
                self.figure.canvas.stop_event_loop()
            elif event.key == 'pagedown':
                self.pressed_key_value = PRESSED_KEY_VALUE_CHARACTER_END
                self.mark_single_target_and_show_verification(self.pressed_key_value)
                self.figure.canvas.stop_event_loop()
            elif event.key == 'backspace':
                self.pressed_key_value = PRESSED_KEY_VALUE_DELETE_PREVIOUS
                self.delete_previous_single_target_and_verification()
                self.figure.canvas.stop_event_loop()
            elif event.key == '=':
                self.pressed_key_value = PRESSED_KEY_VALUE_DELETE_PREVIOUS_GROUP
                self.delete_previous_group_and_verification()
                self.figure.canvas.stop_event_loop()
            elif event.key == '-':
                self.pressed_key_value = PRESSED_KEY_VALUE_DELETE_MULTIPLE_GROUPS
                self.delete_multiple_groups_and_verification()
                self.figure.canvas.stop_event_loop()
            else:
                self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
                self.lock.release()
            print(f"pressed key: '{event.key}', interpretation: {self.pressed_key_value}")

    def mark_single_target_and_show_verification(self, target_value):
        self.mark_verification[0:self.mark_verification_padding_start, self.mark_position] *= \
            verification_image_and_mark_color_factors[target_value][0]
        self.mark_verification[self.mark_verification_padding_start:, self.mark_position] = \
            self.verification_samples_dict[target_value]

        # mark highest and lowest level of the character
        if self.mark_lowest_and_highest_level:
            inversion_pixel = torch.tensor([1, 1, 1, 2])
            factor = verification_image_and_mark_color_factors[NEGATIVE_LOWEST_AND_HIGHEST_LEVEL_COLOR_TENSOR][0]
            negative_pixel = inversion_pixel - self.mark_verification[self.character_lowest_point_index,
                                                                      self.mark_position]
            self.mark_verification[self.character_lowest_point_index, self.mark_position] = \
                inversion_pixel - negative_pixel * factor
            negative_pixel = inversion_pixel - self.mark_verification[self.character_highest_point_index,
                                                                      self.mark_position]
            self.mark_verification[self.character_highest_point_index, self.mark_position] = \
                inversion_pixel - negative_pixel * factor

        if target_value == PRESSED_KEY_VALUE_CHARACTER_START:
            self.mark_lowest_and_highest_level = True
        elif target_value == PRESSED_KEY_VALUE_CHARACTER_END:
            self.mark_lowest_and_highest_level = False

        self.mark_position += 1
        self.strip_of_targets_list.append(target_value)

    def delete_previous_single_target_and_verification(self):
        if self.mark_position > 0:
            self.mark_position -= 1
            self.mark_verification[0:self.mark_verification_padding_start, self.mark_position] = \
                self.image[:, self.mark_position]
            self.mark_verification[self.mark_verification_padding_start:, self.mark_position] = \
                self.verification_samples_dict[PRESSED_KEY_VALUE_DELETE_PREVIOUS]
            if self.strip_of_targets_list[-1] == END_OF_CHARACTER_START_GROUP:
                while self.strip_of_targets_list[-1] != PRESSED_KEY_VALUE_CHARACTER_START:
                    del self.strip_of_targets_list[-1]
                del self.strip_of_targets_list[-1]
                self.character_class_index -= 1
            else:
                del self.strip_of_targets_list[-1]

    def delete_multiple_groups_and_verification(self):
        group_counter = 0
        number_of_groups_to_delete = 5
        while group_counter < number_of_groups_to_delete:
            self.delete_previous_group_and_verification()
            group_counter += 1

    def delete_previous_group_and_verification(self):
        if (self.mark_position > 0) and (self.strip_of_targets_list[-1] == PRESSED_KEY_VALUE_CHARACTER_END):
            self.delete_previous_single_target_and_verification()
        while (self.mark_position > 0) and (self.strip_of_targets_list[-1] != PRESSED_KEY_VALUE_CHARACTER_END):
            self.delete_previous_single_target_and_verification()

    def on_key_press_event_when_reached_image_end(self, event):
        if not self.lock.locked():
            self.lock.acquire()
            if event.key == 'backspace':
                self.pressed_key_value = PRESSED_KEY_VALUE_DELETE_PREVIOUS
                self.delete_previous_single_target_and_verification()
                self.figure.canvas.stop_event_loop()
            elif event.key == '=':
                self.pressed_key_value = PRESSED_KEY_VALUE_DELETE_PREVIOUS_GROUP
                self.delete_previous_group_and_verification()
                self.figure.canvas.stop_event_loop()
            elif event.key == '-':
                self.pressed_key_value = PRESSED_KEY_VALUE_DELETE_MULTIPLE_GROUPS
                self.delete_multiple_groups_and_verification()
                self.figure.canvas.stop_event_loop()
            elif event.key == 'enter':
                self.pressed_key_value = PRESSED_KEY_VALUE_SAVING_FILE
                self.save_file = True
                self.figure.canvas.stop_event_loop()
            else:
                self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
                self.lock.release()
            print(f"\t\tpressed key: '{event.key}', interpretation: {self.pressed_key_value}")

    def on_key_press_event_when_marking_start_group(self, event):
        # this callback is active when start group marking is in progress
        if not self.lock.locked():
            self.lock.acquire()
            # switch reaction to the pressed key depending on the parameter of the start group list to be marked
            if start_group_parameters_list[self.start_group_parameters_list_index] == CHARACTER_LOWEST_POINT:
                self.on_key_press_when_marking_character_lowest_point(event)
            elif start_group_parameters_list[self.start_group_parameters_list_index] == CHARACTER_HIGHEST_POINT:
                self.on_key_press_when_marking_character_highest_point(event)
            elif start_group_parameters_list[self.start_group_parameters_list_index] == CHARACTER_CLASS:
                self.on_key_press_when_marking_character_class(event)

    def on_key_press_when_marking_character_lowest_point(self, event):
        if event.key == 'up':
            self.pressed_key_value = PRESSED_KEY_VALUE_MOVE_UP
            # check if height marker index is within limitations
            if self.height_marker_index - 1 > self.height_marker_index_low_limit:
                # decrease height marker index
                self.height_marker_index -= 1
            self.figure.canvas.stop_event_loop()
        elif event.key == 'down':
            self.pressed_key_value = PRESSED_KEY_VALUE_MOVE_DOWN
            # check if height marker index is within limitations
            if self.height_marker_index + 1 < self.height_marker_index_high_limit:
                # increase height marker index
                self.height_marker_index += 1
            self.figure.canvas.stop_event_loop()
        elif event.key == 'enter':
            self.pressed_key_value = PRESSED_KEY_VALUE_ENTER
            self.character_lowest_point_index = self.height_marker_index
            self.height_marker_index_low_limit = self.height_marker_index
            if ((self.height_marker_index + self.height_marker_index_high_limit // 3) <
                    self.height_marker_index_high_limit):
                self.height_marker_index += self.height_marker_index_high_limit // 3
            else:
                self.height_marker_index += 1
            self.height_marker_index_high_limit += 1
            self.start_group_parameters_list_index += 1
            self.figure.canvas.stop_event_loop()
            print(f"\r\t\tentered LOWEST character point: {self.character_lowest_point_index}")
        else:
            self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
            self.lock.release()

    def on_key_press_when_marking_character_highest_point(self, event):
        if event.key == 'up':
            self.pressed_key_value = PRESSED_KEY_VALUE_MOVE_UP
            # check if height marker index is within limitations
            if self.height_marker_index - 1 > self.height_marker_index_low_limit:
                # decrease height marker index
                self.height_marker_index -= 1
            self.figure.canvas.stop_event_loop()
        elif event.key == 'down':
            self.pressed_key_value = PRESSED_KEY_VALUE_MOVE_DOWN
            # check if height marker index is within limitations
            if self.height_marker_index + 1 < self.height_marker_index_high_limit:
                # increase height marker index
                self.height_marker_index += 1
            self.figure.canvas.stop_event_loop()
        elif event.key == 'enter':
            self.pressed_key_value = PRESSED_KEY_VALUE_ENTER
            self.character_highest_point_index = self.height_marker_index
            self.start_group_parameters_list_index += 1
            self.figure.canvas.stop_event_loop()
            print(f"\r\t\tentered HIGHEST character point: {self.character_highest_point_index}")
        else:
            self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
            self.lock.release()

    def on_key_press_when_marking_character_class(self, event):
        if event.key in character_classes_dict:

            self.strip_of_targets_list.append(self.character_lowest_point_index)
            self.strip_of_targets_list.append(self.character_highest_point_index)
            self.strip_of_targets_list.append(character_classes_dict[event.key])
            self.strip_of_targets_list.append(END_OF_CHARACTER_START_GROUP)
            self.start_group_parameters_list_index += 1
            self.pressed_key_value = END_OF_CHARACTER_START_GROUP

            character_classes_list_item = {
                    COORDINATE_X_KEY: self.mark_position - 1,
                    CHARACTER_CLASS_KEY: character_classes_dict[event.key]
                }
            if self.character_class_index < len(self.string_of_character_classes_list):
                self.string_of_character_classes_list[self.character_class_index] = character_classes_list_item
                self.rearrange_characters_in_string_of_character_classes_list()
            else:
                self.string_of_character_classes_list.append(character_classes_list_item)
            self.character_class_index += 1
            self.figure.canvas.stop_event_loop()
            print(f"\r\t\tentered character CLASS: {character_classes_dict[event.key]}")
        elif event.key == 'enter':
            if self.character_class_index < len(self.string_of_character_classes_list):
                self.string_of_character_classes_list[self.character_class_index][COORDINATE_X_KEY] = \
                    self.mark_position - 1
                self.rearrange_characters_in_string_of_character_classes_list()
                self.strip_of_targets_list.append(self.character_lowest_point_index)
                self.strip_of_targets_list.append(self.character_highest_point_index)
                character_class = self.string_of_character_classes_list[self.character_class_index][CHARACTER_CLASS_KEY]
                self.strip_of_targets_list.append(character_class)
                self.strip_of_targets_list.append(END_OF_CHARACTER_START_GROUP)
                self.start_group_parameters_list_index += 1
                self.pressed_key_value = END_OF_CHARACTER_START_GROUP
                self.character_class_index += 1
                self.figure.canvas.stop_event_loop()
                print(f"\r\t\tentered character CLASS: {character_class}")
            else:
                self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
                self.lock.release()
        else:
            self.pressed_key_value = PRESSED_KEY_VALUE_WRONG_BUTTON
            self.lock.release()

    def rearrange_characters_in_string_of_character_classes_list(self):
        distance_divider = len(self.string_of_character_classes_list) - self.character_class_index
        distance_between_characters = (
                (self.image.shape[1] -
                 self.string_of_character_classes_list[self.character_class_index][COORDINATE_X_KEY]) /
                distance_divider)
        for i in range(1, distance_divider):
            self.string_of_character_classes_list[self.character_class_index + i][COORDINATE_X_KEY] = \
                self.string_of_character_classes_list[self.character_class_index + i - 1][COORDINATE_X_KEY] + \
                distance_between_characters

    def start_group_marking_loop_step(self, image_sector, rectangle_marker):
        self.mark_draft_sector_ax.cla()
        self.mark_draft_sector_ax.imshow(image_sector, aspect=self.image_sector_aspect_ratio)
        self.mark_draft_sector_ax.add_patch(rectangle_marker)
        marker_start = 0

        if self.character_class_index < len(self.string_of_character_classes_list):
            self.mark_draft_sector_ax.text(
                NEXT_CHARACTER_TEXT_COORDINATE_X,
                - self.draft_sector_text_shift,
                ': '.join([NEXT_CHARACTER_TEXT,
                           self.string_of_character_classes_list[self.character_class_index][CHARACTER_CLASS_KEY]]),
                fontsize=self.draft_sector_text_font_size
            )
        else:
            self.mark_draft_sector_ax.text(
                NO_NEXT_CHARACTER_WARNING_COORDINATE_X,
                - self.draft_sector_text_shift,
                NO_NEXT_CHARACTER_WARNING,
                fontsize=self.draft_sector_text_font_size
            )

        # switch visualisation depending on the parameter of the start group list to be marked
        if start_group_parameters_list[self.start_group_parameters_list_index] == CHARACTER_LOWEST_POINT:
            print("\rUse 'up', 'down' and 'enter' to mark LOWEST character point", end='')
            self.add_horizontal_marker_to_mark_draft_sector_ax(
                marker_start, self.height_marker_index, image_sector.shape[1], COLOR_TEAL_A200_STRING)
        elif start_group_parameters_list[self.start_group_parameters_list_index] == CHARACTER_HIGHEST_POINT:
            print("\rUse 'up', 'down' and 'enter' to mark HIGHEST character point", end='')
            self.add_horizontal_marker_to_mark_draft_sector_ax(
                marker_start, self.height_marker_index, image_sector.shape[1], COLOR_TEAL_A200_STRING)
            self.add_horizontal_marker_to_mark_draft_sector_ax(
                marker_start, self.character_lowest_point_index, image_sector.shape[1], COLOR_ORANGE_800_STRING)
        elif start_group_parameters_list[self.start_group_parameters_list_index] == CHARACTER_CLASS:
            print("\rUse letters, digits and ',' to mark CHARACTER CLASS", end='')
            self.add_horizontal_marker_to_mark_draft_sector_ax(
                marker_start, self.character_lowest_point_index, image_sector.shape[1], COLOR_ORANGE_800_STRING)
            self.add_horizontal_marker_to_mark_draft_sector_ax(
                marker_start, self.character_highest_point_index, image_sector.shape[1], COLOR_ORANGE_800_STRING)

        # waiting for user to input a value for the current start group parameter
        self.block_until_key_is_pressed()

    def add_horizontal_marker_to_mark_draft_sector_ax(self, marker_x, marker_y, marker_length, marker_color):
        marker = Rectangle((marker_x - 0.4, marker_y - 0.4), marker_length - 0.2, 0.8,
                           linewidth=1, edgecolor=marker_color, facecolor='none')
        self.mark_draft_sector_ax.add_patch(marker)

    def marking_loop_step(self):
        image_sector, image_sector_start = self.calculate_image_sector()
        self.update_mark_verification_on_verification_ax()
        self.image_sector_ax.cla()
        self.image_sector_ax.imshow(image_sector, aspect=self.image_sector_aspect_ratio)
        self.mark_draft_sector_ax.cla()
        mark_position_shift = self.mark_position - image_sector_start
        rectangle_marker = Rectangle((mark_position_shift - 0.4, 0.2), 0.8, image_sector.shape[0] - 1.4,
                                     linewidth=2, edgecolor=COLOR_BLUE_A400_STRING, facecolor='none')
        if not self.mark_position_reached_image_end():
            self.mark_draft_sector_ax.imshow(image_sector, aspect=self.image_sector_aspect_ratio)
            self.mark_draft_sector_ax.add_patch(rectangle_marker)
            if self.character_class_index < len(self.string_of_character_classes_list):
                self.mark_draft_sector_ax.text(
                    NEXT_CHARACTER_TEXT_COORDINATE_X,
                    - self.draft_sector_text_shift,
                    ': '.join([NEXT_CHARACTER_TEXT,
                               self.string_of_character_classes_list[self.character_class_index][CHARACTER_CLASS_KEY]]),
                    fontsize=self.draft_sector_text_font_size
                )
            else:
                self.mark_draft_sector_ax.text(
                    NO_NEXT_CHARACTER_WARNING_COORDINATE_X,
                    - self.draft_sector_text_shift,
                    NO_NEXT_CHARACTER_WARNING,
                    fontsize=self.draft_sector_text_font_size
                )
        else:
            image_with_cut_characters = torch.nn.functional.pad(input=self.image, pad=(
                0, 0, 0, 0, self.image.shape[0], 0), mode='constant', value=1.0)
            target_index = 0
            image_index = 0
            list_of_character_classes = []

            while target_index < len(self.strip_of_targets_list):
                if self.strip_of_targets_list[target_index] == PRESSED_KEY_VALUE_CHARACTER_START:
                    character_start = image_index
                    target_index += 1
                    character_lowest_level = self.strip_of_targets_list[target_index]
                    target_index += 1
                    character_highest_level = self.strip_of_targets_list[target_index] + 1
                    target_index += 1
                    class_item = {
                        COORDINATE_X_KEY: character_start,
                        CHARACTER_CLASS_KEY: self.strip_of_targets_list[target_index]
                    }
                    list_of_character_classes.append(class_item)
                    while self.strip_of_targets_list[target_index] != END_OF_CHARACTER_START_GROUP:
                        target_index += 1
                    target_index += 1
                    image_index += 1
                    while self.strip_of_targets_list[target_index] == PRESSED_KEY_VALUE_CHARACTER_MIDDLE:
                        target_index += 1
                        image_index += 1
                    if self.strip_of_targets_list[target_index] == PRESSED_KEY_VALUE_CHARACTER_END:
                        target_index += 1
                        image_index += 1
                        character_end = image_index
                        image_with_cut_characters[character_lowest_level: character_highest_level,
                        character_start: character_end] = \
                            image_with_cut_characters[character_lowest_level + self.image.shape[0]:
                                                      character_highest_level + self.image.shape[0],
                            character_start: character_end]
                        image_with_cut_characters[character_lowest_level + self.image.shape[0]:
                                                  character_highest_level + self.image.shape[0],
                        character_start: character_end] = 1.0
                    target_index -= 1
                    image_index -= 1
                target_index += 1
                image_index += 1

            self.mark_draft_sector_ax.imshow(image_with_cut_characters)
            for item in list_of_character_classes:
                self.mark_draft_sector_ax.text(
                    item[COORDINATE_X_KEY],
                    - self.draft_sector_text_shift,
                    item[CHARACTER_CLASS_KEY],
                    fontsize=self.draft_sector_text_font_size
                )

        image_sector_brightness_view = self.calculate_image_sector_brightness_view(image_sector)
        self.update_image_sector_brightness_view_on_brightness_sector_ax(image_sector_brightness_view)

        # waiting for user to input a target for the current position in the image
        self.block_until_key_is_pressed()

        return image_sector, rectangle_marker

    def block_until_key_is_pressed(self):
        if self.lock.locked():
            self.lock.release()
        self.figure.canvas.start_event_loop()

        # this flush_events aims to flush all excessive pressed button events while the lock is still on
        self.figure.canvas.flush_events()

    def marked_all_start_group_parameters(self):
        return self.start_group_parameters_list_index >= len(start_group_parameters_list)

    def mark_position_reached_image_end(self):
        return self.mark_position >= self.image.shape[1]

    def update_mark_verification_on_verification_ax(self):
        self.verification_ax.cla()
        self.verification_ax.imshow(self.mark_verification, aspect=self.verification_aspect_ratio)

        if self.string_of_character_classes_list:
            for character_group in self.string_of_character_classes_list:
                if self.string_of_character_classes_list.index(character_group) < self.character_class_index:
                    text_color = COLOR_PLAIN_BLACK_STRING
                elif self.string_of_character_classes_list.index(character_group) > self.character_class_index:
                    text_color = COLOR_PLAIN_GRAY_STRING
                else:
                    text_color = COLOR_GREEN_A700_STRING
                self.verification_ax.text(
                    character_group[COORDINATE_X_KEY], - self.mark_verification_text_shift,
                    character_group[CHARACTER_CLASS_KEY], fontsize=self.mark_verification_text_font_size,
                    color=text_color
                )
        else:
            self.verification_ax.text(
                NO_CLASSES_ENTERED_WARNING_START_COORDINATE_X, - self.mark_verification_text_shift,
                NO_CLASSES_ENTERED_WARNING, fontsize=self.mark_verification_text_font_size
            )

    def calculate_image_sector(self):
        if self.mark_position >= self.image.shape[1] - self.image_sector_width + self.marker_position_in_image_sector:
            image_sector_start = self.image.shape[1] - self.image_sector_width
        elif self.mark_position < self.marker_position_in_image_sector:
            image_sector_start = 0
        else:
            image_sector_start = self.mark_position - self.marker_position_in_image_sector

        image_sector = self.image[:, image_sector_start: image_sector_start + self.image_sector_width] \
            .detach().clone()
        return image_sector, image_sector_start

    def calculate_image_sector_brightness_view(self, image_sector):
        image_sector_brightness_view = torch.ones([BRIGHTNESS_VIEW_HEIGHT, image_sector.shape[1], 4])
        for w_index in range(image_sector.shape[1]):
            for h_index in range(image_sector.shape[0]):
                brightness_view_h_index = int((1 - image_sector[h_index, w_index, 0])
                                              * (image_sector_brightness_view.shape[0] - 1))
                if brightness_view_h_index < 0:
                    brightness_view_h_index = 0
                elif brightness_view_h_index > image_sector_brightness_view.shape[0] - 1:
                    brightness_view_h_index = image_sector_brightness_view.shape[0] - 1
                image_sector_brightness_view[brightness_view_h_index, w_index] = image_sector[h_index, w_index]
        self.image_sector_brightness_view_aspect_ratio = \
            self.image_sector_aspect_ratio * image_sector.shape[0] / image_sector_brightness_view.shape[0]
        return image_sector_brightness_view

    def update_image_sector_brightness_view_on_brightness_sector_ax(self, image_sector_brightness_view):
        self.brightness_sector_ax.cla()
        self.brightness_sector_ax.imshow(
            image_sector_brightness_view, aspect=self.image_sector_brightness_view_aspect_ratio)
        threshold = image_sector_brightness_view.shape[0] * (1 - self.auto_marking_threshold)
        rectangle_marker = Rectangle((0.2, threshold - 0.2),
                                     image_sector_brightness_view.shape[1] - 1.2, 0.4,
                                     linewidth=1, edgecolor=COLOR_BLUE_A400_STRING, facecolor='none')
        self.brightness_sector_ax.add_patch(rectangle_marker)


if __name__ == '__main__':
    IMAGE_DIRECTORY_PATH = "Dataset/"
    target_marker = WHCTargetMarker(IMAGE_DIRECTORY_PATH)
    target_marker.mark_targets()
