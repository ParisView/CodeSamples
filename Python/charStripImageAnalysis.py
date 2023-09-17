from os.path import join
import numpy as np
from matplotlib import image as mpimg, pyplot as plt
from threading import Lock
from math import cos, exp

from imageLoading import create_lists_of_image_and_target_files

target_file_name_suffix = "_target.txt"
target_directory_name = "targets/"

# targets to values conversion dictionary
targets_to_values_dict = {
    "space": 0.0,
    "character_start": 1.0,
    "character_middle": 2.0,
    "character_end": 1.0
}

# creating a figure
plt.ion()
figure = plt.figure(figsize=(15.0, 8.0))

# creating a lock
lock = Lock()

# setting the number of samples, which are used for local fft calculation
fft_number_of_samples = 64


def analyze_text_string_frequency_in_image(image_directory_path):
    target_directory_path = join(image_directory_path, target_directory_name)

    # creating a list of image files and a list of target files
    list_of_image_files, list_of_target_files = create_lists_of_image_and_target_files(
        image_directory_path, target_directory_path, target_file_name_suffix
    )

    # connecting a callback for key press event
    connection_id = figure.canvas.mpl_connect('key_press_event', on_key_press_event)

    # one by one loading target data to a numpy array from the list of targets
    for image_index in range(len(list_of_target_files)):
        # load the content of the image file
        image_file = ''.join([image_directory_path, list_of_image_files[image_index]])
        image_content = mpimg.imread(image_file)

        # load the content of the target file into an array [height (1), width(~700)]
        target_file = ''.join([target_directory_path, list_of_target_files[image_index]])
        target_file = open(target_file, 'r', encoding='utf8')
        target_content_list = [string.strip() for string in target_file]
        target_file.close()
        # converting strings of target names to values
        target_in_values = np.array([targets_to_values_dict[item] for item in target_content_list])
        target_width = target_in_values.shape[0]

        # set parameters of the figure
        figure.clf()
        ax4_start_margin = 10  # margin at the start of the target strip
        ax4_end_margin = ax4_start_margin  # margin at the end of the target strip
        number_of_ax4_parts = (target_width - ax4_start_margin - ax4_end_margin) // fft_number_of_samples
        ax4_remainder = (target_width - ax4_start_margin - ax4_end_margin) % fft_number_of_samples
        ax4_middle_margin = ax4_remainder // (number_of_ax4_parts - 1)
        spec = figure.add_gridspec(4, number_of_ax4_parts)
        ax1 = figure.add_subplot(spec[0, :])
        ax2 = figure.add_subplot(spec[1, :])
        ax3 = figure.add_subplot(spec[2, :])
        ax4 = [figure.add_subplot(spec[3, j]) for j in range(number_of_ax4_parts)]
        ax2_y_margin = 0.5

        # create arrays for frequency modulation calculation and Fourier coefficients
        frequency_index_row = 0
        number_of_rows = 2 + number_of_ax4_parts
        fm_calculation_array = np.zeros((number_of_rows, number_of_ax4_parts), dtype=np.int32)
        main_tone_fourier_coefficients_array = np.zeros((3, number_of_ax4_parts), dtype=np.complex128)
        main_tone_row = 1
        main_tone_minus_1_row = 0
        main_tone_plus_1_row = 2
        step_in_space = fft_number_of_samples + ax4_middle_margin

        # plot the image content
        ax1.imshow(image_content)
        # plot the target content
        ax2.plot(target_in_values, 'b')
        ax2.axis([0, len(target_in_values), min(targets_to_values_dict.values()) - ax2_y_margin,
                  max(targets_to_values_dict.values()) + ax2_y_margin])
        # plot the target spectrum, calculated on the whole target content
        spectrum = np.abs(np.fft.rfft(target_in_values))
        spectrum[0] = 0.0
        ax3.plot(spectrum)
        # plot the target spectrum, calculated on each fft_number_of_samples of the target content,
        # and save frequency with max amplitude
        start_index = ax4_start_margin
        stop_index = start_index + fft_number_of_samples
        for ax4_part_index in range(number_of_ax4_parts):
            spectrum = np.fft.rfft(target_in_values[start_index: stop_index], fft_number_of_samples)
            spectrum_abs = np.abs(spectrum)
            spectrum_abs[0] = 0.0
            ax4[ax4_part_index].plot(spectrum_abs)
            ax4[ax4_part_index].plot(np.real(spectrum), 'r')
            ax4[ax4_part_index].plot(np.imag(spectrum), 'g')

            max_amplitude_frequency_index = np.argmax(spectrum_abs)
            # fill the frequency index row of the frequency modulation calculation array
            fm_calculation_array[frequency_index_row, ax4_part_index] = max_amplitude_frequency_index

            # fill the main_tone_fourier_coefficients_array with max amplitude frequency coefficients,
            # and coefficients for frequencies with max_amplitude_frequency_index minus 1 and plus 1
            main_tone_fourier_coefficients_array[main_tone_minus_1_row, ax4_part_index] = \
                spectrum[max_amplitude_frequency_index - 1]
            main_tone_fourier_coefficients_array[main_tone_row, ax4_part_index] = \
                spectrum[max_amplitude_frequency_index]
            main_tone_fourier_coefficients_array[main_tone_plus_1_row, ax4_part_index] = \
                spectrum[max_amplitude_frequency_index + 1]

            # calculate and plot reconstructed main tone
            filtered_spectrum = np.zeros_like(spectrum)
            filtered_spectrum[max_amplitude_frequency_index] = spectrum[max_amplitude_frequency_index]
            reconstructed_main_tone = \
                np.fft.irfft(filtered_spectrum) + targets_to_values_dict["character_start"]
            ax2.plot(range(start_index, stop_index), reconstructed_main_tone, 'r')

            space_range = np.array(range(fft_number_of_samples))
            reconstructed_main_tone = (np.real(
                (spectrum[max_amplitude_frequency_index] / float(fft_number_of_samples)) *
                np.exp(2j * np.pi * float(max_amplitude_frequency_index) *
                       space_range / float(fft_number_of_samples))) +
                                       np.real(
                                           (spectrum[max_amplitude_frequency_index + 1] / float(
                                               fft_number_of_samples)) *
                                           np.exp(2j * np.pi * float(max_amplitude_frequency_index + 1) *
                                                  space_range / float(fft_number_of_samples))) +
                                       np.real(
                                           (spectrum[max_amplitude_frequency_index - 1] / float(
                                               fft_number_of_samples)) *
                                           np.exp(2j * np.pi * float(max_amplitude_frequency_index - 1) *
                                                  space_range / float(fft_number_of_samples)))) * 1.5 + \
                                      targets_to_values_dict["character_start"]

            ax2.plot(range(start_index, stop_index), reconstructed_main_tone, 'g')

            # calculating and plotting main frequency signal
            angle_factor = 2 * np.pi * float(max_amplitude_frequency_index) / \
                           float(fft_number_of_samples)
            angle_factor_minus_1 = 2 * np.pi * float(max_amplitude_frequency_index - 1) / \
                                   float(fft_number_of_samples)
            angle_factor_plus_1 = 2 * np.pi * float(max_amplitude_frequency_index + 1) / \
                                  float(fft_number_of_samples)
            spectrum_re = np.real(spectrum[max_amplitude_frequency_index]) / \
                          float(fft_number_of_samples)
            spectrum_im = np.imag(spectrum[max_amplitude_frequency_index]) / \
                          float(fft_number_of_samples)
            spectrum_re_minus_1 = np.real(spectrum[max_amplitude_frequency_index - 1]) / \
                                  float(fft_number_of_samples)
            spectrum_im_minus_1 = np.imag(spectrum[max_amplitude_frequency_index - 1]) / \
                                  float(fft_number_of_samples)
            spectrum_re_plus_1 = np.real(spectrum[max_amplitude_frequency_index + 1]) / \
                                 float(fft_number_of_samples)
            spectrum_im_plus_1 = np.imag(spectrum[max_amplitude_frequency_index + 1]) / \
                                 float(fft_number_of_samples)

            for i1 in range(fft_number_of_samples):
                # calculating signal only for positive frequency.
                # for negative (mirrored) frequency spectrum_im = minus spectrum_im for positive frequency,
                # and angle_factor = minus angle_factor due to the negative index of the frequency.
                # so the correct reconstructed main tone is sum of positive frequency signal and negative
                # frequency signal, which is equal to positive frequency signal multiplied by two.
                vertical_shift_coefficient = 0.4
                reconstructed_main_tone[i1] = spectrum_re * np.cos(angle_factor * i1) - \
                                              spectrum_im * np.sin(angle_factor * i1) + \
                                              targets_to_values_dict["character_start"] * \
                                              vertical_shift_coefficient

                reconstructed_main_tone[i1] += spectrum_re_minus_1 * np.cos(angle_factor_minus_1 * i1) - \
                                               spectrum_im_minus_1 * np.sin(angle_factor_minus_1 * i1)

                reconstructed_main_tone[i1] += spectrum_re_plus_1 * np.cos(angle_factor_plus_1 * i1) - \
                                               spectrum_im_plus_1 * np.sin(angle_factor_plus_1 * i1)

            ax2.plot(range(start_index, stop_index), reconstructed_main_tone, 'c')

            start_index += step_in_space
            stop_index += step_in_space

        # fill the calculation rows of the frequency modulation calculation array with frequency
        # change and corresponding number of steps in space
        for i1 in range(number_of_ax4_parts - 1):
            for j1 in range(i1 + 1, number_of_ax4_parts):
                fm_calculation_array[j1, i1] = fm_calculation_array[frequency_index_row, j1] - \
                                               fm_calculation_array[frequency_index_row, i1]

        # awaiting for user to press a button
        if lock.locked():
            lock.release()
        figure.canvas.start_event_loop()

        # this flush_events aims to flush all excessive pressed button events while the lock is still on
        figure.canvas.flush_events()

    figure.canvas.mpl_disconnect(connection_id)


def on_key_press_event(event):
    if not lock.locked():
        lock.acquire()
        figure.canvas.stop_event_loop()


if __name__ == '__main__':
    IMAGE_DIRECTORY_PATH = "Dataset/"
    analyze_text_string_frequency_in_image(IMAGE_DIRECTORY_PATH)
