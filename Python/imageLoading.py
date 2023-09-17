from os import listdir
from os.path import isfile, join


def create_lists_of_image_and_target_files(image_dir_path, target_dir_path, target_file_suffix):
    """
    this function creates a list of image files and a list of target files from the specified image directory
    :param image_dir_path: a path of the directory, that contains images
    :param target_dir_path: a path of the directory, that contains targets
    :param target_file_suffix: a suffix, that is added to the name of an image file to create the name of its target file
    :return: list_of_image_files, list_of_target_files: a list of image files and a list of target files
    """
    list_of_image_files = []
    list_of_target_files = []
    tmp_list_of_image_files = create_list_of_files(image_dir_path)
    tmp_list_of_target_files = create_list_of_files(target_dir_path)
    for image_file in tmp_list_of_image_files:
        target_file = ''.join([image_file[:-4], target_file_suffix])
        if target_file in tmp_list_of_target_files:
            list_of_image_files.append(image_file)
            list_of_target_files.append(target_file)
    return list_of_image_files, list_of_target_files


def create_list_of_files(dir_path):
    """
    this function creates a list of files from the specified directory
    :param dir_path: a path of the directory, that contains files
    :return: list_of_files: a list of files
    """
    list_of_files = [file_name for file_name in listdir(
        dir_path) if isfile(join(dir_path, file_name))]
    return list_of_files

