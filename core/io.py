import os


def create_folder_for_file(filename: str):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
