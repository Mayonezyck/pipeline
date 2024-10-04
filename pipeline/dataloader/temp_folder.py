import os
import shutil


def create(content, mute = False):
    if __check_and_create_path__('temp'):
        __copy_in__(content, mute = mute)
        return 'temp'
    else:
        raise TempExistError("Pipeline terminated because previous temp folder found. Consider manually remove the temp folder.")
        return None

def clear_and_load_folder(path, mute = False):
    clear()
    if os.path.exists(path):
        shutil.copytree(path, 'temp')
        print(f"Temp folder loaded with {path}")
    else:
        print("Temp folder does not exist (for some reason)")

def clear():
    if os.path.exists('temp'):
        shutil.rmtree('temp')
        print("Temp folder removed")
    else:
        print("Temp folder does not exist (for some reason)")
        


def __copy_in__(content, dest_folder = 'temp', mute = False):
    for file_path in content:
        if os.path.exists(file_path):
            # Copy the file to the destination folder
            shutil.copy(file_path, dest_folder)
            if not mute:
                print(f"Copied {file_path} to {dest_folder}")
        else:
            print(f"File not found: {file_path}")
    


def __check_and_create_path__(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
        return True
    else:
        print(f"Directory '{path}' already exists. Please check and consider if to proceed.")
        return False
    

class TempExistError(Exception):
    pass