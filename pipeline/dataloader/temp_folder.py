import os
import shutil


def create(content, mute = False):
    if __check_and_create_path__('temp'):
        __copy_in__(content, mute = mute)
    else:
        raise TempExistError("Pipeline terminated because previous temp folder found. Consider manually remove the temp folder.")
    

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