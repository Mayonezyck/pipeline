import os

def getFrameList(path, mute = False):
    try:
        # List all files in the directory and filter those ending with .jpg
        png_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
        
        # Sort the files based on the numerical part of the filename
        sorted_png_files = sorted(png_files, key=lambda x: int(x.split('.')[0]))
        
        # Convert filenames to absolute paths
        absolute_paths = [os.path.abspath(os.path.join(path, f)) for f in sorted_png_files]
        
        if not mute:
            print(absolute_paths)
        __printFrameListLength__(absolute_paths)
        return absolute_paths
    except FileNotFoundError:
        print(f"The directory {path} does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def pickFrameList(frameList,scoreList, threshold):
    if len(frameList) != len(scoreList):
        raise ValueError("frameList and scoreList must have the same length")

    newFrameList = [frame for frame, score in zip(frameList, scoreList) if score > threshold]
    __printFrameListLength__(newFrameList)
    return newFrameList
    

def __printFrameListLength__(framelist):
    print(f'Frame list has a new length of {len(framelist)}')