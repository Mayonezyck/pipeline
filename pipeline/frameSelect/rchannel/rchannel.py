import cv2
import os

def rchannel(framelist, config):
    selected_frames = []
    for frame in framelist:
        image = cv2.imread(frame)
        if image is not None:
            red_channel = image[:, :, 2]
            print(red_channel.mean())
            if red_channel.mean() > config['RC_THRESHOLD']:
                selected_frames.append(frame)
    return selected_frames

# def get_framelist(folder_path):
    # return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# folder_path = '/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/image01_50'
# threshold = 126
# framelist = get_framelist(folder_path)
# selected_frames = rchannel(framelist, threshold)

# print(len(selected_frames))