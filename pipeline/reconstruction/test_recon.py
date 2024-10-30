import torch, os
from pytorch3d.structures import Pointclouds
def recon(config):
    mute = config['IF_MUTE']
    in_path = config['DEPTH_FOLDER']
    firstTime = True
    #device = torch.device("cuda:0" if torch.cuda.is_available() and config['USE_GPU'] else "cpu")
    if not mute:
        print(f'Device chosen: {device}')
    worldmap = Pointclouds(points=[])
    worldmap = worldmap.to(device)
    file_list = sorted([f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))])
    for filename in file_list:
        print(filename)


#recon({'IF_MUTE': False, 'DEPTH_FOLDER': 'C:/Users/alexa/OneDrive/Desktop/3DReconstructionPipeline/3DReconstructionPipeline/pipeline/temp/2021-09-29_15-00-00', 'USE_GPU': False})