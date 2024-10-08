import os
import subprocess
from datetime import datetime
from dataloader import temp_folder
def estimate(method, path,config, ifmute):
    try:
        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        output_path_raw = f'output/{method}_{current_time}_output'
        if method == 'depth-anything':
            output_path = f"../../../{output_path_raw}"
            command = [
                "python", "run.py", 
                "--encoder", config['DA_ENCODER'], 
                "--img-path", '../../../temp', 
                "--outdir", output_path, 
            ]
            if config['PRED_ONLY']:
                command.append('--pred-only')
            if config['GRAY_SCALE']:
                command.append('--grayscale')
            os.chdir('pipeline/depthEstimate/Depth-Anything')
            result = subprocess.run(command, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if not ifmute:
                print("Output:\n", result.stdout.decode('utf-8'))
                print("Errors:\n", result.stderr.decode('utf-8'))
            os.chdir('../../..')
        elif method == 'endo-depth':
            output_path = f"../../../{output_path_raw}"
            
            modelchoice = config['ED_MODEL']
            command = [
                "python", "apps/depth_estimate/__main__.py",
                "--image_path", '../../../temp',
                "--model_path", f'pretrained_models/{modelchoice}',
                "--output_path", output_path
            ]
            os.chdir('pipeline/depthEstimate/Endo-Depth-and-Motion')
            temp_folder.__check_and_create_path__(output_path)
            result = subprocess.run(command, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if not ifmute:
                print("Output:\n", result.stdout.decode('utf-8'))
                print("Errors:\n", result.stderr.decode('utf-8'))
            os.chdir('../../..')
        return output_path_raw
    except FileNotFoundError:
         print(f"The directory {path} does not exist.")
         return 
    except Exception as e:
        print(f"An error occurred: {e}")
        return 