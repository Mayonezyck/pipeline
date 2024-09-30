import os
import subprocess
from datetime import datetime
def estimate(method, path,config, ifmute):
    try:
        now = datetime.now()
        current_time = now.strftime("%Y%m%d%H%M%S")
        output_path_raw = f'output/depth_anything_{current_time}_output'
        if method == 'depth-anything':
            output_path = f"../../../{output_path_raw}"
            command = [
                "python", "run.py", 
                "--encoder", config[0], 
                "--img-path", '../../../temp', 
                "--outdir", output_path, 
            ]
            if config[1]:
                command.append('--pred-only')
            if config[2]:
                command.append('--grayscale')
            os.chdir('pipeline/depthEstimate/Depth-Anything')
            result = subprocess.run(command, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if not ifmute:
                print("Output:\n", result.stdout.decode('utf-8'))
                print("Errors:\n", result.stderr.decode('utf-8'))
            os.chdir('../../..')
            pass
        return output_path_raw
    except FileNotFoundError:
         print(f"The directory {path} does not exist.")
         return 
    except Exception as e:
        print(f"An error occurred: {e}")
        return 