import os
import numpy as np
from PIL import Image
from datetime import datetime

def postprocess(output_folder, config):
    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")
    output_path_raw = output_folder
    if config['DEPTH_SCHEME'] == 'endo-dac':
        output_path_raw = f'{output_folder}_{current_time}_postprocessed'
        input_path = config['DATA_PATH']
        # Get a picture from input_path and read its original height and width
        sample_image_path = None
        for file in os.listdir(input_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                sample_image_path = os.path.join(input_path, file)
                break
        
        if sample_image_path is None:
            raise FileNotFoundError("No image file found in the input path.")
        
        with Image.open(sample_image_path) as img:
            original_width, original_height = img.size
        # Create the output directory if it doesn't exist
        os.makedirs(output_path_raw, exist_ok=True)
        
        fx = config['fx']
        baseline = config['t1']
        
        for filename in os.listdir(output_folder):
            if filename.endswith('_disp.npy'):
                disp_path = os.path.join(output_folder, filename)
                disp = np.load(disp_path)

                # Remove singleton dimensions from disp
                disp = np.squeeze(disp)
                                
                # Resize disp to original height and width
                disp = np.array(Image.fromarray(disp).resize((original_width, original_height), Image.BILINEAR))
                depth = fx * baseline / disp
                
                # Normalize depth and remove singleton dimensions
                depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
                depth_normalized = depth_normalized.astype(np.uint8)
                
                # Remove any extra dimensions
                depth_normalized = np.squeeze(depth_normalized)
                
                depth_path = os.path.join(
                    output_path_raw, 
                    os.path.basename(disp_path).replace('_disp.npy', '_depth.png')
                )
                
                # Create the image from the normalized depth array
                depth_image = Image.fromarray(depth_normalized)
                depth_image.save(depth_path)
                # # Visualize depth and disparity side by side
                # disp_normalized = (disp - np.min(disp)) / (np.max(disp) - np.min(disp)) * 255
                # disp_normalized = disp_normalized.astype(np.uint8)
                
                # # Create the image from the normalized disparity array
                # disp_image = Image.fromarray(disp_normalized)
                
                # # Combine depth and disparity images side by side
                # combined_image = Image.new('L', (depth_normalized.shape[1] * 2, depth_normalized.shape[0]))
                # combined_image.paste(depth_image, (0, 0))
                # combined_image.paste(disp_image, (depth_normalized.shape[1], 0))
                
                # # Save the combined image
                # combined_image_path = os.path.join(
                #     output_path_raw, 
                #     os.path.basename(disp_path).replace('_disp.npy', '_combined.png')
                # )
                # combined_image.save(combined_image_path)
    return output_path_raw
