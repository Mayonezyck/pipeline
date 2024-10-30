import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def load_image(file_path):
    # Modify load_image to handle both .png and .npy files
    if file_path.endswith('.npy'): #npy is ENDODAC output
        print(f"Loading .npy file: {file_path}")
        baseline = 0.5196155
        focal_length = 417.9036255  # pixels    
        disp = np.load(file_path)
        depth = baseline * focal_length / (disp + 1e-6)
        return depth
    if file_path.endswith('.png'):#npy is Depth Anything output, which is flipped
        value = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        #value = value + 1e-6
        return value
    else:
        return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

def compute_rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

def compute_mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def compute_sq_rel(pred, gt):
    return np.mean(((pred - gt) ** 2) / gt)

def compute_delta_accuracy(pred, gt, threshold=1.25):
    max_ratio = np.maximum(pred / gt, gt / pred)
    return np.mean(max_ratio < threshold)

def compute_ssim(pred, gt, mask):
    data_range = max(np.max(pred[mask]), np.max(gt[mask]))
    result = ssim(pred, gt, data_range=data_range, mask=mask)
    return result

def compute_log_rmse(pred, gt):
    return np.sqrt(np.mean((np.log(pred + 1e-6) - np.log(gt + 1e-6)) ** 2))

def evaluate_depth_maps(pred_folder='temp', gt_folder='/path/to/gt'):
    # List of valid extensions
    pred_extensions = ['_depth.png', '_disp.npy', '.png']
    gt_extensions = ['.png', '.npy']

    # Gather prediction files with valid extensions
    pred_files = sorted([
        f for f in os.listdir(pred_folder)
        if any(f.endswith(ext) for ext in pred_extensions)
    ])

    gt_files = []
    for pred_file in pred_files:
        # Remove '_depth' and the file extension to get the base filename
        if pred_file.endswith('_depth.png'):
            base_name = pred_file.replace('_depth.png', '')
        elif pred_file.endswith('.png'):
            base_name = pred_file.replace('.png', '')
        elif pred_file.endswith('_disp.npy'):
            base_name = pred_file.replace('_disp.npy', '')
        else:
            continue  # Skip files that don't match the expected pattern

        # Find the corresponding ground truth file
        found_gt = False
        for ext in gt_extensions:
            gt_file = base_name + ext
            gt_path = os.path.join(gt_folder, gt_file)
            if os.path.exists(gt_path):
                gt_files.append(gt_file)
                found_gt = True
                break
        if not found_gt:
            raise FileNotFoundError(f"Ground truth file for {pred_file} not found in {gt_folder}")

    if len(pred_files) != len(gt_files):
        raise ValueError("The number of prediction files and ground truth files do not match.")

    scales = []
    rmses = []
    maes = []
    sq_rels = []
    delta_accuracies = []
    ssims = []
    log_rmses = []

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, gt_file)

        pred_image = load_image(pred_path)
        gt_image = load_image(gt_path)
        if pred_file.endswith('.png'):
            pred_image = 255 - pred_image #post processing for depth-anything v2 outpu
        # Ensure images are 2D arrays
        print(f'{pred_file} shape: {pred_image.shape}')
        if pred_image.ndim == 3 and pred_image.shape[2] == 3:
            pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2GRAY)
        if gt_image.ndim == 3 and gt_image.shape[2] == 3:
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)

        pred_image = pred_image.astype(np.float32)
        gt_image = gt_image.astype(np.float32)
        print(f'{pred_file} shape: {pred_image.shape}')
        print(f'{gt_file} shape: {gt_image.shape}')

        pred_image = np.squeeze(pred_image)
        gt_image = np.squeeze(gt_image)

        # Resize prediction to match ground truth
        pred_image_resized = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Create mask where ground truth has valid data
        mask = gt_image > 0

        # Replace zeros in prediction to avoid division by zero or log of zero
        pred_image_resized[pred_image_resized == 0] = 1e-6

        # Compute scaling factor using masked values
        # Normalize pred_image_resized to [0, 1]
        #pred_image_resized = (pred_image_resized - np.min(pred_image_resized)) / (np.max(pred_image_resized) - np.min(pred_image_resized))

        # Compute scaling factor using masked values
        scale = np.median(gt_image[mask]) / np.median(pred_image_resized[mask])
        pred_image_scaled = pred_image_resized * scale
        # scale = np.median(gt_image[mask]) / np.median(pred_image_resized[mask])
        # pred_image_scaled = pred_image_resized * scale
        scales.append(scale)

        # Ensure predictions are positive in masked area
        pred_image_scaled[mask][pred_image_scaled[mask] <= 0] = 1e-6
        # Plot pred_image_scaled[mask] and gt_image[mask] side by side
        print(f'{pred_file} scaled shape: {pred_image_scaled.shape}')

        # Evaluate metrics with scaled predictions, only on masked pixels
        rmse = compute_rmse(pred_image_scaled[mask], gt_image[mask])
        rmses.append(rmse)
        mae = compute_mae(pred_image_scaled[mask], gt_image[mask])
        maes.append(mae)
        sq_rel = compute_sq_rel(pred_image_scaled[mask], gt_image[mask])
        sq_rels.append(sq_rel)
        delta_accuracy = compute_delta_accuracy(pred_image_scaled[mask], gt_image[mask])
        delta_accuracies.append(delta_accuracy)
        ssim_value = compute_ssim(pred_image_scaled, gt_image, mask)
        ssims.append(ssim_value)
        log_rmse = compute_log_rmse(pred_image_scaled[mask], gt_image[mask])
        log_rmses.append(log_rmse)

                
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(12, 6))
        # masked_prediction_image = np.zeros_like(pred_image_scaled)
        # masked_prediction_image[mask] = pred_image_scaled[mask]
        # plt.subplot(1, 2, 1)
        # plt.imshow(masked_prediction_image, cmap='rainbow')
        # plt.title('Predicted Depth Map')
        # plt.colorbar()

        # plt.subplot(1, 2, 2)
        # plt.imshow(gt_image, cmap='rainbow')
        # plt.title('Ground Truth Depth Map')
        # plt.colorbar()
        # plt.text(0.5, -0.3, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nSq Rel: {sq_rel:.4f}\nDelta Accuracy: {delta_accuracy:.4f}\nSSIM: {ssim_value:.4f}\nLog RMSE: {log_rmse:.4f}', 
        #      ha='center', va='center', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        # plt.tight_layout()
        # plt.show()
        

    mean_scales = np.mean(scales)
    std_scales = np.std(scales)
    mean_rmse = np.mean(rmses)
    mean_mae = np.mean(maes)
    mean_sq_rel = np.mean(sq_rels)
    mean_delta_accuracy = np.mean(delta_accuracies)
    mean_ssim = np.mean(ssims)
    mean_log_rmse = np.mean(log_rmses)
    print(f"Scale Mean: {mean_scales}")
    print(f"Scale SD: {std_scales}")
    print(f"Mean RMSE: {mean_rmse}")
    print(f"Mean MAE: {mean_mae}")
    print(f"Mean Sq Rel: {mean_sq_rel}")
    print(f"Mean Delta Accuracy: {mean_delta_accuracy}")
    print(f"Mean SSIM: {mean_ssim}")
    print(f"Mean Log RMSE: {mean_log_rmse}")

    # Plotting the metrics
    import matplotlib.pyplot as plt

    num_images = len(pred_files)
    x_axis = range(1, num_images + 1)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 4, 1)
    plt.plot(x_axis, rmses, marker='o')
    plt.title('RMSE')
    plt.xlabel('Image Index')
    plt.ylabel('RMSE')

    plt.subplot(2, 4, 2)
    plt.plot(x_axis, maes, marker='o')
    plt.title('MAE')
    plt.xlabel('Image Index')
    plt.ylabel('MAE')

    plt.subplot(2, 4, 3)
    plt.plot(x_axis, sq_rels, marker='o')
    plt.title('Squared Relative Error')
    plt.xlabel('Image Index')
    plt.ylabel('Sq Rel')

    plt.subplot(2, 4, 4)
    plt.plot(x_axis, delta_accuracies, marker='o')
    plt.title('Delta Accuracy')
    plt.xlabel('Image Index')
    plt.ylabel('Delta Accuracy')

    plt.subplot(2, 4, 5)
    plt.plot(x_axis, ssims, marker='o')
    plt.title('SSIM')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM')

    plt.subplot(2, 4, 6)
    plt.plot(x_axis, log_rmses, marker='o')
    plt.title('Log RMSE')
    plt.xlabel('Image Index')
    plt.ylabel('Log RMSE')

    plt.subplot(2, 4, 7)
    plt.plot(x_axis, scales, marker='o')
    plt.title('Scaling Factor')
    plt.xlabel('Image Index')
    plt.ylabel('Scale')

    plt.tight_layout()
    plt.show()

    return [mean_rmse, mean_mae, mean_sq_rel, mean_delta_accuracy, mean_ssim, mean_log_rmse]

# Example usage:
#evaluate_depth_maps(pred_folder='/home/yicheng/Github/pipeline/test_npy_eval', gt_folder='data/Hamlyn/rectified01/depth01')
#evaluate_depth_maps(pred_folder='/home/yicheng/Github/pipeline/output/endo-depth_20241016153512_output', gt_folder='data/Hamlyn/rectified01/depth01')
#evaluate_depth_maps(pred_folder='/home/yicheng/Github/pipeline/test_visible_npy_eval',gt_folder='data/Hamlyn/rectified17/depth01')
#evaluate_depth_maps(pred_folder='/home/yicheng/Github/pipeline/output/endo-depth_20241023122510_output',gt_folder='data/Hamlyn/rectified17/depth01')
#evaluate_depth_maps(pred_folder='/home/yicheng/Github/pipeline/test_npy_eval',gt_folder='data/Hamlyn/rectified17/depth01')
#outputfolder = 'depth-anything:v2_20241028171332_output'
#evaluate_depth_maps(pred_folder=f'/home/yicheng/Github/pipeline/output/{outputfolder}',gt_folder='data/Hamlyn/rectified17/depth01')