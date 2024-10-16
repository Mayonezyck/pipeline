import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

def compute_rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

def compute_mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def compute_sq_rel(pred, gt):
    valid_mask = (gt > 0) & (gt < np.percentile(gt, 99))
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    return np.mean(((pred - gt) ** 2) / gt)

def compute_delta_accuracy(pred, gt, threshold=1.25):
    max_ratio = np.maximum(pred / gt, gt / pred)
    return np.mean(max_ratio < threshold)

def compute_ssim(pred, gt):
    data_range = max(np.max(pred), np.max(gt))
    result, _ = ssim(pred, gt, data_range= data_range, full=True)
    return result

def compute_psnr(pred, gt):
    return cv2.PSNR(pred, gt)

def compute_log_rmse(pred, gt):
    return np.sqrt(np.mean((np.log(pred + 1e-6) - np.log(gt + 1e-6)) ** 2))

def evaluate_depth_maps(pred_folder='temp', gt_folder = '/home/yicheng/Github/pipeline/data/Hamlyn/rectified01/depth01'):
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('_depth.png')])
    gt_files = [f.replace('_depth', '') for f in pred_files]
    if len(pred_files) != len(gt_files):
        raise ValueError("The number of prediction files and ground truth files do not match.")

    rmses = []
    maes = []
    sq_rels = []
    delta_accuracies = []
    ssims = []
    psnrs = []
    log_rmses = []
    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, gt_file)

        pred_image = load_image(pred_path)
        gt_image = load_image(gt_path)
        if pred_image.ndim == 3 and pred_image.shape[2] == 3:
            pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2GRAY)
        if gt_image.ndim == 3 and gt_image.shape[2] == 3:
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
        pred_image = pred_image.astype(np.float32)
        pred_image[pred_image == 0] = 1e-6
        gt_image = gt_image.astype(np.float32)
        gt_image[gt_image == 0] = 1e-6
        rmse = compute_rmse(pred_image, gt_image)
        rmses.append(rmse)
        mae = compute_mae(pred_image, gt_image)
        maes.append(mae)
        sq_rel = compute_sq_rel(pred_image, gt_image)
        sq_rels.append(sq_rel)
        delta_accuracy = compute_delta_accuracy(pred_image, gt_image)
        delta_accuracies.append(delta_accuracy)
        ssim = compute_ssim(pred_image, gt_image)
        ssims.append(ssim)
        #psnr = compute_psnr(pred_image, gt_image)
        #psnrs.append(psnr)
        log_rmse = compute_log_rmse(pred_image, gt_image)
        log_rmses.append(log_rmse)

    mean_rmse = np.mean(rmses)
    mean_mae = np.mean(maes)
    mean_sq_rel = np.mean(sq_rels)
    mean_delta_accuracy = np.mean(delta_accuracies)
    mean_ssim = np.mean(ssims)
    #mean_psnr = np.mean(psnrs)
    mean_log_rmse = np.mean(log_rmses)
    print(f"Mean RMSE: {mean_rmse}")
    print(f"Mean MAE: {mean_mae}")
    print(f"Mean Sq Rel: {mean_sq_rel}")
    print(f"Mean Delta Accuracy: {mean_delta_accuracy}")
    print(f"Mean SSIM: {mean_ssim}")
    #print(f"Mean PSNR: {mean_psnr}")
    print(f"Mean Log RMSE: {mean_log_rmse}")
    return [mean_rmse, mean_mae, mean_sq_rel, mean_delta_accuracy, mean_ssim, mean_log_rmse]
# Example usage:
# evaluate_depth_maps(pred_folder= '/home/yicheng/Github/pipeline/output/endo-depth_20241010123140_output')