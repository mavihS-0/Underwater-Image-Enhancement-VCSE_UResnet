import os
import math
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def ssim_compare(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    dim = (6022, 5513)
    img1 = cv2.resize(img1, dim)
    img2 = cv2.resize(img2, dim)
    ssim_score, _ = ssim(img1, img2, full=True)
    return ssim_score

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_metrics(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    psnr = calculate_psnr(img1, img2)
    ssim_value = ssim_compare(img1_path, img2_path)
    return psnr, ssim_value

def main():
    psnr_list = []
    ssim_list = []

    to_compare_dir = 'to_compare'
    original_dir = 'original'

    for filename in os.listdir(to_compare_dir):
        base_name = os.path.splitext(filename)[0]
        original_filename = f"processed_{base_name}-output.png"
        original_path = os.path.join(original_dir, original_filename)
        compare_path = os.path.join(to_compare_dir, filename)
        
        if os.path.exists(original_path):
            psnr, ssim_value = calculate_metrics(compare_path, original_path)
            psnr_list.append(psnr)
            ssim_list.append(float(ssim_value))
            
        else:
            print(f"Original image for {filename} not found in {original_dir}")

    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
    avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0
    
    print("\nPSNR List:", psnr_list)
    print("SSIM List:", ssim_list)
    print(f"\nAverage PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == '__main__':
    main()
