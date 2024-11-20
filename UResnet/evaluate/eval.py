import os
import cv2
import numpy as np
from skimage import img_as_float
from metrics import metric_functions  # Ensure custom_metrics is the filename where these functions are saved

def load_image(img_path):
    """Load and convert the image to float."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at {img_path} could not be loaded.")
    return img_as_float(img)

def calculate_metrics(org_img, pred_img):
    """Calculate all metrics between original and predicted images."""
    results = {}
    for metric_name, metric_func in metric_functions.items():
        try:
            results[metric_name] = metric_func(org_img, pred_img)
        except Exception as e:
            print(f"Error calculating {metric_name}: {e}")
            results[metric_name] = None
    return results

def main():
    to_compare_dir = 'to_compare'
    original_dir = 'original'
    results_list = []

    for filename in os.listdir(to_compare_dir):
        base_name = os.path.splitext(filename)[0]
        original_filename = f"processed_{base_name}-output.png"
        original_path = os.path.join(original_dir, original_filename)
        compare_path = os.path.join(to_compare_dir, filename)

        if os.path.exists(original_path):
            org_img = load_image(original_path)
            print(1)
            pred_img = load_image(compare_path)
            print(2)
            metrics = calculate_metrics(org_img, pred_img)
            print(metrics)
            results_list.append((filename, metrics))
        else:
            print(f"Original image for {filename} not found in {original_dir}")

    # Print results for each image pair
    for filename, metrics in results_list:
        print(f"\nMetrics for {filename}:")
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                print(f"{metric_name.upper()}: {metric_value:.4f}")
            else:
                print(f"{metric_name.upper()}: Calculation error")

    # Calculate average for each metric across all images
    if results_list:
        avg_metrics = {metric: 0 for metric in results_list[0][1].keys()}
        valid_counts = {metric: 0 for metric in results_list[0][1].keys()}
        for _, metrics in results_list:
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    avg_metrics[metric_name] += metric_value
                    valid_counts[metric_name] += 1

        print("\nAverage Metrics Across All Images:")
        for metric_name, total in avg_metrics.items():
            if valid_counts[metric_name] > 0:
                print(f"{metric_name.upper()}: {total / valid_counts[metric_name]:.4f}")
            else:
                print(f"{metric_name.upper()}: No valid calculations")

if __name__ == '__main__':
    main()
