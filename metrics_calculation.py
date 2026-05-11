import os
import argparse
import torch
import pyiqa # type: ignore
import cv2
from basicsr.utils import img2tensor
import pandas as pd
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=False, default="", help="Ground truth image directory")
    parser.add_argument("--gen_dir", required=False, default="", help="Generated image directory")
    parser.add_argument("--save_dir", required=False, default="", help="Directory to save metrics results")
    args = parser.parse_args()

    REF_DIR = args.gt_dir
    SR_DIR  = args.gen_dir
    SAVE_DIR = args.save_dir

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMG_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    METRICS = ['psnr', 'ssim', 'lpips', 'dists', 'niqe', 'clipiqa', 'musiq', 'maniqa']

    metricators = {
        'psnr': pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device),
        'ssim': pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device),
        'lpips': pyiqa.create_metric('lpips', device=device).to(device),
        'dists': pyiqa.create_metric('dists', device=device).to(device),
        'niqe': pyiqa.create_metric('niqe', device=device).to(device)
    }

    sr_files = sorted([f for f in os.listdir(SR_DIR) if os.path.splitext(f)[1].lower() in IMG_EXTS])
    results = []

    for fname in tqdm(sr_files, desc="Processing images"):
        sr_path = os.path.join(SR_DIR, fname)
        ref_path = os.path.join(REF_DIR, fname)
        if not os.path.isfile(ref_path):
            print(f"Reference missing for {fname}, skipping.")
            continue

        sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)

        if sr_img is None or ref_img is None:
            print(f"Error reading {fname}, skipping.")
            continue

        sr = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0
        ref = img2tensor(ref_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0

        row = {'image': fname}

        # Full-reference metrics
        for m in ['psnr', 'ssim', 'lpips', 'dists']:
            metric_fn = metricators.get(m)
            try:
                score = metric_fn(sr, ref)
                row[m] = float(score.cpu().item())
            except Exception as e:
                print(f"Error computing {m} on {fname}: {e}")
                row[m] = None

        # No-reference metrics
        for m in ['niqe', 'clipiqa', 'musiq', 'maniqa']:
            metric_fn = metricators.get(m)
            try:
                score = metric_fn(sr)
                row[m] = float(score.cpu().item())
            except Exception as e:
                print(f"Error computing {m} on {fname}: {e}")
                row[m] = None

        results.append(row)


    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_DIR, "iqa_metrics_per_image.csv"), index=False)
    print("Saved per-image metrics to iqa_metrics_per_image.csv")


    mean_metrics = {}
    for col in df.columns:
        if col == 'image':
            continue
        valid_values = df[col].dropna()
        mean_metrics[col] = valid_values.mean() if len(valid_values) > 0 else None


    try:
        fid_metric = pyiqa.create_metric('fid', device=device)
        fid_value = fid_metric(REF_DIR, SR_DIR)
        mean_metrics['FID'] = float(fid_value.cpu().item()) if hasattr(fid_value, 'cpu') else float(fid_value)
    except Exception as e:
        print("Error calculating FID:", e)
        mean_metrics['FID'] = None


    print("\n=== Final Average per Metric ===")
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")


    df_mean = pd.DataFrame([mean_metrics])
    df_mean.to_csv(os.path.join(SAVE_DIR, "iqa_metrics_mean.csv"), index=False)
    print("Saved mean metrics to iqa_metrics_mean.csv")


if __name__ == "__main__":
    main()
