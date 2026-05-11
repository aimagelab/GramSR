import os
import argparse
import torch
from torchvision import transforms
from tqdm import tqdm

#from pisasr_emb_siglip2b import PiSASR
from gramsr import GramSR
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from src.datasets.dataset import TestDatasetv2


def main(args):
    pred_dir = os.path.join(args.output_dir, f"pred")
    gt_dir = os.path.join(args.output_dir, f"GT")

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print(f"[INFO] Pred dir: {pred_dir}")
    print(f"[INFO] GT dir:   {gt_dir}")

    model = GramSR(args)

    model.add_dino()
    model.load_dino(args.lora_dir, 53001)
    model.unet.set_adapter([
        'default_encoder_pix', 'default_decoder_pix', 'default_others_pix',
        'default_encoder_sem', 'default_decoder_sem', 'default_others_sem',
        'default_unet_dino'
    ])

    model.to("cuda")
    model.eval()


    dataset = TestDatasetv2(args, hr_folder=args.HR_test_folder, lr_folder=args.LR_test_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            x_src = batch["lr"].cuda()
            x_tgt = batch["hr"].cuda()
            x_basename = batch["base_name"][0]

            assert x_src.shape[0] == 1, "Batch size must be 1"

            # Prompt vuoto
            batch["prompt"] = [""]

            # Forward
            x_tgt_pred, *_ = model(x_src, x_tgt, batch=batch, args=args)
            #torch.save(x_tgt_pred, f"tensor_main_{x_basename.replace('.png', '.pt')}")

            # [-1,1] -> [0,1]
            x_tgt_pred = x_tgt_pred * 0.5 + 0.5
            x_src = x_src * 0.5 + 0.5
            x_tgt = x_tgt * 0.5 + 0.5

            # Remove batch dim
            x_tgt_pred = x_tgt_pred[0].cpu()
            x_src = x_src[0].cpu()
            x_tgt = x_tgt[0].cpu()

            # PIL
            output_pil = transforms.ToPILImage()(x_tgt_pred)
            input_pil = transforms.ToPILImage()(x_src)

            # Color alignment
            if args.align_method == "adain":
                output_pil = adain_color_fix(
                    target=output_pil,
                    source=input_pil
                )
            elif args.align_method == "wavelet":
                output_pil = wavelet_color_fix(
                    target=output_pil,
                    source=input_pil
                )

            # Save SR
            output_pil.save(os.path.join(pred_dir, x_basename))

            # Save GT
            #gt_pil = transforms.ToPILImage()(x_tgt)
            #gt_pil.save(os.path.join(gt_dir, x_basename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--lora_dir", type=str, default="./weights/checkpoints")
    parser.add_argument("--LR_test_folder", type=str, default="./LR")
    parser.add_argument("--HR_test_folder", type=str, default="/work/baraldimaticad/icpr_sr/osediff_dataset/RealSR/test_HR")


    parser.add_argument("--pretrained_model_path", type=str, default="/your/path/to/stable-diffusion-2-1-base")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)

    parser.add_argument("--neg_prompt", default="painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth", type=str)

    parser.add_argument("--lora_rank_unet_pix", type=int, default=4)
    parser.add_argument("--lora_rank_unet_sem", type=int, default=4)
    parser.add_argument("--lora_rank_unet_dino", type=int, default=4)

    parser.add_argument("--timesteps1", type=float, default=1)
    parser.add_argument("--null_text_ratio", type=float, default=0.0)

    parser.add_argument("--align_method", type=str,
                        choices=["wavelet", "adain", "nofix"],
                        default="adain")

    args = parser.parse_args()

    main(args)
