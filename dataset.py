import os
import random
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from pathlib import Path

import numpy as np
from src.datasets.realesrgan import RealESRGAN_degradation



class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split

        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
                transforms.RandomHorizontalFlip(),
            ])
            with open(args.dataset_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]

        elif split == 'test':
            '''self.input_folder = os.path.join(args.dataset_test_folder, "X4_bicubic/val")
            self.output_folder = os.path.join(args.dataset_test_folder, "HR/val")
            self.lr_list = []
            self.gt_list = []
            lr_names = os.listdir(os.path.join(self.input_folder))
            gt_names = os.listdir(os.path.join(self.output_folder))
            assert len(lr_names) == len(gt_names)
            for i in range(len(lr_names)):
                # ricostruisci lr_name a partire da gt_name
                lr_name = gt_names[i].replace(".png", "x4.png")
                self.lr_list.append(os.path.join(self.input_folder, lr_name))
                self.gt_list.append(os.path.join(self.output_folder, gt_names[i]))

            self.crop_preproc = transforms.Compose([
                transforms.CenterCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])
            assert len(self.lr_list) == len(self.gt_list)'''

            self.crop_preproc = transforms.Compose([
                transforms.CenterCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])

            with open(args.dataset_val_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]
            self.lr_list = []
            for i in range(len(self.gt_list)):
                filename = os.path.basename(self.gt_list[i])
                if filename.startswith('FFHQ_'):
                    base_path = "/work/tesi_fdoronzio/dataset/FFHQ/val/X4_bicubic"
                else:
                    base_path = "/work/tesi_fdoronzio/dataset/LSDIR/val/val1/X4_bicubic/val"
                self.lr_list.append(os.path.join(base_path, filename.replace(".png", "x4.png")))



    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            img_t = self.crop_preproc(input_img)
            output_t = self.crop_preproc(output_img)
            # input images scaled to -1, 1
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            return example


class Clamp:
    def __call__(self, img):
        return img.clamp(min=0, max=1)

def normalize(img):
        return (img - 0.5) * 2

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args, file_list=None):
        super().__init__()
        self.args = args
        self.directories = None
        if file_list is not None:
            self.directories = [os.path.dirname(p) for p in file_list]
            self.gt_list = [os.path.basename(p) for p in file_list]
        else:
            self.input_folder = args.dataset_test_folder
            self.gt_list = os.listdir(self.input_folder)
        self.upscale = args.upscale
        self.crop_HR = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((args.process_size, args.process_size))
        ])
        self.crop_LR = transforms.Compose([
            transforms.Resize((args.process_size//self.upscale, args.process_size//self.upscale),InterpolationMode.BICUBIC),
            transforms.Resize((args.process_size, args.process_size), InterpolationMode.BICUBIC),
            Clamp()
        ])
    
    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        filename = self.gt_list[idx]
        if self.directories is not None:
            file_path = os.path.join(self.directories[idx], filename)
        else:
            file_path = os.path.join(self.input_folder, filename)
        input_img = Image.open(file_path).convert('RGB')
        hr = self.crop_HR(input_img)
        lr = self.crop_LR(hr)
        hr = normalize(hr)
        lr = normalize(lr)

        example = {}
        example["neg_prompt"] = self.args.neg_prompt
        example["null_prompt"] = ""
        example["hr"] = hr
        example["lr"] = lr
        example["base_name"] = filename

        return example


class TestDatasetv2(torch.utils.data.Dataset):
    def __init__(self, args, hr_folder=None, lr_folder=None, file_list=None):
        super().__init__()
        self.args = args

        # Cartelle HR e LR
        self.hr_folder = hr_folder if hr_folder is not None else os.path.join(args.dataset_test_folder, "HR")
        self.lr_folder = lr_folder if lr_folder is not None else os.path.join(args.dataset_test_folder, "LR")

        # Lista dei file da processare
        if file_list is not None:
            self.hr_list = [os.path.basename(p) for p in file_list]
        else:
            self.hr_list = sorted(os.listdir(self.hr_folder))

        # Trasformazioni HR (ritaglio centrale + tensor)
        self.crop_HR = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((args.process_size, args.process_size))
        ])

        # Trasformazioni LR (solo tensor e ritaglio centrale, nessun resize)
        self.crop_LR = transforms.Compose([
        transforms.Resize((args.process_size, args.process_size), 
                        InterpolationMode.BICUBIC),
        transforms.ToTensor(),
])

    def __len__(self):
        return len(self.hr_list)

    def __getitem__(self, idx):
        filename = self.hr_list[idx]

        # Percorsi HR e LR
        hr_path = os.path.join(self.hr_folder, filename)
        lr_path = os.path.join(self.lr_folder, filename)

        # Caricamento immagini
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        # Applicazione trasformazioni
        hr = self.crop_HR(hr_img)
        lr = self.crop_LR(lr_img)

        # Normalizzazione
        hr = normalize(hr)
        lr = normalize(lr)

        return {
            "neg_prompt": self.args.neg_prompt,
            "null_prompt": "",
            "hr": hr,
            "lr": lr,
            "base_name": filename
        }
