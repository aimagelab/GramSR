import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

import numpy as np
from src.datasets.realesrgan import RealESRGAN_degradation



class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None, resume=False):
        super().__init__()

        self.args = args
        self.split = split
        self.resume = resume
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        if split == 'train':
            self.dest_embeddings = "/work/tesi_fdoronzio/dataset/LSDIR/text_embeddings/train"
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
                transforms.RandomHorizontalFlip(),
            ])
            with open(args.dataset_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]
            if args.highquality_dataset_txt_paths is not None:
                with open(args.highquality_dataset_txt_paths, 'r') as f:
                    self.hq_gt_list = [line.strip() for line in f.readlines()]
            #self.make_dirs(self.gt_list)
            if resume :
                self.gt_list = self.filter_from_list(self.gt_list)

        elif split == 'test':
            self.dest_embeddings = "/work/tesi_fdoronzio/dataset/LSDIR/text_embeddings/val"
            #self.input_folder = os.path.join(args.dataset_test_folder, "X4/val")
            self.output_folder = os.path.join(args.dataset_test_folder, "HR/val")
            #self.lr_list = []
            self.gt_list = []
            #lr_names = os.listdir(os.path.join(self.input_folder))
            gt_names = os.listdir(os.path.join(self.output_folder))
            #assert len(lr_names) == len(gt_names)
            for i in range(len(gt_names)):#lr_names
                #self.lr_list.append(os.path.join(self.input_folder, lr_names[i]))
                self.gt_list.append(os.path.join(self.output_folder,gt_names[i]))
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])
            self.make_dirs(self.gt_list)
            if resume:
                self.gt_list = self.filter_from_list(self.gt_list)
    
    def get_embedding_dest(self, path):
        path = path[:-4].split('/') #remove extension
        return os.path.join(self.dest_embeddings, path[-2], path[-1])

    def make_dirs(self, file_list):
        folder_set = set()
        for p in file_list:
            path_component = p.split('/')
            new_path = os.path.join(self.dest_embeddings, path_component[-2])
            folder_set.add(new_path)
        for f in folder_set:
            os.makedirs(f, exist_ok=True)
    

    def _get_subdir_names(self, file_list):
        folder_set = set()
        for p in file_list:
            path_components = p.split('/')
            new_path = path_components[-2]
            folder_set.add(new_path)
        return folder_set

    def filter_from_list(self, file_list):
        result = []
        sub_dir_set = self._get_subdir_names(file_list)
        remaining_dir = set()
        #compute source path /work/tesi_fdoronzio/dataset/LSDIR/train
        dataset_path = os.path.join('/', *file_list[0].split('/')[:-2])
        for d in sub_dir_set:
            source_path = os.path.join(dataset_path, d)
            destination_path = os.path.join(self.dest_embeddings, d)
            source_file_number = len([file for file in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, file))])
            dest_file_number = len([file for file in os.listdir(destination_path) if os.path.isfile(os.path.join(destination_path, file))])
            if 2*source_file_number != dest_file_number:
                remaining_dir.add(d)
        for p in file_list:
            if p.split('/')[-2] in remaining_dir:
                result.append(p)
        return result


    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            if self.args.highquality_dataset_txt_paths is not None:
                if np.random.uniform() < self.args.prob:
                    gt_img = Image.open(self.gt_list[idx]).convert('RGB')
                else:
                    idx = random.sample(range(0, len(self.hq_gt_list)), 1)
                    gt_img = Image.open(self.hq_gt_list[idx[0]]).convert('RGB')
            else:
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
            example["destination_embeddings"] = self.get_embedding_dest(self.gt_list[idx])

            return example
            
        elif self.split == 'test':

            #input_img = Image.open(self.lr_list[idx]).convert('RGB')
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)
            #img_t = self.crop_preproc(input_img)
            #output_t = self.crop_preproc(output_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1, 1
            #img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            #output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.gt_list[idx])#lr_list
            example["destination_embeddings"] = self.get_embedding_dest(self.gt_list[idx])

            return example
