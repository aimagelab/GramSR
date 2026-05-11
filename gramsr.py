import os
import sys
import time
import random
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel, DINOv3ViTModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules, ModulesToSaveWrapper
from peft.utils.other import transpose
from safetensors.torch import load_file
from torchvision import transforms
from adapters import *

sys.path.append(os.getcwd())
#from src.models.autoencoder_kl import AutoencoderKL
#from src.models.unet_2d_condition import UNet2DConditionModel
from src.my_utils.vaehook import VAEHook

from diffusers import AutoencoderKL, UNet2DConditionModel

import glob
def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None


import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def initialize_unet(rank_pix, rank_sem, return_lora_module_names=False, pretrained_model_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix = [], [], []
    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder_pix.append(n.replace(".weight",""))
                l_target_modules_decoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others_pix.append(n.replace(".weight",""))
                l_modules_others_sem.append(n.replace(".weight",""))
                break

    lora_conf_encoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_pix)
    lora_conf_decoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_pix)
    lora_conf_others_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_modules_others_pix)
    lora_conf_encoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_sem)
    lora_conf_decoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_sem)
    lora_conf_others_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_modules_others_sem)

    unet.add_adapter(lora_conf_encoder_pix, adapter_name="default_encoder_pix")
    unet.add_adapter(lora_conf_decoder_pix, adapter_name="default_decoder_pix")
    unet.add_adapter(lora_conf_others_pix, adapter_name="default_others_pix")
    unet.add_adapter(lora_conf_encoder_sem, adapter_name="default_encoder_sem")
    unet.add_adapter(lora_conf_decoder_sem, adapter_name="default_decoder_sem")
    unet.add_adapter(lora_conf_others_sem, adapter_name="default_others_sem")

    if return_lora_module_names:
        return unet, l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix, l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem
    else:
        return unet


class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        
        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)
            noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)
            

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))

        return loss

    def stopgrad(self, x):
        return x.detach()



class DinoEnbedder(torch.nn.Module):
    def __init__(self, path, checkpoint = 0, device = 'cuda'):
        super().__init__()
        self.path = path
        self.model = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitb16-pretrain")
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.model.to(device)
        self.model.requires_grad_(False)
        self.model.eval()

        self.adapter = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )
        #self.adapter = ABMIL()
        self.adapter.to(device)

        if checkpoint > 0:
            self.load_adapter(checkpoint)
        
    
    def forward(self, img):
        x = self.normalize(img * 0.5 + 0.5)
        x_feat = self.model(x).last_hidden_state
        output = self.adapter(x_feat)
        return output


    def save_adapter(self, checkpoint):
        adapter_weights = {name: param.data for name, param in self.adapter.named_parameters()}
        torch.save(adapter_weights, os.path.join(self.path, f'adapter_{checkpoint}.pth'))
    
    def load_adapter(self, checkpoint):
        adapter_weights = torch.load(os.path.join(self.path, f'adapter_{checkpoint}.pth'))
        for name, param in adapter_weights.items():
            if name in self.adapter.state_dict():
                self.adapter.state_dict()[name].copy_(param)
    
    def freeze_adapter(self):
        self.adapter.requires_grad_(False)




class GramSR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.embedder = DinoEnbedder(path=args.lora_dir, checkpoint=53001)

        ckpt_path = os.path.join(args.lora_dir, f'model_{12501}.pkl')
        print(f'====> resume from {ckpt_path}')
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        self.lora_rank_unet_pix = args.lora_rank_unet_pix
        self.lora_rank_unet_sem = args.lora_rank_unet_sem
        self.lora_rank_unet_dino = args.lora_rank_unet_dino
        GramSR = torch.load(ckpt_path)
        self.load_ckpt_from_state_dict(GramSR)
        # unet.enable_xformers_memory_efficient_attention()
        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.vae_fix.to('cuda')

        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()
        
        self.neg_prompt_embeds = torch.zeros((1, 1029, 1024), device='cuda')

    def freeze_emb(self):
        self.embedder.freeze_adapter()

    def set_train_pix(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "pix" in n:
                _p.requires_grad = True
            if "sem" in n:
                _p.requires_grad = False
    
    def set_train_sem(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sem" in n:
                _p.requires_grad = True
            if "pix" in n:
                _p.requires_grad = False
    
    def set_train_dino(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "dino" in n:
                _p.requires_grad = True
            elif "sem" in n or "pix" in n:
                _p.requires_grad = False

    def add_dino(self):
        unet_lora_config_dino = LoraConfig(
            r=self.lora_rank_unet_dino,
            lora_alpha=self.lora_rank_unet_dino,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"],
        )
        self.unet.add_adapter(unet_lora_config_dino, adapter_name="default_unet_dino")

    def _load_lora_ckpt(self, cpkt, adapter_name):
        for n, p in self.unet.named_parameters():
            if adapter_name in n:
                name = n.replace(f".{adapter_name}", "")
                p.data.copy_(cpkt[name])

    def load_dino(self, input_dir, resume):
        #self.unet.load_lora_adapter(input_dir, weight_name=f"lora_weights_dino_{resume}.safetensors", adapter_name="default_unet_dino")
        lora_dino = load_file(os.path.join(input_dir, f"lora_weights_dino_{resume}.safetensors"))
        self._load_lora_ckpt(lora_dino, "default_unet_dino")

    def save_dino(self, out_dir, steps):
        self.unet.save_lora_adapter(save_directory=out_dir, adapter_name="default_unet_dino", weight_name=f"lora_weights_dino_{steps}.safetensors")
        self.embedder.save_adapter(steps)

    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"])
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"])
        self.lora_conf_others_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"])

        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")

        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix, \
        self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
        sd["unet_lora_encoder_modules_pix"], sd["unet_lora_decoder_modules_pix"], sd["unet_lora_others_modules_pix"], \
            sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])



    def forward(self, c_t, c_tgt, batch=None, args=None):

        bs = c_t.shape[0]
        #encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        # Using mode() instead of sample() for reproducibility (adjusted LoRA may break with forward())
        encoded_control = self.vae_fix.encode(c_t).latent_dist.mode() * self.vae_fix.config.scaling_factor

        #prompt_embeds = self.learnable_embeds(bs)#self.prompt_embeds.repeat(bs, 1, 1)
        emb = self.embedder(c_t)

        neg_prompt_embeds = self.neg_prompt_embeds.repeat(bs, 1, 1)

        model_pred = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=emb.to(torch.float32),).sample
        x_denoised = encoded_control - model_pred
        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, emb, neg_prompt_embeds
    

    def adjust_lora(self, c_t, l1, l2, l3):
        #encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        # Using mode() instead of sample() for reproducibility (doesn't work with standard forward())
        encoded_control = self.vae_fix.encode(c_t).latent_dist.mode() * self.vae_fix.config.scaling_factor
        emb = self.embedder(c_t)

        # enable only pixel lora
        self.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix'])
        model_pred_lora1 = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=emb.to(torch.float32),).sample

        # enable pixel and semantic lora
        self.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix','default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
        model_pred_lora12 = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=emb.to(torch.float32),).sample

        # enable all lora
        self.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix','default_encoder_sem', 'default_decoder_sem', 'default_others_sem', 'default_unet_dino'])
        model_pred_lora123 = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=emb.to(torch.float32),).sample

        model_pred = l1 * model_pred_lora1 + l2 * (model_pred_lora12 - model_pred_lora1) + l3 * (model_pred_lora123 - model_pred_lora12)
        x_denoised = encoded_control - model_pred
        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return output_image


    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_pix"], sd["unet_lora_decoder_modules_pix"], sd["unet_lora_others_modules_pix"] =\
            self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix
        sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"] =\
            self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem
        sd["lora_rank_unet_pix"] = self.lora_rank_unet_pix
        sd["lora_rank_unet_sem"] = self.lora_rank_unet_sem
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        torch.save(sd, outf)
        ckpt = os.path.basename(outf).split('.')[0]
        ckpt = ckpt.split('_')[1]
        self.embedder.save_adapter(ckpt)



        # Extension of load_dino: loads the dict into memory instead of applying it immediately
    def load_lora_ckpt_into_memory(self, filepath, adapter_name):
        """
        Load file (safetensors / torch) and save the raw dict in self._loaded_lora_ckpts[adapter_name]
        The dict must have keys corresponding to base names (without .<adapter_name>)
        """
        cpkt = load_file(filepath)  # as you use it
        # Normalize tensors to cuda/float32 for faster summation later
        for k, v in cpkt.items():
            cpkt[k] = v.to(torch.float32).to("cuda")
        self._loaded_lora_ckpts[adapter_name] = cpkt
        print(f"[load_lora_ckpt_into_memory] loaded {len(cpkt)} tensors for adapter {adapter_name}")

    # Utility to list which keys exist in the loaded checkpoints
    def list_loaded_loras(self):
        return {k: len(v) for k, v in self._loaded_lora_ckpts.items()}

    def apply_lora_weights(self, adapter_scales: dict):
        """
        adapter_scales: dict mapping adapter_name -> scale (float).
        Example: {"default_encoder_pix": 0.6, "default_encoder_sem": 0.3, "default_unet_dino": 0.1}
        This combines the various checkpoints already loaded in memory and overwrites the corresponding
        model parameters (parameters containing the adapter_name substring).
        """
        # Quick checks
        if not adapter_scales:
            return
        # For numerical stability / speed, ensure all ckpts in cuda float32
        # Iterate over model named_parameters and match parameters containing an adapter_name
        with torch.no_grad():
            # For speed: pre-compute list of adapter names present in memory
            loaded_adapters = set(self._loaded_lora_ckpts.keys())
            # For each model parameter containing "lora" and one of the adapters
            for n, p in self.unet.named_parameters():
                if "lora" not in n:
                    continue
                # Find which adapter is in this name (there could be multiple adapter substrings; use the first match)
                matched_adapter = None
                for adapter_name in adapter_scales.keys():
                    if f".{adapter_name}." in n or n.endswith(f".{adapter_name}"):
                        matched_adapter = adapter_name
                        break
                if matched_adapter is None:
                    # No adapter among the requested ones in this name; skip
                    continue

                # Base key (name present in cpkt dict) = remove .<adapter_name> from the name
                base_key = n.replace(f".{matched_adapter}", "")
                # Build the combined tensor: weighted sum over all requested adapters
                # WARNING: some adapters may not have this key: we consider it as zero in that case
                combined = None
                # Sum over all adapters passed in adapter_scales (not just matched_adapter)
                # This allows using a single function that combines all adapters mapping to the same layer
                # but in most cases only the specific adapter will contain the key.
                for adapter_name, scale in adapter_scales.items():
                    if scale == 0.0:
                        continue
                    ckpt = self._loaded_lora_ckpts.get(adapter_name, None)
                    if ckpt is None:
                        # Adapter not loaded in memory; skip
                        continue
                    # The ckpt uses the base_key (without suffix)
                    if base_key not in ckpt:
                        # The key doesn't exist in this checkpoint
                        continue
                    tensor = ckpt[base_key]
                    # Cast/move to the parameter's device
                    if tensor.device != p.data.device:
                        tensor = tensor.to(p.data.device)
                    if combined is None:
                        combined = tensor.mul(scale)
                    else:
                        combined = combined + tensor.mul(scale)
                if combined is None:
                    # None of the adapters had the key: set to zero (or leave the current value)
                    # Better to zero if you want the LoRA component to be null; otherwise leave p as is
                    p.data.zero_()
                else:
                    # Overwrite the LoRA parameter with the combination
                    p.data.copy_(combined.to(p.data.dtype))
        # End apply_lora_weights

    # Utility: reset all LoRA parameters to zero (useful to disable all adapters)
    def zero_all_lora_params(self):
        with torch.no_grad():
            for n, p in self.unet.named_parameters():
                if "lora" in n:
                    p.data.zero_()





