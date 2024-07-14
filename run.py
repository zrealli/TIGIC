import argparse, os
import PIL
import torch
import re
import gc
import sys
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import cv2
import time
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import torch.nn.functional as F

def gaussian_blur_tensor(tensor, kernel_size, sigma):

    tensor_np = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()

    blurred_tensor_np = np.empty_like(tensor_np)

    for i in range(tensor_np.shape[0]):
        blurred_tensor_np[i] = cv2.GaussianBlur(tensor_np[i], kernel_size, sigma)

    blurred_tensor = torch.from_numpy(blurred_tensor_np).permute(0, 3, 1, 2)

    return blurred_tensor

def load_img(path, SCALE, pad=False, seg=False, target_size=None):
    if seg:
        # Load the input image and segmentation map
        image = Image.open(path).convert("RGB")
        seg_map = Image.open(seg).convert("1")

        # Get the width and height of the original image
        w, h = image.size

        # Calculate the aspect ratio of the original image
        aspect_ratio = h / w

        # Determine the new dimensions for resizing the image while maintaining aspect ratio
        if aspect_ratio > 1:
            new_w = int(SCALE * 256 / aspect_ratio)
            new_h = int(SCALE * 256)
        else:
            new_w = int(SCALE * 256)
            new_h = int(SCALE * 256 * aspect_ratio)

        # Resize the image and the segmentation map to the new dimensions
        image_resize = image.resize((new_w, new_h))
        segmentation_map_resize = cv2.resize(np.array(seg_map).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad the segmentation map to match the target size
        padded_segmentation_map = np.zeros((target_size[1], target_size[0]))
        start_x = (target_size[1] - segmentation_map_resize.shape[0]) // 2
        start_y = (target_size[0] - segmentation_map_resize.shape[1]) // 2
        padded_segmentation_map[start_x: start_x + segmentation_map_resize.shape[0], start_y: start_y + segmentation_map_resize.shape[1]] = segmentation_map_resize

        # Create a new RGB image with the target size and place the resized image in the center
        padded_image = Image.new("RGB", target_size)
        start_x = (target_size[0] - image_resize.width) // 2
        start_y = (target_size[1] - image_resize.height) // 2
        padded_image.paste(image_resize, (start_x, start_y))

        # Update the variable "image" to contain the final padded image
        image = padded_image
    else:
        image = Image.open(path).convert("RGB")
        w, h = image.size        
        print(f"loaded image from {path}")
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        w = h = 512
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    
    if pad or seg:
        return 2. * image - 1., new_w, new_h, padded_segmentation_map
    
    return 2. * image - 1., w, h 


def load_model_and_get_prompt_embedding(model, opt, device, prompts, inv=False):
           
    if inv:
        inv_emb = model.get_learned_conditioning(prompts, inv)
     
        c = uc = inv_emb
    else:
        # inv_emb = None
        inv_emb = model.get_learned_conditioning(prompts, inv)
     
        c = uc = inv_emb
        
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(opt.n_samples * [""])
    else:
        uc = None
    c = model.get_learned_conditioning(prompts)
        
    return c, uc, inv_emb
    
def load_model_and_get_prompt_embedding_00(model, opt, device, prompts, prompts_person ,inv=False):
           
    if inv:
        inv_emb = model.get_learned_conditioning(prompts, inv)
        inv_emb_per = model.get_learned_conditioning(prompts_person, inv)
        inv_emb = torch.cat([inv_emb[:, :-5], inv_emb_per], 1)
        c = uc = inv_emb
    else:
        inv_emb = None
        
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(opt.n_samples * [""])
    else:
        uc = None
    c = model.get_learned_conditioning(prompts)
        
    return c, uc, inv_emb
  

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=gpu)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--ref-img",
        type=list,
        nargs="?",
        help="path to the input image"
    )
    
    parser.add_argument(
        "--seg",
        type=str,
        nargs="?",
        help="path to the input image"
    )
        
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )

    parser.add_argument(
        "--dpm_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    
    parser.add_argument(
        "--f",
        type=int,
        default=16,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=3.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./ckpt/v2-1_512-ema-pruned.ckpt",
        help="path to checkpoint of model",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    parser.add_argument(
        "--root",
        type=str,
        help="",
        default='./inputs'
    )
    
    parser.add_argument(
        "--dpm_order",
        type=int,
        help="",
        choices=[1, 2, 3],
        default=2
    ) 
    
    parser.add_argument(
        "--tau_a",
        type=float,
        help="",
        default=0.5
    )
      
    parser.add_argument(
        "--tau_b",
        type=float,
        help="",
        default=0.8
    )
          
    parser.add_argument(
        "--gpu",
        type=str,
        help="",
        default='cuda:0'
    ) 
    
    opt = parser.parse_args()       
    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
        
    batch_size = opt.n_samples

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, opt.gpu)    
    model = model.to(device)
    sampler = DPMSolverSampler(model)
    count = 0
    p_path = opt.root

    for subdir, _, files in os.walk(opt.root):
        for file in files:
            torch.cuda.empty_cache()
            file_path = os.path.join(subdir, file)
            if(file_path[-5:] =='Store'):
                os.remove(file_path)
            result = re.search(r'\./examples/([^/]+)/scene\.', file_path)

         
            if result:
                prompt1 = result.group(1)
                prompt  = ''

            if file_path.endswith('.jpg') or file_path.endswith('.jpeg') or file_path.endswith('.png'):
                if file.startswith('scene'):
                    ## background
                    opt.init_img = file_path
                elif file.startswith('subject'):
                    ## subject
                    opt.ref_img = file_path
                elif file.startswith('selected_mask'):
                    ## region mask
                    opt.mask = file_path
                elif file.startswith('sub_mask'):
                    ## subject segmentation
                    opt.seg = file_path
                        
            if file == files[-1]:
            
                seed_everything(opt.seed)
                
                img = cv2.imread(opt.mask, 0)
                # Threshold the image to create binary image
                _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                # Find the contours of the white region in the image
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Find the bounding rectangle of the largest contour
                x, y, new_w, new_h = cv2.boundingRect(contours[0])
                # Calculate the center of the rectangle
                center_x = x + new_w / 2
                center_y = y + new_h / 2
                # Calculate the percentage from the top and left
                center_row_from_top = round(center_y / 512, 2)
                center_col_from_left = round(center_x / 512, 2)

                aspect_ratio = new_h / new_w
                
                if aspect_ratio > 1:  
                    scale = new_w * aspect_ratio / 256  
                    scale = new_h / 256
                else:  
                    scale = new_w / 256
                    scale = new_h / (aspect_ratio * 256) 
                        
                scale = round(scale, 2)
                
 
                assert prompt is not None
                data = [prompt]
                data1 = [prompt1]
                # read background image              
                assert os.path.isfile(opt.init_img)
                init_image, target_width, target_height = load_img(opt.init_img, scale)
                init_image = repeat(init_image.to(device), '1 ... -> b ...', b=batch_size)

                save_image = init_image.clone()
                save_image_ori = init_image.clone()
        
                ref_image, width, height, segmentation_map  = load_img(opt.ref_img, scale, seg=opt.seg, target_size=(target_width, target_height))

            
                segmentation_map_img = segmentation_map
                
                ref_image = repeat(ref_image.to(device), '1 ... -> b ...', b=batch_size)

                segmentation_map_orig = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 4 ...', b=batch_size)
                segmentation_map_save = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 3 ...', b=batch_size)
                segmentation_map = segmentation_map_orig[:, :, ::8, ::8].to(device)



                top_rr = int((0.5*(target_height - height))/target_height * init_image.shape[2])  # xx% from the top
                bottom_rr = int((0.5*(target_height + height))/target_height * init_image.shape[2])  
                left_rr = int((0.5*(target_width - width))/target_width * init_image.shape[3])  # xx% from the left
                right_rr = int((0.5*(target_width + width))/target_width * init_image.shape[3]) 

                center_row_rm = int(center_row_from_top * target_height)
                center_col_rm = int(center_col_from_left * target_width)

                step_height2, remainder = divmod(height, 2)
                step_height1 = step_height2 + remainder
                step_width2, remainder = divmod(width, 2)
                step_width1 = step_width2 + remainder
                    
                # image composition
                save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] \
                        = save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2].clone() \
                        * (1 - segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]) \
                        + ref_image[:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
                        * segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]
                
    
                # save mask
                save_mask = torch.zeros_like(init_image)
                save_mask[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] = 1
                save_mask = save_mask[:,:1,:,:]
                save_mask = save_mask.squeeze(0).squeeze(0)
                save_mask_orig = repeat(torch.tensor(save_mask)[None, None, ...].to(device), '1 1 ... -> b 4 ...', b=batch_size)
                save_mask = save_mask_orig[:, :, ::8, ::8].to(device)

                
                image = Image.fromarray(((save_image/torch.max(save_image.max(), abs(save_image.min())) + 1) * 127.5)[0].permute(1,2,0).to(dtype=torch.uint8).cpu().numpy())
            
                count += 1
                
                precision_scope = autocast if opt.precision == "autocast" else nullcontext
                
                
                # image composition
                with torch.no_grad():
                    with precision_scope("cuda"):
                        for prompts in data:
                            print(prompts)
                            print(prompt1)
                            c, uc, inv_emb = load_model_and_get_prompt_embedding(model, opt, device, prompts, inv=True)
                            c_edit, uc_edit, inv_emb_edit = load_model_and_get_prompt_embedding(model, opt, device, prompt1, inv=True)
                        
                            
                            init_image_ori = save_image_ori
                            init_image = save_image
                        
                            T1 = time.time()
                            ## init_latent
                            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  
                            init_latent_edit = model.get_first_stage_encoding(model.encode_first_stage(init_image_ori))  

                            # ref location in the latent space
                            top_rr = int((0.5*(target_height - height))/target_height * init_latent.shape[2])  
                            bottom_rr = int((0.5*(target_height + height))/target_height * init_latent.shape[2])  
                            left_rr = int((0.5*(target_width - width))/target_width * init_latent.shape[3])  
                            right_rr = int((0.5*(target_width + width))/target_width * init_latent.shape[3]) 
                    
                            new_height = bottom_rr - top_rr
                            new_width = right_rr - left_rr
                            
                            step_height2, remainder = divmod(new_height, 2)
                            step_height1 = step_height2 + remainder
                            step_width2, remainder = divmod(new_width, 2)
                            step_width1 = step_width2 + remainder
                            
                            center_row_rm = int(center_row_from_top * init_latent.shape[2])
                            center_col_rm = int(center_col_from_left * init_latent.shape[3])
                            
                            param = [max(0, int(center_row_rm - step_height1)), 
                                    min(init_latent.shape[2] - 1, int(center_row_rm + step_height2)),
                                    max(0, int(center_col_rm - step_width1)), 
                                    min(init_latent.shape[3] - 1, int(center_col_rm + step_width2))]
                            
                            ref_latent = model.get_first_stage_encoding(model.encode_first_stage(ref_image))
                            
                            shape = [init_latent.shape[1], init_latent.shape[2], init_latent.shape[3]]
                        
                            
                            z_enc_ori, _ = sampler.sample(steps=opt.dpm_steps,
                                                    inv_emb=inv_emb,
                                                    unconditional_conditioning=uc,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    eta=opt.ddim_eta,
                                                    order=opt.dpm_order,
                                                    x_T=init_latent_edit,
                                                    width=width,
                                                    height=height,
                                                    DPMencode=True,
                                                    )  

                            z_enc, _ = sampler.sample(steps=opt.dpm_steps,
                                                    inv_emb=inv_emb,
                                                    unconditional_conditioning=uc,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    eta=opt.ddim_eta,
                                                    order=opt.dpm_order,
                                                    x_T=init_latent,
                                                    width=width,
                                                    height=height,
                                                    DPMencode=True,
                                                    )  
                                

                            z_ref_enc, _ = sampler.sample(steps=opt.dpm_steps,
                                                        inv_emb=inv_emb_edit,
                                                        unconditional_conditioning=uc_edit,
                                                        conditioning=c_edit,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        eta=opt.ddim_eta,
                                                        order=opt.dpm_order,
                                                        x_T=ref_latent,
                                                        DPMencode=True,
                                                        width=width,
                                                        height=height,
                                                        ref=True,
                                                        )
                                            
                            samples_orig2 = z_enc_ori.clone()
                            samples_orig = z_enc.clone()

                            z_enc_in = z_enc.clone()
                            z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                                = z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                                * (  save_mask[:, :, param[0]:param[1], param[2]:param[3]]) \
                                + torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) \
                                * ( 1 - save_mask[:, :, param[0]:param[1], param[2]:param[3]])
                            
                            
                            samples_ref = z_enc_in.clone()
                            samples = z_enc.clone()
                            samples_bg = z_enc_ori.clone()


                            samples[:, :, param[0]:param[1], param[2]:param[3]] = torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) 
                            # apply the segmentation mask on the noise
                            samples[:, :, param[0]:param[1], param[2]:param[3]] \
                                    = samples[:, :, param[0]:param[1], param[2]:param[3]].clone() \
                                    * ( 1 - segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr]) \
                                    + z_ref_enc[:, :, top_rr: bottom_rr, left_rr: right_rr].clone() \
                                    * ( segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr])

                            mask = torch.zeros_like(z_enc, device=device)
                            mask[:, :, param[0]:param[1], param[2]:param[3]] = 1
            
                        
                            samples, _ = sampler.sample(steps=opt.dpm_steps,
                                                        inv_emb=inv_emb,
                                                        inv_emb_edit=inv_emb_edit,
                                                        conditioning_edit=c_edit,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning_edit=uc_edit,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        order=opt.dpm_order,
                                                        x_T=[samples_orig, samples, samples_bg, samples_ref],
                                                        param=param,
                                                        mask=mask,
                                                        tau_a=opt.tau_a,
                                                        tau_b=opt.tau_b,
                                                        )
                                                
            
                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            
                            T2 = time.time()
                            print('Running Time: %s s' % ((T2 - T1)))
                            
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))

                                image_bg = Image.open(opt.init_img)
                                image_fg = Image.open(opt.ref_img)
                                images = [image_bg, image_fg, image, img]

                                total_width = sum(img.width for img in images)
                                max_height = max(img.height for img in images)

                                new_image = Image.new('RGB', (total_width, max_height))

                                x_offset = 0
                                for img in images:
                                    new_image.paste(img, (x_offset, 0))
                                    x_offset += img.width

                                new_image.save(os.path.join(opt.outdir, f"{count:05}-"+prompt1+'.png'))
                        

                torch.cuda.empty_cache()
                gc.collect()

    print("End")


if __name__ == "__main__":
    main()

