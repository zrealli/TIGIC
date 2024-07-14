"""SAMPLING ONLY."""
import torch
import ptp_scripts.ptp_scripts as ptp
import sys
sys.path.append('..')
import ptp_scripts.ptp_utils_ori as ptp_utils_ori

from ldm.models.diffusion.dpm_solver.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver

from tqdm import tqdm

MODEL_TYPES = {
    "eps": "noise",
    "v": "v"
}


class DPMSolverSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer('alphas_cumprod', to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.model.device:
                attr = attr.to(self.model.device)
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(self,
               steps,
               batch_size,
               shape,
               conditioning=None,
               conditioning_edit=None,
               inv_emb=None,
               inv_emb_edit=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               unconditional_conditioning_edit=None,
               t_start=None,
               t_end=None,
               DPMencode=False,
               order=2,
               width=None,
               height=None,
               ref=False,
               param=None,
               tau_a=0.5,
               tau_b=0.8,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)


        device = self.model.betas.device
        if x_T is None:
            x = torch.randn(size, device=device)
        else:
            x = x_T

        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
     

        
        if DPMencode:
            # x_T is not a list
            model_fn = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=None, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb,
                unconditional_condition=inv_emb,
                guidance_scale=unconditional_guidance_scale,
            )


            dpm_solver = DPM_Solver(model_fn, ns)
            data, _ = dpm_solver.sample_lower(x, dpm_solver, steps, order, t_start, t_end, device, DPMencode=DPMencode)
            
            for step in range(order, steps + 1):
                data = dpm_solver.sample_one_step(data, step, steps, order=order, DPMencode=DPMencode)   
                     
            return data['x'].to(device), None
        
        else:
            # x_T is a list

            model_fn_decode = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb,
                unconditional_condition=inv_emb,
                guidance_scale=unconditional_guidance_scale,
            )
            model_fn_gen = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=unconditional_guidance_scale,
            )


            model_fn_decode_edit = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=inv_emb_edit,
                unconditional_condition=inv_emb_edit,
                guidance_scale=unconditional_guidance_scale,
            )
            model_fn_gen_edit = model_wrapper(
                lambda x, t, c, DPMencode, controller, inject: self.model.apply_model(x, t, c, encode=DPMencode, controller=controller, inject=inject),
                ns,
                model_type=MODEL_TYPES[self.model.parameterization],
                guidance_type="classifier-free",
                condition=conditioning_edit,
                unconditional_condition=unconditional_conditioning_edit,
                guidance_scale=unconditional_guidance_scale,
            )
            
            orig_controller = ptp.AttentionStore()
            ref_controller = ptp.AttentionStore()
            gen_controller = ptp.AttentionStore()
            Inject_controller = ptp.AttentionStore()
            
            dpm_solver_decode = DPM_Solver(model_fn_decode, ns)
            dpm_solver_gen = DPM_Solver(model_fn_gen, ns)   
            

            dpm_solver_decode_edit = DPM_Solver(model_fn_decode_edit, ns)

            dpm_solver_gen_edit = DPM_Solver(model_fn_gen_edit, ns)
        
            # decoded background
           
            ptp_utils_ori.register_attention_control(self.model, orig_controller)
            

            orig, orig_controller = dpm_solver_decode_edit.sample_lower(x[0], dpm_solver_decode, steps, order, t_start, t_end, device, DPMencode=DPMencode, controller=orig_controller)            
            # decoded reference
            ptp_utils_ori.register_attention_control(self.model, ref_controller)
            ref, ref_controller = dpm_solver_decode_edit.sample_lower(x[3], dpm_solver_decode_edit, steps, order, t_start, t_end, device, DPMencode=DPMencode, controller=ref_controller)
            

            # generation
            Inject_controller = [orig_controller, ref_controller]
            ptp_utils_ori.register_attention_control(self.model, gen_controller, inject_bg=False)
    
            gen, _ = dpm_solver_decode_edit.sample_lower(x[1], dpm_solver_gen_edit, steps, order, t_start, t_end, device, 
                                           DPMencode=DPMencode, controller=Inject_controller, inject=True)

            del orig_controller, ref_controller, gen_controller, Inject_controller
                                    
            orig_controller = ptp.AttentionStore()
            ref_controller = ptp.AttentionStore()
            gen_controller = ptp.AttentionStore()

            for step in range(order, 21):
                # decoded background
                ptp_utils_ori.register_attention_control(self.model, orig_controller)
                orig = dpm_solver_decode.sample_one_step(orig, step, steps, order=order, DPMencode=DPMencode)
                ptp_utils_ori.register_attention_control(self.model, ref_controller)
                ref = dpm_solver_decode_edit.sample_one_step(ref, step, steps, order=order, DPMencode=DPMencode)

                
                if step >= int(0.2*(steps) + 1 - order) and step <= int(0.5*(steps) + 1 - order):
                    inject = True
                    controller = [orig_controller, ref_controller]
                else:
                    inject = False
                    controller = [orig_controller, None]

                if step < int(0.5 * (steps) + 1 - order) and step > int(0.* (steps) + 1 - order) :
                    inject_bg = True
                else:
                    inject_bg = False
                    

                ptp_utils_ori.register_attention_control(self.model, gen_controller, inject_bg=inject_bg)
                gen = dpm_solver_gen_edit.sample_one_step(gen, step, steps, order=order, DPMencode=DPMencode, controller=controller, inject=inject)
                       
                if step < int(1.0*(steps) + 1 - order):
                        blended = orig['x'].clone() 
                        blended[:, :, param[0] : param[1], param[2] : param[3]] \
                            = gen['x'][:, :, param[0] : param[1], param[2] : param[3]].clone()
                        gen['x'] = blended.clone()  
        
                        
            return gen['x'].to(device), None
            
 