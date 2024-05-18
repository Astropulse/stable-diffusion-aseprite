"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

from functools import partial

import sys, os, time, math
from io import StringIO

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_only

from optimization.distributions import DiagonalGaussianDistribution
from ldm.models import VQModelInterface
from ldm.util import (
    default,
    exists,
    extract_into_tensor,
    instantiate_from_config,
    make_beta_schedule,
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    rescale_noise_cfg,
    parse_blocks,
    timestep
)
from samplers import (
    CompVisDenoiser,
    append_dims,
    to_d,
)

from rich import print as rprint
from colorama import just_fix_windows_console
just_fix_windows_console()

def map_range(value, original_range, target_range, power=2):
    in_min, in_max = original_range
    out_min, out_max = target_range

    # Normalize the value to a 0-1 range
    normalized_value = (value - in_min) / (in_max - in_min)
    # Apply the power to the normalized value to weight it
    weighted_value = normalized_value ** power
    # Scale and shift the weighted value to the desired output range
    return weighted_value * (out_max - out_min) + out_min

def clbar(iterable, name = "", printEnd = "\r", position = "", unit = "it", disable = False, prefixwidth = 1, suffixwidth = 1, total = 0):

    # Console manipulation stuff
    def up(lines = 1):
        for _ in range(lines):
            sys.stdout.write('\x1b[1A')
            sys.stdout.flush()

    def down(lines = 1):
        for _ in range(lines):
            sys.stdout.write('\n')
            sys.stdout.flush()

    # Allow the complete disabling of the progress bar
    if not disable:
        # Positions the bar correctly
        down(int(position == "last")*2)
        up(int(position == "first")*3)
        
        # Set up variables
        if total > 0:
            #iterable = iterable[0:total]
            pass
        else:
            total = max(1, len(iterable))
        name = f"{name}"
        speed = f" {total}/{total} at 100.00 {unit}/s "
        prediction = f" 00:00 < 00:00 "
        prefix = max(len(name), len("100%"), prefixwidth)
        suffix = max(len(speed), len(prediction), suffixwidth)
        barwidth = os.get_terminal_size().columns-(suffix+prefix+2)

        # Prints the progress bar
        def printProgressBar (iteration, delay):

            # Define progress bar graphic
            line1 = ["[#494b9b on #3b1725]▄[/#494b9b on #3b1725]", 
                    "[#c4f129 on #494b9b]▄[/#c4f129 on #494b9b]" * int(int(barwidth * iteration // total) > 0), 
                    "[#ffffff on #494b9b]▄[/#ffffff on #494b9b]" * max(0, int(barwidth * iteration // total)-2),
                    "[#c4f129 on #494b9b]▄[/#c4f129 on #494b9b]" * int(int(barwidth * iteration // total) > 1),
                    "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]" * max(0, barwidth-int(barwidth * iteration // total)),
                    "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]"]
            line2 = ["[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]", 
                    "[#494b9b on #48a971]▄[/#494b9b on #48a971]" * int(int(barwidth * iteration // total) > 0), 
                    "[#494b9b on #c4f129]▄[/#494b9b on #c4f129]" * max(0, int(barwidth * iteration // total)-2),
                    "[#494b9b on #48a971]▄[/#494b9b on #48a971]" * int(int(barwidth * iteration // total) > 1),
                    "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]" * max(0, barwidth-int(barwidth * iteration // total)),
                    "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]"]

            percent = ("{0:.0f}").format(100 * (iteration / float(total)))

            # Avoid predicting speed until there's enough data
            if len(delay) >= 1:
                delay.append(time.time()-delay[-1])
                del delay [-2]

            # Fancy color stuff and formating
            if iteration == 0:
                speedColor = "[#48a971]"
                measure = f"... {unit}/s"
                passed = f"00:00"
                remaining = f"??:??"
            else:
                if np.mean(delay) <= 1:
                    measure = f"{round(1/max(0.01, np.mean(delay)), 2)} {unit}/s"
                else:
                    measure = f"{round(np.mean(delay), 2)} s/{unit}"

                if np.mean(delay) <= 1:
                    speedColor = "[#c4f129]"
                elif np.mean(delay) <= 10:
                    speedColor = "[#48a971]"
                elif np.mean(delay) <= 30:
                    speedColor = "[#494b9b]"
                else:
                    speedColor = "[#ab333d]"

                passed = "{:02d}:{:02d}".format(math.floor(sum(delay)/60), round(sum(delay))%60)
                remaining = "{:02d}:{:02d}".format(math.floor((total*np.mean(delay)-sum(delay))/60), round(total*np.mean(delay)-sum(delay))%60)

            speed = f" {iteration}/{total} at {measure} "
            prediction = f" {passed} < {remaining} "

            # Print single bar across two lines
            rprint(f'\r{f"{name}".center(prefix)} {"".join(line1)}{speedColor}{speed.center(suffix-1)}[white]')
            rprint(f'[#48a971]{f"{percent}%".center(prefix)}[/#48a971] {"".join(line2)}[#494b9b]{prediction.center(suffix-1)}', end = printEnd)
            delay.append(time.time())

            return delay

        # Print at 0 progress
        delay = []
        delay = printProgressBar(0, delay)
        down(int(position == "first")*2)
        # Update the progress bar
        for i, item in enumerate(iterable):
            yield item
            up(int(position == "first")*2+1)
            delay = printProgressBar(i + 1, delay)
            down(int(position == "first")*2)
            
        down(int(position != "first"))
    else:
        for i, item in enumerate(iterable):
            yield item

def disabled_train(self):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n, sigma_min=0.1, sigma_max=10, rho=7.0, device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def get_sigmas_ays(n, sigma_min=0.0, sigma_max=14.6146, device='cpu'):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    sigmas = torch.empty((n+1,), device=device)
    for i in range(n+1):
        sigmas[i] = torch.tan((i/n) * alpha_min + (1.0-i/n) * alpha_max)
    return sigmas.to(device)

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )
        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))


class FirstStage(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__()
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        self.instantiate_first_stage(first_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(
                    z, force_not_quantize=predict_cids or force_not_quantize
                )
            else:
                return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(
                    z, force_not_quantize=predict_cids or force_not_quantize
                )
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    x, ks, stride, df=df
                )
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                output_list = [
                    self.first_stage_model.encode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)


class CondStage(DDPM):
    """main class"""

    def __init__(
        self,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__()
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, x, t, cc, transformer_options={}):
        out = self.diffusion_model(x, t, context=cc, transformer_options=transformer_options)
        return out


class DiffusionWrapperOut(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, h, emb, tp, hs, t, cc, transformer_options={}):
        return self.diffusion_model(h, emb, tp, hs, t, context=cc, transformer_options=transformer_options)


class UNet(DDPM):
    """main class"""

    def __init__(
        self,
        unet_encode_config,
        unet_decode_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        unet_bs=1,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0
        self.cdevice = "cuda"
        self.unet_encode_config = unet_encode_config
        self.unet_decode_config = unet_decode_config
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.model1 = DiffusionWrapper(self.unet_encode_config)
        self.model2 = DiffusionWrapperOut(self.unet_decode_config)
        self.model1.eval()
        self.model2.eval()
        self.turbo = False
        self.unet_bs = unet_bs
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if (
            self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            assert (
                self.scale_factor == 1.0
            ), "rather not use custom rescaling and std-rescaling simultaneously"
            # set rescale weight to 1./std of encodings
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.cdevice)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())

    def apply_model(self, x_noisy, t, cond, transformer_options={}, return_ids=False):
        if not self.turbo:
            self.model1.to(self.cdevice)

        step = self.unet_bs
        h, emb, hs = self.model1(x_noisy[0:step], t[:step], cond[:step], transformer_options)
        bs = cond.shape[0]
        # assert bs%2 == 0
        lenhs = len(hs)

        if not self.turbo:
            self.model1.to("cpu")
            self.model2.to(self.cdevice)

        hs_temp = [hs[j][:step] for j in range(lenhs)]
        x_recon = self.model2(h[:step], emb[:step], x_noisy.dtype, hs_temp, t[:step], cond[:step], transformer_options)

        if not self.turbo:
            self.model1.to(self.cdevice)
            self.model2.to("cpu")

        for i in range(step, bs, step):
            h_temp, emb_temp, hs_temp = self.model1(
                x_noisy[i : i + step],
                t[i : i + step], cond[i : i + step],
                transformer_options,
            )
            h = torch.cat((h, h_temp))
            emb = torch.cat((emb, emb_temp))
            for j in range(lenhs):
                hs[j] = torch.cat((hs[j], hs_temp[j]))

        if not self.turbo:
            self.model1.to("cpu")
            self.model2.to(self.cdevice)

        for i in range(step, bs, step):
            hs_temp = [hs[j][i : i + step] for j in range(lenhs)]
            x_recon1 = self.model2(
                h[i : i + step],
                emb[i : i + step],
                x_noisy.dtype,
                hs_temp,
                t[i : i + step],
                cond[i : i + step],
                transformer_options,
            )
            x_recon = torch.cat((x_recon, x_recon1))

        if not self.turbo:
            self.model2.to("cpu")

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def register_buffer1(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(self.cdevice):
                attr = attr.to(dtype=torch.float32).to(torch.device(self.cdevice))
        setattr(self, name, attr)

    def make_schedule(
        self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.num_timesteps,
            verbose=verbose,
        )

        assert (
            self.alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = lambda x: x.to(self.cdevice)
        self.register_buffer1("betas", to_torch(self.betas))
        self.register_buffer1("alphas_cumprod", to_torch(self.alphas_cumprod))
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=self.alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer1("ddim_sigmas", ddim_sigmas)
        self.register_buffer1("ddim_alphas", ddim_alphas)
        self.register_buffer1("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer1("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))

    @torch.no_grad()
    def sample(
        self,
        S,
        conditional_guidance,
        x0=None,
        shape=None,
        seed=1234,
        callback=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        sampler="ddim",
        denoise=1.0,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=False,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_guidance=None,
    ):
        if self.turbo:
            self.model1.to(self.cdevice)
            self.model2.to(self.cdevice)

        if x0 is None:
            batch_size, b1, b2, b3 = shape
            img_shape = (1, b1, b2, b3)
            tens = []
            # print("seeds used = ", [seed+s for s in range(batch_size)])
            for _ in range(batch_size):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=self.cdevice))
                seed += 1
            noise = torch.cat(tens)
            del tens

        x_latent = noise if x0 is None else x0
        # sampling

        transformer_options = {}

        _, _, height, width = x_latent.shape
        size = round(math.sqrt(width * height))

        transformer_options["use_hidiff"] = size >= 80
        transformer_options["use_mswmsa"] = size > 144 # enable for > 144x144

        # RAUnet settings
        percent_end = (((((size - 80) / 8) * 5) ** 0.33) - 0.2) / 10
        ra_use_blocks = parse_blocks("input", "3")
        ra_use_blocks |= parse_blocks("output", "8")
        transformer_options["ra_use_blocks"] = ra_use_blocks
        if transformer_options["use_hidiff"]:
            transformer_options["ra_range"] = (0.0, percent_end) # (0.0, 0.1) for 80x80 - (0.0, 0.2) for 96x96 - (0.0, 0.3) for 128x128 - (0.0, 0.4) for 192x192 - (0.0, 0.5) for 320x240
        else:
            transformer_options["ra_range"] = (1.0, 0.0)

        # Cross Attention blocks
        percent_end = ((((size - 124) / 8) / 2.2) ** 0.33) / 10
        ca_use_blocks = parse_blocks("input", "3") # empty for low, 3 for 320x240
        ca_use_blocks |= parse_blocks("output", "6") # empty for low, 6 for 320x240
        transformer_options["ca_use_blocks"] = ca_use_blocks
        if size >= 160 and transformer_options["use_hidiff"]:
            transformer_options["ca_range"] = (0.0, percent_end) # (1.0, 0.0) for < 112x112 - (0.0, 0.05) for 128x128 - (0.0, 0.15) for 192x192 - (0.0, 0.2) for 320x240
        else:
            transformer_options["ca_range"] = (1.0, 0.0)

        # mswmsa attention
        attn_use_blocks = parse_blocks("input", "1, 2")
        attn_use_blocks |= parse_blocks("middle", "")
        attn_use_blocks |= parse_blocks("output", "9,10,11")
        transformer_options["attn_use_blocks"] = attn_use_blocks
        transformer_options["attn_range"] = (0.0, 1.0)

        if sampler == "ddim":
            for samples in self.ddim_sampling(
                x_latent,
                conditional_guidance,
                S,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_guidance=unconditional_guidance,
                mask=mask,
                init_latent=x_T,
                use_original_steps=False,
                transformer_options=transformer_options
            ):
                yield samples

        elif sampler == "pxlcm":
            self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
            for samples in self.pxlcm_sampling(
                self.alphas_cumprod,
                x_latent,
                S,
                conditional_guidance,
                unconditional_guidance=unconditional_guidance,
                unconditional_guidance_scale=unconditional_guidance_scale,
                denoise=denoise,
                transformer_options=transformer_options,
            ):
                yield samples

        if self.turbo:
            self.model1.to("cpu")
            self.model2.to("cpu")

        yield samples
        

    @torch.no_grad()
    def stochastic_encode(
        self, x0, t, seed, ddim_eta, ddim_steps, use_original_steps=False, noise=None
    ):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        self.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)

        if noise is None:
            b0, b1, b2, b3 = x0.shape
            img_shape = (1, b1, b2, b3)
            tens = []
            # print("seeds used = ", [seed+s for s in range(b0)])
            for _ in range(b0):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=x0.device))
                seed += 1
            noise = torch.cat(tens)
            del tens
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape) * noise
        )

    @torch.no_grad()
    def add_noise(self, x0, t):
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        noise = torch.randn(x0.shape, device=x0.device)

        # print(extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape),
        #       extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape))
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape) * noise
        )
    
    @torch.no_grad()
    def ddim_sampling(
        self,
        x_latent,
        conditional_guidance,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_guidance=None,
        mask=None,
        init_latent=None,
        use_original_steps=False,
        transformer_options={},
    ):
        timesteps = self.ddim_timesteps
        timesteps = timesteps[:t_start]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = clbar(time_range, name = "Samples", position = "first", prefixwidth = 12, suffixwidth = 28, total = total_steps)

        x_dec = x_latent
        x0 = init_latent
        for i, step in enumerate(iterator):
            # Reverse conditioning for image to image
            index = total_steps - i - 1
            ts = torch.full(
                (x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long
            )

            if mask is not None:
                # x0_noisy = self.add_noise(mask, torch.tensor([index] * x0.shape[0]).to(self.cdevice))
                x0_noisy = x0
                x_dec = x0_noisy * mask + (1.0 - mask) * x_dec

            if len(conditional_guidance) == len(time_range-1):
                text_embed = conditional_guidance[index]
                neg_text_embed = unconditional_guidance[index]
            else:

                # Get conditioning from timestep
                condStep = map_range(index, (0, len(time_range)-1), (0, len(conditional_guidance)-1), 0.5)
                condStepInt = round(condStep)

                text_embed = conditional_guidance[condStepInt]
                neg_text_embed = unconditional_guidance[condStepInt]

                # Fraction off of whole number
                fraction_off = condStep - math.floor(condStep)

                # Lerp conditioning smoothly
                if fraction_off > 0.5 and fraction_off > 0:
                    lerp_text_embed = text_embed
                    text_embed = conditional_guidance[condStepInt-1]
                    text_embed = text_embed * (1-fraction_off) + lerp_text_embed * fraction_off
                elif condStepInt < len(conditional_guidance)-1 and fraction_off > 0:
                    lerp_text_embed = conditional_guidance[condStepInt+1]
                    text_embed = text_embed * (1-fraction_off) + lerp_text_embed * fraction_off

            x_dec = self.p_sample_ddim(
                x_dec,
                text_embed,
                ts,
                index=index,
                total=len(time_range),
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_guidance=neg_text_embed,
                transformer_options=transformer_options,
            )
            yield x_dec

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        conditional_guidance,
        t,
        index,
        total,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_guidance=None,
        transformer_options={},
    ):
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        transformer_options["sigmas"] = sigmas
        transformer_options["sigma"] = sigmas[index]
        b, *_, device = *x.shape, x.device

        transformer_options["percent"] = abs(1.0 - (index/(total-1)))
        if not transformer_options["use_hidiff"] and len(conditional_guidance) == 1 and (unconditional_guidance is None or unconditional_guidance_scale == 1.0 or (index % min(5, max(1, round(total/10))) > 0 and (index >= min(20, max(5, total/3))))):
            denoised = self.apply_model(x, t, conditional_guidance, transformer_options)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_guidance, conditional_guidance])
            e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in, transformer_options).chunk(2)
            denoised = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # Rescale CFG
            denoised = rescale_noise_cfg(denoised, e_t)

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * denoised) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * denoised
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    @torch.no_grad()
    def pxlcm_sampling(
        self,
        ac,
        x,
        S,
        conditional_guidance,
        unconditional_guidance=None,
        unconditional_guidance_scale=1,
        denoise=1.0,
        extra_args=None,
        callback=None,
        disable=None,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        transformer_options={},
    ):
        extra_args = {} if extra_args is None else extra_args
        cvd = CompVisDenoiser(ac)
        add_steps = round(8 / denoise) - 8
        sigmas = get_sigmas_ays(S + add_steps, device=x.device)
        sigmas = sigmas[-(S + 1):]
        transformer_options["sigmas"] = sigmas
        x = x * torch.sqrt(1 + sigmas[0] ** 2.0)

        # MacOS, what the fuck is wrong with you? Why do I need to print this tensor for it's value to be accurate? Fuck you. What the fuck.
        # I'm sending it to nothing because I SHOULD NOT NEED TO SEE THE TENSOR TO HAVE IT'S NUMBERS BE RIGHT.
        original_pipe = sys.stdout
        sys.stdout = StringIO()
        print(x)
        sys.stdout = original_pipe

        s_in = x.new_ones([x.shape[0]]).half()
        for index in clbar(range(len(sigmas) - 1), name = "Samples", position = "first", prefixwidth = 12, suffixwidth = 28):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[index] <= s_tmax else 0.
            transformer_options["sigma"] = sigmas[index]
            sigma_hat = sigmas[index] * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                x = x + eps * (sigma_hat ** 2 - sigmas[index] ** 2) ** 0.5

            s_i = sigma_hat * s_in

            if len(conditional_guidance) == len(sigmas)-1:
                text_embed = conditional_guidance[index]
                neg_text_embed = unconditional_guidance[index]
            else:

                # Get conditioning from timestep
                condStep = map_range(index, (0, len(sigmas)-1), (0, len(conditional_guidance)-1), 0.5)
                condStepInt = round(condStep)

                text_embed = conditional_guidance[condStepInt]
                neg_text_embed = unconditional_guidance[condStepInt]

                # Fraction off of whole number
                fraction_off = condStep - math.floor(condStep)

                # Lerp conditioning smoothly
                if fraction_off > 0.5 and fraction_off > 0:
                    lerp_text_embed = text_embed
                    text_embed = conditional_guidance[condStepInt-1]
                    text_embed = text_embed * (1-fraction_off) + lerp_text_embed * fraction_off
                elif condStepInt < len(conditional_guidance)-1 and fraction_off > 0:
                    lerp_text_embed = conditional_guidance[condStepInt+1]
                    text_embed = text_embed * (1-fraction_off) + lerp_text_embed * fraction_off

            transformer_options["percent"] = index/(len(sigmas)-2)
            if not transformer_options["use_hidiff"] and len(conditional_guidance) == 1 and (neg_text_embed is None or unconditional_guidance_scale == 1.0 or (index % min(5, max(1, round((len(sigmas)-1)/10))) > 0 and (index >= min(20, max(8, (len(sigmas)-1)/3))))):
                c_out, c_in = [append_dims(tmp, x.ndim) for tmp in cvd.get_scalings(s_i)]
                eps = self.apply_model(x * c_in, cvd.sigma_to_t(s_i), text_embed, transformer_options)
                denoised = x + eps * c_out
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([s_i] * 2)
                cond_in = torch.cat([neg_text_embed, text_embed])
                c_out, c_in = [append_dims(tmp, x_in.ndim) for tmp in cvd.get_scalings(t_in)]
                eps = self.apply_model(x_in * c_in, cvd.sigma_to_t(t_in), cond_in, transformer_options)
                e_t_uncond, e_t = (x_in + eps * c_out).chunk(2)
                denoised = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

                # Rescale CFG
                denoised = rescale_noise_cfg(denoised, e_t)

            d = to_d(x, sigma_hat, denoised)
            dt = sigmas[index + 1] - sigma_hat
            x = x + d * dt
            yield x