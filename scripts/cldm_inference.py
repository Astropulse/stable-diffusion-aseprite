import torch

from ldm.controlnet import load_controlnet as load_controlnet_cldm
from ldm.sample import prepare_noise, sample

from ldm.model_management import unload_all_models
from ldm.lora import load_lora_for_models
from ldm.sd import load_checkpoint_guess_config
from PIL import ImageOps
import numpy as np
import torch

from ldm.hidiffusion import ApplyRAUNet, ApplyMSWMSAAttention
import math


# returns a conditioning with a controlnet applied to it, ready to pass it to a KSampler
def apply_controlnet(conditioning, control_net, image, strength):
    if strength == 0:
        return (conditioning,)

    c = []
    control_hint = image.movedim(-1, 1)
    for t in conditioning:
        n = [t[0], t[1].copy()]
        c_net = control_net.copy().set_cond_hint(control_hint, strength)
        if "control" in t[1]:
            c_net.set_previous_controlnet(t[1]["control"])
        n[1]["control"] = c_net
        n[1]["control_apply_to_uncond"] = True
        c.append(n)
    return (c,)


def load_image(image):
    i = ImageOps.exif_transpose(image)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image, mask.unsqueeze(0))


def load_controlnet(
    controlnets,
    width,
    height,
    model_file,
    device,
    conditioning,
    negative_conditioning,
    loras=[],
    unet_dtype=torch.float16,
):
    # Load base model
    out = load_checkpoint_guess_config(
        model_file,
        output_vae=False,
        output_clip=False,
        output_clipvision=False,
    )

    model_patcher = out[0]

    # Apply loras
    lora_model_patcher = model_patcher

    for lora in loras:
        lora_model_patcher, _clip = load_lora_for_models(
            lora_model_patcher, None, lora["sd"], lora["weight"] / 100, 0
        )

    # Compute conditioning
    cldm_conditioning = [[conditioning[0][0], {"pooled_output": None}]]
    cldm_negative_conditioning = [[negative_conditioning[0][0], {"pooled_output": None}]]

    for controlnet_input in controlnets:
        # Load controlnet model
        controlnet = load_controlnet_cldm(controlnet_input["model_file"])

        # Load conditioning image
        (image, _mask) = load_image(controlnet_input["image"])

        # Apply controlnet to conditioning
        (cldm_conditioning,) = apply_controlnet(cldm_conditioning, controlnet, image, controlnet_input["weight"])
        
    # Patch the model
    lora_model_patcher.patch_model()


    size = round(math.sqrt(width * height) // 8)

    use_hidiff = size >= 80

    # RAUnet settings
    percent_end = (((((size - 80) / 8) * 5) ** 0.33) - 0.2) / 10
    ra_use_blocks = ("3", "8")
    if use_hidiff:
        ra_range = (0.0, percent_end) # (0.0, 0.1) for 80x80 - (0.0, 0.2) for 96x96 - (0.0, 0.3) for 128x128 - (0.0, 0.4) for 192x192 - (0.0, 0.5) for 320x240
    else:
        ra_range = (1.0, 0.0)

    # Cross Attention blocks
    percent_end = ((((size - 124) / 8) / 2.2) ** 0.33) / 10
    ca_use_blocks = ("3", "6") # empty for low, 3 for 320x240
    if size >= 160 and use_hidiff:
        ca_range = (0.0, percent_end) # (1.0, 0.0) for < 112x112 - (0.0, 0.05) for 128x128 - (0.0, 0.15) for 192x192 - (0.0, 0.2) for 320x240
    else:
        ca_range = (1.0, 0.0)

    # mswmsa attention
    attn_use_blocks = ("1,2", "", "11,10,9")
    attn_range = (0.2, 1.0)

    if False: #size > 144:
        lora_model_patcher = ApplyMSWMSAAttention().patch(lora_model_patcher, *attn_use_blocks, "percent", *attn_range)

    lora_model_patcher = ApplyRAUNet().patch(
            True,  # noqa: FBT003
            lora_model_patcher,
            *ra_use_blocks,
            "percent",
            *ra_range,
            False,  # noqa: FBT003
            "bilinear",
            *ca_range,
            *ca_use_blocks,
            "bilinear",
        )

    return lora_model_patcher, cldm_conditioning, cldm_negative_conditioning


def sample_cldm(
    model_patcher,
    conditioning,
    negative_conditioning,
    seed,
    steps = 20,
    cfg = 5.0,
    sampler = "euler",
    batch=1,
    width=512,
    height=512,
    latent=None,
    denoise=1.0, 
    scheduler = "normal",
):
    # Generate empty latents for txt2img
    if latent is None:
        latent = torch.zeros([batch, 4, height // 8, width // 8])

    # Prepare noise
    noise = prepare_noise(latent, seed, None)
    
    for samples_cldm in sample(
        model_patcher,
        noise,
        steps,
        cfg,
        sampler,
        scheduler,
        conditioning,
        negative_conditioning,
        latent,
        denoise=denoise,
        seed=seed,
    ):
        yield samples_cldm / 6.0

def unload_cldm():
    # Unload the model
    unload_all_models()
    
    return
