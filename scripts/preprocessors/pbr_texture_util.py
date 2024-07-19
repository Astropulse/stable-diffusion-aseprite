import os, time, sys, math

import cv2
import numpy as np
import torch

from PIL import Image

from rich import print as rprint

import preprocessors.pbr_utils.imgops as ops
import preprocessors.pbr_utils.architecture.architecture as arch

# Print progress bar in console
def clbar(iterable, name="", printEnd="\r", position="", unit="it", disable=False, prefixwidth=1, suffixwidth=1, total=0):
    # Console manipulation stuff
    def up(lines=1):
        for _ in range(lines):
            sys.stdout.write("\x1b[1A")
            sys.stdout.flush()

    def down(lines=1):
        for _ in range(lines):
            sys.stdout.write("\n")
            sys.stdout.flush()

    # Allow the complete disabling of the progress bar
    if not disable:
        # Positions the bar correctly
        down(int(position == "last") * 2)
        up(int(position == "first") * 3)

        # Set up variables
        if total > 0:
            # iterable = iterable[0:total]
            pass
        else:
            total = max(1, len(iterable))
        name = f"{name}"
        speed = f" {total}/{total} at 100.00 {unit}/s "
        prediction = f" 00:00 < 00:00 "
        prefix = max(len(name), len("100%"), prefixwidth)
        suffix = max(len(speed), len(prediction), suffixwidth)
        barwidth = os.get_terminal_size().columns - (suffix + prefix + 2)

        # Prints the progress bar
        def printProgressBar(iteration, delay):
            # Define progress bar graphic
            line1 = [
                "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]",
                "[#c4f129 on #494b9b]▄[/#c4f129 on #494b9b]" * int(int(barwidth * min(total, iteration) // total) > 0),
                "[#ffffff on #494b9b]▄[/#ffffff on #494b9b]" * max(0, int(barwidth * min(total, iteration) // total) - 2),
                "[#c4f129 on #494b9b]▄[/#c4f129 on #494b9b]" * int(int(barwidth * min(total, iteration) // total) > 1),
                "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]" * max(0, barwidth - int(barwidth * min(total, iteration) // total)),
                "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]",
            ]
            line2 = [
                "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]",
                "[#494b9b on #48a971]▄[/#494b9b on #48a971]" * int(int(barwidth * min(total, iteration) // total) > 0),
                "[#494b9b on #c4f129]▄[/#494b9b on #c4f129]" * max(0, int(barwidth * min(total, iteration) // total) - 2),
                "[#494b9b on #48a971]▄[/#494b9b on #48a971]" * int(int(barwidth * min(total, iteration) // total) > 1),
                "[#494b9b on #3b1725]▄[/#494b9b on #3b1725]" * max(0, barwidth - int(barwidth * min(total, iteration) // total)),
                "[#3b1725 on #494b9b]▄[/#3b1725 on #494b9b]",
            ]

            percent = ("{0:.0f}").format(100 * (min(total, iteration) / float(total)))

            # Avoid predicting speed until there's enough data
            if len(delay) >= 1:
                delay.append(time.time() - delay[-1])
                del delay[-2]

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

                passed = "{:02d}:{:02d}".format(math.floor(sum(delay) / 60), round(sum(delay)) % 60)
                remaining = "{:02d}:{:02d}".format(math.floor((total * np.mean(delay) - sum(delay)) / 60), round(total * np.mean(delay) - sum(delay)) % 60)

            speed = f" {min(total, iteration)}/{total} at {measure} "
            prediction = f" {passed} < {remaining} "

            # Print single bar across two lines
            rprint(f'\r{f"{name}".center(prefix)} {"".join(line1)}{speedColor}{speed.center(suffix-1)}[white]')
            rprint(f'[#48a971]{f"{percent}%".center(prefix)}[/#48a971] {"".join(line2)}[#494b9b]{prediction.center(suffix-1)}', end=printEnd)
            delay.append(time.time())

            return delay

        # Print at 0 progress
        delay = []
        delay = printProgressBar(0, delay)
        down(int(position == "first") * 2)
        # Update the progress bar
        for i, item in enumerate(iterable):
            yield item
            up(int(position == "first") * 2 + 1)
            delay = printProgressBar(i + 1, delay)
            down(int(position == "first") * 2)

        down(int(position != "first"))
    else:
        for i, item in enumerate(iterable):
            yield item

def process(img, model, device):
    img = img * 1. / np.iinfo(img.dtype).max
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze(
        0).float().cpu().clamp_(0, 1).numpy()
    output = output[[2, 1, 0], :, :]
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.).round()
    return output

def load_model(model_path, device):
    state_dict = torch.load(model_path)
    model = arch.RRDB_Net(3, 3, 32, 12, gc=32, upscale=1, norm_type=None, act_type='leakyrelu',
                            mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)


def generate_pbr(pbr_model_folder, images, textureStyle = "none", device = "cpu", tile_size = 512):

    # Check gpu availability
    if "cuda" in device and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    models = [
        # NORMAL MAP
        load_model(os.path.join(pbr_model_folder, "normal.pth"), device), 
        # ROUGHNESS/DISPLACEMENT MAPS
        load_model(os.path.join(pbr_model_folder, "other.pth"), device)
        ]
    
    normal_maps = []
    roughness_maps = []
    displacement_maps = []
    
    for pil_image in clbar(images, name="Generated", position="", unit="image", prefixwidth=12, suffixwidth=28):
        numpy_image = np.array(pil_image)
        img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        
        # Seamless modes
        if textureStyle == "seamless":
            img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
        elif textureStyle == "mirror":
            img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
        elif textureStyle == "replicate":
            img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)

        img_height, img_width = img.shape[:2]

        # Whether or not to perform the split/merge action
        do_split = img_height > tile_size or img_width > tile_size

        if do_split:
            rlts = ops.esrgan_launcher_split_merge(img, process, models, scale_factor=1, tile_size=tile_size)
        else:
            rlts = [process(img, model, device) for model in models]

        if textureStyle != "none":
            rlts = [ops.crop_seamless(rlt) for rlt in rlts]

        normal = rlts[0]
        roughness = rlts[1][:, :, 1]
        displacement = rlts[1][:, :, 0]

        normal_maps.append(Image.fromarray(np.uint8(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))))
        roughness_maps.append(Image.fromarray(np.uint8(cv2.cvtColor(roughness, cv2.COLOR_BGR2RGB))))
        displacement_maps.append(Image.fromarray(np.uint8(cv2.cvtColor(displacement, cv2.COLOR_BGR2RGB))))
    
    del models

    return normal_maps, roughness_maps, displacement_maps
