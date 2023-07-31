import os, re, time, sys, asyncio, ctypes, math, traceback, warnings, requests
from io import BytesIO
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from typing import Optional
from safetensors.torch import load_file

# Import built libraries
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts
from autoencoder.pixelvae import load_pixelvae_model
from rembg import remove
import hitherdither
import tomesd

from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional
import logging as pylog

# Import websocket tools
import requests
from websockets import serve, connect
from io import BytesIO

# Import console management libraries
from rich import print as rprint

from scripts.lora.lora import assign_lora_names_to_compvis_modules, load_lora, lora_Conv2d_forward, lora_Linear_forward, unload
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pygetwindow as gw
    except:
        rprint(f"[#ab333d]Pygetwindow could not be loaded. This will limit some cosmetic functionality.")
from colorama import just_fix_windows_console

# Fix windows console for color codes
just_fix_windows_console()

# Patch existing console to remove interactivity
kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 128)

log = pylog.getLogger("pytorch_lightning")
log.propagate = False
log.setLevel(pylog.ERROR)
logging.set_verbosity_error()

global model
global modelCS
global modelFS
global modelPV
global running

global timeout
global loaded
loaded = ""

def patch_conv(**patch):
    cls = torch.nn.Conv2d
    init = cls.__init__
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, **patch)
    cls.__init__ = __init__

def patch_conv_asymmetric(model, x, y):
    for layer in flatten(model):
        if type(layer) == torch.nn.Conv2d:
            layer.padding_modeX = 'circular' if x else 'constant'
            layer.padding_modeY = 'circular' if y else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            layer._conv_forward = __replacementConv2DConvForward.__get__(layer, torch.nn.Conv2d)

def restoreConv2DMethods(model):
        for layer in flatten(model):
            if type(layer) == torch.nn.Conv2d:
                layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, torch.nn.Conv2d)

def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

def patch_tiling(tilingX, tilingY, model, modelFS, modelPV):
    # Convert tilingX and tilingY to boolean values
    X = bool(tilingX == "true")
    Y = bool(tilingY == "true")
    #restoreConv2DMethods(model)
    #restoreConv2DMethods(modelFS)
    patch_conv_asymmetric(model, X, Y)
    patch_conv_asymmetric(modelFS, X, Y)
    patch_conv_asymmetric(modelPV.model, X, Y)

    if X or Y:
        rprint("[#494b9b]Patched for tiling in the [#48a971]" + "X" * X + "[#494b9b] and [#48a971]" * (X and Y) + "Y" * Y + "[#494b9b] direction" + "s" * (X and Y))

    return model, modelFS, modelPV

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def searchString(string, *args):
    out = []
    for x in range(len(args)-1):
        out.append(re.search(f"(?<={{{args[x]}}}).*(?={{{args[x+1]}}})", string).group())
    return out

def climage(file, alignment, *args):

    # Get console bounds with a small margin - better safe than sorry
    twidth, theight = os.get_terminal_size().columns-1, (os.get_terminal_size().lines-1)*2

    # Set up variables
    image = Image.open(file)
    image = image.convert('RGBA')
    iwidth, iheight = min(twidth, image.width), min(theight, image.height)
    line = []
    lines = []

    # Alignment stuff

    margin = 0
    if alignment == "centered":
        margin = int((twidth/2)-(iwidth/2))
    elif alignment == "right":
        margin = int(twidth-iwidth)
    elif alignment == "manual":
        margin = args[0]
    
    # Loop over the height of the image / 2 (because 2 pixels = 1 text character)
    for y2 in range(int(iheight/2)):

        # Add default colors to the start of the line
        line = ["[white on black]" + " "*margin]
        rgbp, rgb2p = "", ""

        # Loop over width
        for x in range(iwidth):

            # Get the color for the upper and lower half of the text character
            r, g, b, a = image.getpixel((x, (y2*2)))
            r2, g2, b2, a2 = image.getpixel((x, (y2*2)+1))

            # Convert to hex colors for Rich to use
            rgb, rgb2 = '#{:02x}{:02x}{:02x}'.format(r, g, b), '#{:02x}{:02x}{:02x}'.format(r2, g2, b2)

            # Lookup table because I was bored
            colorCodes = [f"[{rgb2} on {rgb}]", f"[{rgb2} on black]", f"[black on {rgb}]", "[white on black]", f"[{rgb}]"]
            # ~It just works~
            color = colorCodes[int(a < 200)+(int(a2 < 200)*2)+(int(rgb == rgb2 and a + a2 > 400)*4)]

            # Don't change the color if the color doesn't change...
            if rgb == rgbp and rgb2 == rgb2p:
                color = ""
            
            # Set text characters, nothing, full block, half block. Half block + background color = 2 pixels
            if a < 200 and a2 < 200:
                line.append(color + " ")
            elif rgb == rgb2:
                line.append(color + "█")
            else:
                line.append(color + "▄")

            rgbp, rgb2p = rgb, rgb2
        
        # Add default colors to the end of the line
        lines.append("".join(line) + "[white on black]")
    return "\n".join(lines)

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
            iterable = iterable[0:total]
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
            line1 = ["[#494b9b on #3b1725]▄", 
                    "[#c4f129 on #494b9b]▄" * int(int(barwidth * iteration // total) > 0), 
                    "[#ffffff on #494b9b]▄" * max(0, int(barwidth * iteration // total)-2),
                    "[#c4f129 on #494b9b]▄" * int(int(barwidth * iteration // total) > 1),
                    "[#3b1725 on #494b9b]▄" * max(0, barwidth-int(barwidth * iteration // total)),
                    "[#494b9b on #3b1725]▄[white on black]"]
            line2 = ["[#3b1725 on #494b9b]▄", 
                    "[#494b9b on #48a971]▄" * int(int(barwidth * iteration // total) > 0), 
                    "[#494b9b on #c4f129]▄" * max(0, int(barwidth * iteration // total)-2),
                    "[#494b9b on #48a971]▄" * int(int(barwidth * iteration // total) > 1),
                    "[#494b9b on #3b1725]▄" * max(0, barwidth-int(barwidth * iteration // total)),
                    "[#3b1725 on #494b9b]▄[white on black]"]

            percent = ("{0:.0f}").format(100 * (iteration / float(total)))

            # Avoid predicting speed until there's enough data
            if len(delay) >= 1:
                delay.append(time.time()-delay[-1])
                del delay [-2]

            # Fancy color stuff and formating
            if iteration == 0:
                speedColor = "[#48a971 on black]"
                measure = f"... {unit}/s"
                passed = f"00:00"
                remaining = f"??:??"
            else:
                if np.mean(delay) <= 1:
                    measure = f"{round(1/max(0.01, np.mean(delay)), 2)} {unit}/s"
                else:
                    measure = f"{round(np.mean(delay), 2)} s/{unit}"

                if np.mean(delay) <= 1:
                    speedColor = "[#c4f129 on black]"
                elif np.mean(delay) <= 10:
                    speedColor = "[#48a971 on black]"
                elif np.mean(delay) <= 30:
                    speedColor = "[#494b9b on black]"
                else:
                    speedColor = "[#ab333d on black]"

                passed = "{:02d}:{:02d}".format(math.floor(sum(delay)/60), round(sum(delay))%60)
                remaining = "{:02d}:{:02d}".format(math.floor((total*np.mean(delay)-sum(delay))/60), round(total*np.mean(delay)-sum(delay))%60)

            speed = f" {iteration}/{total} at {measure} "
            prediction = f" {passed} < {remaining} "

            # Print single bar across two lines
            rprint(f'\r{f"{name}".center(prefix)} {"".join(line1)}{speedColor}{speed.center(suffix-1)}[white on black]')
            rprint(f'[#48a971 on black]{f"{percent}%".center(prefix)}[white on black] {"".join(line2)}[#494b9b on black]{prediction.center(suffix-1)}', end = printEnd)
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

def load_model_from_config(model, verbose=False):
    # Load the model's state dictionary from the specified file
    try:
        # First try to load as a Safetensor, then as a pickletensor
        try:
            pl_sd = load_file(model, device="cpu")
        except: 
            rprint(f"[#ab333d]Model is not a Safetensor. Please consider using Safetensors format for better security.")
            pl_sd = torch.load(model, map_location="cpu")

        sd = pl_sd

        # If "state_dict" is found in the loaded dictionary, assign it to sd
        if 'state_dict' in sd:
            sd = pl_sd["state_dict"]

        return sd
    except Exception as e: 
                rprint(f"[#ab333d]{traceback.format_exc()}\n\nThis may indicate a model has not been downloaded fully, or is corrupted.")

def load_img(path, h0, w0):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if h0 is not None and w0 is not None:
        h, w = h0, w0
    w, h = map(lambda x: x - x % 8, (w, h))
    image = image.resize((w, h), resample=Image.Resampling.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

def adjust_gamma(image, gamma=1.0):
    # Create a lookup table for the gamma function
    gamma_map = [255 * ((i / 255.0) ** (1.0 / gamma)) for i in range(256)]
    gamma_table = bytes([(int(x / 255.0 * 65535.0) >> 8) for x in gamma_map] * 3)

    # Apply the gamma correction using the lookup table
    return image.point(gamma_table)

def load_model(modelpath, modelfile, config, device, precision, optimized):
    timer = time.time()

    print()
    if modelfile == "model.pxlm":
        print(f"Loading primary model")
    elif modelfile == "modelmini.pxlm":
        print(f"Loading mini model")
    elif modelfile == "modelmega.pxlm":
        print(f"Loading mega model")
    elif modelfile == "paletteGen.pxlm":
        print(f"Loading PaletteGen model")
    else:
        rprint(f"Loading custom model from [#48a971]{modelfile}")

    turbo = True
    if optimized == "true":
        turbo = False
    sd = load_model_from_config(f"{modelpath+modelfile}")
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    global modelPV
    # Ignore an annoying userwaring
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Load the pixelvae
        modelPV = load_pixelvae_model(f"models/decoder/decoder.px", device, "eVWtlIBjTRr0-gyZB0smWSwxCiF8l4PVJcNJOIFLFqE=")

    # Instantiate and load the main model
    global model
    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = 1
    model.cdevice = device
    model.turbo = turbo
    tomesd.apply_patch(model, ratio=0.6, use_rand=True, merge_attn=True, merge_crossattn=True, merge_mlp=True)

    global modelCS
    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = device

    global modelFS
    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()

    if device != "cpu" and precision == "autocast":
        model.half()
        modelCS.half()
        precision = "half"
    
    rprint(f"[#c4f129]Loaded model to [#48a971]{model.cdevice}[#c4f129] at [#48a971]{precision} precision[#c4f129] in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds")

def kCentroid(image, width, height, centroids):
    image = image.convert("RGB")
    downscaled = np.zeros((height, width, 3), dtype=np.uint8)
    wFactor = image.width/width
    hFactor = image.height/height
    for x, y in product(range(width), range(height)):
            tile = image.crop((x*wFactor, y*hFactor, (x*wFactor)+wFactor, (y*hFactor)+hFactor)).quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")
            color_counts = tile.getcolors()
            most_common_color = max(color_counts, key=lambda x: x[0])[1]
            downscaled[y, x, :] = most_common_color
    return Image.fromarray(downscaled, mode='RGB')

def kDenoise(image, smoothing, strength):
    image = image.convert("RGB")
    downscaled = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    for x, y in product(range(image.width), range(image.height)):
            tile = image.crop((x-1, y-1, min(x+2, image.width), min(y+2, image.height)))
            centroids = max(2, min(round((tile.width*tile.height)*(1/strength)), (tile.width*tile.height)))
            tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")
            color_counts = tile.getcolors()
            final_color = tile.getpixel((1, 1))
            count = 0
            for ele in color_counts:
                if (ele[1] == final_color):
                    count = ele[0]
            if count < 1+round(((tile.width*tile.height)*0.8)*(smoothing/10)):
                final_color = max(color_counts, key=lambda x: x[0])[1]
            downscaled[y, x, :] = final_color
    return Image.fromarray(downscaled, mode='RGB')

def palettize(numFiles, colors, paletteFile, paletteURL, dithering, strength, denoise, smoothness, intensity):
    
    # Find the elbow point (best k value)
    if len(rate_of_change) == 0:
        best_k = 2
    else:
        elbow_index = np.argmax(rate_of_change)
        best_k = elbow_index + 1

    return best_k

def determine_best_palette_verbose(image, paletteFolder):
    # Convert the image to RGB mode
    image = image.convert("RGB")

    paletteImages = []
    paletteImages.extend(os.listdir(paletteFolder))

    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different palettes
    distortions = []
    for palImg in clbar(paletteImages, name = "Searching", position = "first", prefixwidth = 12, suffixwidth = 28):
        try:
            palImg = Image.open(f"{paletteFolder}/{palImg}").convert('RGB')
        except:
            continue
        palette = []

        # Extract palette colors
        palColors = palImg.getcolors(16777216)
        numColors = len(palColors)
        palette = np.concatenate([x[1] for x in palColors]).tolist()
        
        # Create a new palette image
        palImg = Image.new('P', (256, 1))
        palImg.putpalette(palette)

        quantized_image = image.quantize(method=1, kmeans=numColors, palette=palImg, dither=0)
        centroids = np.array(quantized_image.getpalette()[:numColors * 3]).reshape(-1, 3)
        
        # Calculate distortions
        distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        distortions.append(np.sum(min_distances ** 2))
    
    # Find the best match
    best_match_index = np.argmin(distortions)
    best_palette = Image.open(f"{paletteFolder}/{paletteImages[best_match_index]}").convert('RGB')

    return best_palette, paletteImages[best_match_index]

def determine_best_k_verbose(image, max_k, accuracy):
    # Convert the image to RGB mode
    image = image.convert("RGB")

    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different values of k
    # Divided into 'chunks' for nice progress displaying
    distortions = []
    count = 0
    for k in clbar(range(4, round(max_k/8) + 2), name = "Finding K", position = "first", prefixwidth = 12, suffixwidth = 28):
        for n in range(round(max_k/k)):
            count += 1
            quantized_image = image.quantize(colors=count, method=2, kmeans=count, dither=0)
            centroids = np.array(quantized_image.getpalette()[:count * 3]).reshape(-1, 3)
            
            # Calculate distortions
            distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
            min_distances = np.min(distances, axis=1)
            distortions.append(np.sum(min_distances ** 2))

    # Calculate the rate of change of distortions
    rate_of_change = np.diff(distortions) / np.array(distortions[:-1])
    
    # Find the elbow point (best k value)
    if len(rate_of_change) == 0:
        best_k = 1
    else:
        elbow_index = np.argmax(rate_of_change)
        best_k = elbow_index + 1

    return best_k

def palettize(numFiles, source, colors, accuracy, bestPaletteFolder, paletteFile, paletteURL, dithering, strength, denoise, smoothness, intensity):
    # Check if a palette URL is provided and try to download the palette image
    if source == "URL":
        try:
            paletteFile = BytesIO(requests.get(paletteURL).content)
            testImg = Image.open(paletteFile).convert('RGB')
        except:
            rprint(f"\n[#ab333d]ERROR: URL {paletteURL} cannot be reached or is not an image\nReverting to Adaptive palette")
            paletteFile = ""

    timer = time.time()
    files = []
    for n in range(numFiles):
        files.append(f"temp/input{n+1}.png")
    if paletteFile != "":
        palImg = Image.open(paletteFile).convert('RGB')
        numColors = len(palImg.getcolors(16777216))
    else:
        numColors = colors
    string = f"\n[#48a971]Converting output[white] to [#48a971]{numColors}[white] colors"
    if strength > 0 and dithering > 0:
        string = f'{string} with order [#48a971]{dithering}[white] dithering'

    if source == "Automatic":
        string = f"\n[#48a971]Converting output[white] to best number of colors"
    elif source == "Best Palette":
        string = f"\n[#48a971]Converting output[white] to best color palette"

    # Print the conversion message
    rprint(string)

    palFiles = []
    # Process each file in the list
    for file in clbar(files, name = "Processed", position = "last", unit = "image", prefixwidth = 12, suffixwidth = 28):
        img = Image.open(file).convert('RGB')
        if denoise == "true":
            img = kDenoise(img, smoothness, intensity)

        threshold = 4*strength
        
        # Check if a palette file is provided
        if (paletteFile != "" and os.path.isfile(file)) or source == "Best Palette":
            # Open the palette image and calculate the number of colors
            if source == "Best Palette":
                palImg, palFile = determine_best_palette_verbose(img, bestPaletteFolder)
                palFiles.append(str(palFile))
            else:
                palImg = Image.open(paletteFile).convert('RGB')
            
            numColors = len(palImg.getcolors(16777216))

            if strength > 0 and dithering > 0:
                for _ in clbar([img], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28):
                    img = adjust_gamma(img, 1.0-(0.02*strength))

                    # Extract palette colors
                    palette = [x[1] for x in palImg.getcolors(16777216)]

                    # Perform ordered dithering using Bayer matrix
                    palette = hitherdither.palette.Palette(palette)
                    img_indexed = hitherdither.ordered.bayer.bayer_dithering(img, palette, [threshold, threshold, threshold], order=dithering).convert('RGB')
            else:
                # Extract palette colors
                palette = np.concatenate([x[1] for x in palImg.getcolors(16777216)]).tolist()
                
                # Create a new palette image
                palImg = Image.new('P', (256, 1))
                palImg.putpalette(palette)
                for _ in clbar([img], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28):
                    img_indexed = img.quantize(method=1, kmeans=numColors, palette=palImg, dither=0).convert('RGB')
        elif colors > 0 and os.path.isfile(file):

            if strength > 0 and dithering > 0:
                for _ in clbar([img], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28):
                    img_indexed = img.quantize(colors=colors, method=1, kmeans=colors, dither=0).convert('RGB')
                    img = adjust_gamma(img, 1.0-(0.03*strength))

                    # Extract palette colors
                    palette = [x[1] for x in img_indexed.getcolors(16777216)]

                    # Perform ordered dithering using Bayer matrix
                    palette = hitherdither.palette.Palette(palette)
                    img_indexed = hitherdither.ordered.bayer.bayer_dithering(img, palette, [threshold, threshold, threshold], order=dithering).convert('RGB')

            else:
                for _ in clbar([img], name = "Palettizing", position = "first", prefixwidth = 12, suffixwidth = 28):
                    img_indexed = img.quantize(colors=colors, method=1, kmeans=colors, dither=0).convert('RGB')
        img_indexed.save(file)
    rprint(f"[#c4f129]Palettized [#48a971]{len(files)}[#c4f129] images in [#48a971]{round(time.time()-timer, 2)}[#c4f129] seconds")
    if source == "Best Palette":
        rprint(f"[#c4f129]Palettes used: [#494b9b]{', '.join(palFiles)}")

def rembg(numFiles):
    
    timer = time.time()
    files = []

    rprint(f"\n[#48a971]Removing [#48a971]{numFiles}[white] backgrounds")
    
    for n in range(numFiles):
        files.append(f"temp/input{n+1}.png")

    for file in clbar(files, name = "Processed", position = "", unit = "image", prefixwidth = 12, suffixwidth = 28):
        img = Image.open(file).convert('RGB')
        
        if os.path.isfile(file):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                remove(img).save(file)
    rprint(f"[#c4f129]Removed [#48a971]{len(files)}[#c4f129] backgrounds in [#48a971]{round(time.time()-timer, 2)}[#c4f129] seconds")

def kCentroidVerbose(width, height, centroids):

    assert os.path.isfile("temp/input.png")
    init_img = Image.open("temp/input.png")

    rprint(f"\n[#48a971]K-Centroid downscaling[white] from [#48a971]{init_img.width}[white]x[#48a971]{init_img.height}[white] to [#48a971]{width}[white]x[#48a971]{height}[white] with [#48a971]{centroids}[white] centroids")

    for _ in clbar(range(1), name = "Processed", unit = "image", prefixwidth = 12, suffixwidth = 28):
        kCentroid(init_img, int(width), int(height), int(centroids)).save("temp/temp.png")
        
def paletteGen(colors, device, precision, prompt, seed):

    base = 2**round(math.log2(colors))

    width = 512+((512/base)*(colors-base))

    # Generate text-to-image conversion with specified parameters
    txt2img("false", device, precision, prompt, "", int(width), 512, 20, 7.0, int(seed), 1, "false", "false", "false")

    image = Image.open("temp/temp.png")

    image = image.convert('RGB')

    image = kCentroid(image, int(image.width/(512/base)), int(image.height/512), 2)

    palette = Image.new('P', (colors, 1))

    for x in range(image.width):
        for y in range(image.height):
            r, g, b = image.getpixel((x, y))

            palette.putpixel((x, y), (r, g, b))

    palette.save("temp/temp.png")

    rprint(f"[#c4f129]Image converted to color palette with [#48a971]{colors}[#c4f129] colors")

def txt2img(pixel, device, precision, prompt, negative, W, H, ddim_steps, scale, seed, n_iter, tilingX, tilingY, pixelvae):
    os.makedirs("temp", exist_ok=True)
    outpath = "temp"

    timer = time.time()
    
    if seed == None:
        seed = randint(0, 1000000)
    seed_everything(seed)

    rprint(f"\n[#48a971]Text to Image[white] generating for [#48a971]{n_iter}[white] iterations with [#48a971]{ddim_steps}[white] steps per iteration at [#48a971]{W}[white]x[#48a971]{H}")

    start_code = None
    sampler = "euler"

    assert prompt is not None
    data = [prompt]
    negative_data = [negative]

    global model
    global modelCS
    global modelFS
    global modelPV

    # Patch tiling for model and modelFS
    model, modelFS, modelPV = patch_tiling(tilingX, tilingY, model, modelFS, modelPV)

    if device != "cpu" and precision == "autocast":
        precision_scope = autocast
    else:
        precision_scope = nullcontext
        
    # Lora
    assign_lora_names_to_compvis_modules(model)
    
    # load lora model
    lora_filename = "pixelbuildings64-v1-000004.safetensors"
    lora_tensors = load_file(lora_filename)
    load_lora("PixelBuildings64", lora_filename, lora_tensors, model) # model should be a standard compvis ldm model loaded with v1-inference.yaml
    
    # apply lora
    torch.nn.Linear.forward = lora_Linear_forward
    torch.nn.Conv2d.forward = lora_Conv2d_forward
    
    # save for restore
    if not hasattr(torch.nn, 'Linear_forward_before_lora'):
        torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

    if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
        torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

    seeds = []
    with torch.no_grad():

        base_count = 1
        for n in clbar(range(n_iter), name = "Iterations", position = "last", unit = "image", prefixwidth = 12, suffixwidth = 28):
            for prompts in data:

                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    uc = modelCS.get_learned_conditioning(negative_data)
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(modelCS.get_learned_conditioning([""]))
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [1, 4, H // 8, W // 8]

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=ddim_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=0.0,
                        x_T=start_code,
                        sampler = sampler,
                    )

                    skip_downscale = False
                    if pixelvae == "true":
                        # Pixel clustering mode, lower threshold means bigger clusters
                        denoise = 0.08
                        x_sample = modelPV.run_cluster(samples_ddim, threshold=denoise, wrap_x=bool(tilingX == "true"), wrap_y=bool(tilingY == "true"))
                        #x_sample = modelPV.run_plain(samples_ddim)
                        # Convert to numpy format, skip downscale later
                        x_sample = x_sample[0].cpu().numpy()
                        skip_downscale = True
                    else:
                        modelFS.to(device)
                        # Decode the samples using the first stage of the model
                        x_sample = [modelFS.decode_first_stage(samples_ddim[i:i+1].to(device))[0].cpu() for i in range(samples_ddim.size(0))]
                        x_sample = torch.stack(x_sample).float()
                        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    
                    x_sample_image = Image.fromarray(x_sample.astype(np.uint8))

                    file_name = "temp"
                    if n_iter > 1:
                        file_name = "temp" + f"{base_count}"
                    if pixel == "true" and not skip_downscale:
                        # Resize the image if pixel is true
                        x_sample_image = kCentroid(x_sample_image, int(W/8), int(H/8), 2)
                    x_sample_image.save(
                        os.path.join(outpath, file_name + ".png")
                    )
                    seeds.append(str(seed))
                    seed += 1
                    base_count += 1

                    # Move modelFS to CPU if necessary to free up GPU memory
                    if device != "cpu" and modelFS.device != torch.device("cpu"):
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
        rprint(f"[#c4f129]Image generation completed in [#48a971]{round(time.time()-timer, 2)} [#c4f129]seconds\n[#48a971]Seeds: [#494b9b]{', '.join(seeds)}")
        
        # unload lora
        unload()

def img2img(pixel, device, precision, prompt, negative, W, H, ddim_steps, scale, strength, seed, n_iter, tilingX, tilingY, pixelvae):
    timer = time.time()
    init_img = "temp/input.png"

    assert os.path.isfile(init_img)
    init_image = load_img(init_img, H, W).to(device)

    os.makedirs("temp", exist_ok=True)
    outpath = "temp"

    if seed == None:
        seed = randint(0, 1000000)
    seed_everything(seed)

    rprint(f"\n[#48a971]Image to Image[white] generating for [#48a971]{n_iter}[white] iterations with [#48a971]{ddim_steps}[white] steps per iteration at [#48a971]{W}[white]x[#48a971]{H}")

    sampler = "ddim"

    assert prompt is not None
    data = [prompt]
    negative_data = [negative]

    global model
    global modelCS
    global modelFS
    global modelPV

    # Patch tiling for model and modelFS
    model, modelFS, modelPV = patch_tiling(tilingX, tilingY, model, modelFS, modelPV)

    modelFS.to(device)

    init_image = repeat(init_image, "1 ... -> b ...", b=1)

    init_latent = modelFS.get_first_stage_encoding(modelFS.encode_first_stage(init_image))  # move to latent space

    init_latent = torch.nn.functional.interpolate(init_latent, size=(H // 8, W // 8), mode="bilinear")

    if device != "cpu":
        mem = torch.cuda.memory_allocated(device=device) / 1e6
        modelFS.to("cpu")
        while torch.cuda.memory_allocated(device=device) / 1e6 >= mem:
            time.sleep(1)

    if device != "cpu" and precision == "autocast":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = []

    assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(strength * ddim_steps)

    with torch.no_grad():

        base_count = 1
        for n in clbar(range(n_iter), name = "Iterations", position = "last", unit = "image", prefixwidth = 12, suffixwidth = 28):
            for prompts in data:

                with precision_scope("cuda"):
                    modelCS.to(device)
                    uc = None
                    uc = modelCS.get_learned_conditioning(negative_data)
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated(device=device) / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated(device=device) / 1e6 >= mem:
                            time.sleep(1)

                    # encode (scaled latent)
                    z_enc = model.stochastic_encode(
                        init_latent,
                        torch.tensor([t_enc]).to(device),
                        seed,
                        0.0,
                        ddim_steps,
                    )
                    # decode it
                    samples_ddim = model.sample(
                        t_enc,
                        c,
                        z_enc,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        sampler = sampler
                    )

                    skip_downscale = False
                    if pixelvae == "true":
                        # Pixel clustering mode, lower threshold means bigger clusters
                        denoise = 0.08
                        x_sample = modelPV.run_cluster(samples_ddim, threshold=denoise, wrap_x=bool(tilingX == "true"), wrap_y=bool(tilingY == "true"))
                        #x_sample = modelPV.run_plain(samples_ddim)
                        # Convert to numpy format, skip downscale later
                        x_sample = x_sample[0].cpu().numpy()
                        skip_downscale = True
                    else:
                        modelFS.to(device)
                        # Decode the samples using the first stage of the model
                        x_sample = [modelFS.decode_first_stage(samples_ddim[i:i+1].to(device))[0].cpu() for i in range(samples_ddim.size(0))]
                        x_sample = torch.stack(x_sample).float()
                        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    
                    # Convert the numpy array to an image
                    x_sample_image = Image.fromarray(x_sample.astype(np.uint8))

                    file_name = "temp"
                    if n_iter > 1:
                        file_name = "temp" + f"{base_count}"
                    if pixel == "true" and not skip_downscale:
                        # Resize the image if pixel is true
                        x_sample_image = kCentroid(x_sample_image, int(W/8), int(H/8), 2)
                    x_sample_image.save(
                        os.path.join(outpath, file_name + ".png")
                    )
                    seeds.append(str(seed))
                    seed += 1
                    base_count += 1

                    # Move modelFS to CPU if necessary to free up GPU memory
                    if device != "cpu" and modelFS.device != torch.device("cpu"):
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated(device=device) / 1e6 >= mem:
                            time.sleep(1)
                    
                    # Delete the samples to free up memory
                    del samples_ddim
        rprint(f"[#c4f129]Image generation completed in [#48a971]{round(time.time()-timer, 2)} seconds\n[#48a971]Seeds: [#494b9b]{', '.join(seeds)}")

async def server(websocket):
    background = False
    async for message in websocket:
        if re.search(r"txt2img.+", message):
            await websocket.send("running txt2img")

            # Extract parameters from the message
            pixel, device, precision, prompt, negative, w, h, ddim_steps, scale, seed, n_iter, tilingX, tilingY, pixelvae = searchString(message, "dpixel", "ddevice", "dprecision", "dprompt", "dnegative", "dwidth", "dheight", "dstep", "dscale", "dseed", "diter", "dtilingx", "dtilingy", "dpixelvae", "end")
            try:
                txt2img(pixel, device, precision, prompt, negative, int(w), int(h), int(ddim_steps), float(scale), int(seed), int(n_iter), tilingX, tilingY, pixelvae)
                await websocket.send("returning txt2img")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                await websocket.send("returning error")

        elif re.search(r"txt2pal.+", message):
            await websocket.send("running txt2pal")
            device, precision, prompt, seed, colors = searchString(message, "ddevice", "dprecision", "dprompt", "dseed", "dcolors", "end")
            try:
                paletteGen(int(colors), device, precision, prompt, int(seed))
                await websocket.send("returning txt2pal")
            except Exception as e:
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                await websocket.send("returning error")

        elif re.search(r"img2img.+", message):
            await websocket.send("running img2img")

            # Extract parameters from the message
            pixel, device, precision, prompt, negative, w, h, ddim_steps, scale, strength, seed, n_iter, tilingX, tilingY, pixelvae = searchString(message, "dpixel", "ddevice", "dprecision", "dprompt", "dnegative", "dwidth", "dheight", "dstep", "dscale", "dstrength", "dseed", "diter", "dtilingx", "dtilingy", "dpixelvae", "end")
            try:
                img2img(pixel, device, precision, prompt, negative, int(w), int(h), int(ddim_steps), float(scale), float(strength)/100, int(seed), int(n_iter), tilingX, tilingY, pixelvae)
                await websocket.send("returning img2img")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                await websocket.send("returning error")

        elif re.search(r"palettize.+", message):
            await websocket.send("running palettize")

            # Extract parameters from the message
            numFiles, source, colors, accuracy, bestPaletteFolder, paletteFile, paletteURL, dithering, strength, denoise, smoothness, intensity = searchString(message, "dnumfiles", "dsource", "dcolors", "daccuracy", "dbestpalettefolder", "dpalettefile", "dpaletteURL", "ddithering", "dstrength", "ddenoise", "dsmoothness", "dintensity", "end")
            try:
                palettize(int(numFiles), source,  int(colors), int(accuracy), bestPaletteFolder, paletteFile, paletteURL, int(dithering), int(strength), denoise, int(smoothness), int(intensity))
                await websocket.send("returning palettize")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                await websocket.send("returning error")

        elif re.search(r"rembg.+", message):
            await websocket.send("running rembg")
            numFiles = searchString(message, "dnumfiles", "end")
            try:
                rembg(int(numFiles[0]))
                await websocket.send("returning rembg")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                await websocket.send("returning error")

        elif re.search(r"kcentroid.+", message):
            await websocket.send("running kcentroid")
            width, height, centroids = searchString(message, "dwidth", "dheight", "dcentroids", "end")
            try:
                kCentroidVerbose(int(width), int(height), int(centroids))
                await websocket.send("returning kcentroid")
            except Exception as e: 
                rprint(f"\n[#ab333d]ERROR:\n{traceback.format_exc()}")
                await websocket.send("returning error")

        elif re.search(r"load.+", message):
            await websocket.send("loading model")
            global loaded
            if loaded != message:
                device, optimized, precision, path, model = searchString(message, "ddevice", "doptimized", "dprecision", "dpath", "dmodel", "end")
                try:
                    load_model(path, model, "scripts/v1-inference.yaml", device, precision, optimized)
                    loaded = message
                except Exception as e: rprint(f"\n[#ab333d]ERROR:\n{e}")
                
            await websocket.send("loaded model")

        elif re.search(r"connected.+", message):
            background = searchString(message, "dbackground", "end")[0]
            try:
                rd = gw.getWindowsWithTitle("Retro Diffusion Image Generator")[0]
                if background == "false":
                    try:
                        # Restore and activate the window
                        rd.restore()
                        rd.activate()
                    except:
                        pass
                else:
                    try:
                        # Minimize the window
                        rd.minimize()
                    except:
                        pass
            except:
                pass
            await websocket.send("connected")
        elif message == "no model":
            await websocket.send("loaded model")
        elif message == "recieved":
            if background == "false":
                try:
                    rd = gw.getWindowsWithTitle("Retro Diffusion Image Generator")[0]
                    if gw.getActiveWindow() is not None:
                        if gw.getActiveWindow().title == "Retro Diffusion Image Generator":
                            # Minimize the window
                            rd.minimize()
                except:
                    pass
            await websocket.send("free")
            torch.cuda.empty_cache()
        elif message == "shutdown":
            rprint("[#ab333d]Shutting down...")
            global running
            global timeout
            running = False
            await websocket.close()
            asyncio.get_event_loop().call_soon_threadsafe(asyncio.get_event_loop().stop)

async def connectSend(uri, message):
    async with connect(uri) as websocket:
        await websocket.send(message)

os.system("title Retro Diffusion Image Generator")

rprint("\n" + climage("logo.png", "centered") + "\n\n")

rprint("[#48a971]Starting Image Generator...")

start_server = serve(server, "localhost", 8765)

rprint("[#c4f129]Connected")

timeout = 1

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
