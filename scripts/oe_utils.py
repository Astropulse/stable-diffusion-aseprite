# Code from https://github.com/KohakuBlueleaf/PixelOE
# Adapted for use with k-centroid algorithm

from functools import partial

import numpy as np
import cv2


def expansion_weight(img, k=16, avg_scale=10, dist_scale=3):
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0] / 255
    avg_y = apply_chunk(img_y, k * 3, k, partial(np.median, axis=1, keepdims=True))
    max_y = apply_chunk(img_y, k, k, partial(np.max, axis=1, keepdims=True))
    min_y = apply_chunk(img_y, k, k, partial(np.min, axis=1, keepdims=True))
    bright_dist = max_y - avg_y
    dark_dist = avg_y - min_y

    weight = (avg_y - 0.5) * avg_scale
    weight = weight - (bright_dist - dark_dist) * dist_scale

    output = sigmoid(weight)
    output = cv2.resize(
        output, (img.shape[1] // k, img.shape[0] // k), interpolation=cv2.INTER_NEAREST
    )
    output = cv2.resize(
        output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    return (output - np.min(output)) / (np.max(output))

# For outline expansion
kernel_expansion = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.uint8)
kernel_smoothing = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]).astype(np.uint8)

def outline_expansion(img, erode=2, dilate=2, k=16, avg_scale=10, dist_scale=3):
    weight = expansion_weight(img, k, avg_scale, dist_scale)[..., np.newaxis]
    orig_weight = sigmoid((weight - 0.5) * 5) * 0.25

    img_erode = img.copy()
    img_erode = cv2.erode(img_erode, kernel_expansion, iterations=erode).astype(
        np.float32
    )
    img_dilate = img.copy()
    img_dilate = cv2.dilate(img_dilate, kernel_expansion, iterations=dilate).astype(
        np.float32
    )

    output = img_erode * weight + img_dilate * (1 - weight)
    output = output * (1 - orig_weight) + img.astype(np.float32) * orig_weight
    output = output.astype(np.uint8).copy()

    output = cv2.erode(output, kernel_smoothing, iterations=erode)
    output = cv2.dilate(output, kernel_smoothing, iterations=dilate * 2)
    output = cv2.erode(output, kernel_smoothing, iterations=erode)

    return output


def match_color(source, target):
    # Convert RGB to L*a*b*, and then match the std/mean
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    result = (source_lab - np.mean(source_lab)) / np.std(source_lab)
    result = result * np.std(target_lab) + np.mean(target_lab)
    source = cv2.cvtColor(
        (result * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )

    source = source.astype(np.float32)
    # Use wavelet colorfix method to match original low frequency data at first
    source[:, :, 0] = wavelet_colorfix(source[:, :, 0], target[:, :, 0])
    source[:, :, 1] = wavelet_colorfix(source[:, :, 1], target[:, :, 1])
    source[:, :, 2] = wavelet_colorfix(source[:, :, 2], target[:, :, 2])
    output = source
    return output.clip(0, 255).astype(np.uint8)


def wavelet_colorfix(input, target):
    input_high, _ = wavelet_decomposition(input, 5)
    _, target_low = wavelet_decomposition(target, 5)
    output = input_high + target_low
    return output


def wavelet_decomposition(input, levels):
    high_freq = np.zeros_like(input)
    for i in range(1, levels + 1):
        radius = 2**i
        low_freq = wavelet_blur(input, radius)
        high_freq = high_freq + (input - low_freq)
        input = low_freq
    return high_freq, low_freq


def wavelet_blur(input, radius):
    kernel_size = 2 * radius + 1
    output = cv2.GaussianBlur(input, (kernel_size, kernel_size), 0)
    return output


def im2col_2d(input_data, kernel_h, kernel_w, pad=0, stride=1):
    type = input_data.dtype
    N, H, W = input_data.shape

    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (pad, pad), (pad, pad)], "constant").astype(type)
    col = np.zeros((N, kernel_h, kernel_w, out_h, out_w)).astype(type)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, y, x, :, :] = img[:, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 3, 4, 1, 2).reshape(N * out_h * out_w, -1)

    return col


def col2im_2d(col, input_shape, kernel_h, kernel_w, pad=0, stride=1):
    type = col.dtype
    C, H, W = input_shape

    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    col = (
        col.reshape(1, out_h, out_w, C, kernel_h, kernel_w)
        .transpose(0, 3, 4, 5, 1, 2)
        .astype(type)
    )
    img = np.zeros((1, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1)).astype(
        type
    )

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[0, :, pad : H + pad, pad : W + pad]


def unfold(data_im, kernel=1, pad=1, stride=1):
    # Call im2col with calculated dimensions
    return im2col_2d(data_im, kernel, kernel, pad, stride)


def fold(data_col, target_shape, kernel=1, pad=1, stride=1):
    # Call col2im with calculated dimensions
    return col2im_2d(
        data_col,
        target_shape,
        kernel,
        kernel,
        pad,
        stride,
    )


def apply_chunk(data, kernel, stride, func):
    org_shape = data.shape
    unfold_shape = org_shape
    k_shift = max(kernel - stride, 0) // 2
    data = np.pad(data, ((k_shift, k_shift), (k_shift, k_shift)), mode="edge")
    if len(org_shape) < 3:
        data = data[np.newaxis, ...]
        unfold_shape = (1, *unfold_shape)
    data = unfold(data, kernel, 0, stride)
    data[..., : stride**2] = func(data)
    data = fold(data[..., : stride**2], unfold_shape, stride, 0, stride)
    if len(org_shape) < 3:
        data = data[0]
    return data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
