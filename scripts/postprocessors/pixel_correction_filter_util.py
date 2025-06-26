import math
import numpy as np
from PIL import Image
from collections import Counter
from scipy.signal import convolve2d

# ---------------- Configuration Variables ----------------

TOLERANCE = 5.0                       # initial color tolerance
PATTERN_CORRECTION = True             # if True, apply pattern recognition to exclude shapes
NUM_ITERATIONS = 4                    # maximum number of iterations if stopping conditions are not met

# New stopping condition parameters (in percentages)
MIN_BAD_PERCENTAGE = 0.2              # stop if bad pixel percentage is below this percent
DELTA_BAD_PERCENTAGE = 0.1            # stop if change from previous iteration is less than this percent
MIN_FINAL_BAD_PERCENTAGE = 1.0        # if initial bad pixel percentage is below this, return the unedited image

# Parameters for pattern heatmap analysis
PATTERN_HEAT_MIN = 3.0                # brightness threshold: if any pattern's heatmap value is >= this, remove that bad pixel
PATTERN_HEAT_MAX = 5.0



ALLOWED_SHAPES = [
    {'mask': np.array([[False, False, False, False, False],
                         [False,  True, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[-1, -1],
                          [ 0,  0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False,  True, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [1, -1]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False,  True, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [1, 1]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False,  True, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[-1, 1],
                          [ 0, 0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[-1, 0],
                          [ 0, 0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False,  True,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[ 0, -1],
                          [ 0,  0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [1, 0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True,  True, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [0, 1]], dtype=np.int64)},
    {'mask': np.array([[False,  True, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[-2, -1],
                          [ 0,  0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [ True, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [1, -2]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False,  True, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [2, 1]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False,  True],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[-1, 2],
                          [ 0, 0]], dtype=np.int64)},
    {'mask': np.array([[False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[-2, 0],
                          [ 0, 0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [ True, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, -2],
                          [0,  0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [2, 0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False,  True],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [0, 2]], dtype=np.int64)},
    {'mask': np.array([[False, False, False,  True, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[-2, 1],
                          [ 0, 0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [ True, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[-1, -2],
                          [ 0,  0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False, False],
                         [False,  True, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [2, -1]], dtype=np.int64)},
    {'mask': np.array([[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False,  True, False, False],
                         [False, False, False, False,  True],
                         [False, False, False, False, False]]),
     'center': (2, 2),
     'offsets': np.array([[0, 0],
                          [1, 2]], dtype=np.int64)}
]

CORRECTION_SHAPES = [
    {'mask': np.array([[False, True, False],
                         [False, True, False],
                         [False, True, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 0],
                          [ 0, 0],
                          [ 1, 0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False],
                         [True, True, True],
                         [False, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[ 0, -1],
                          [ 0,  0],
                          [ 0,  1]], dtype=np.int64)},
    {'mask': np.array([[False, True, True],
                         [True, True, False],
                         [False, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 0],
                          [-1, 1],
                          [ 0, -1],
                          [ 0,  0]], dtype=np.int64)},
    {'mask': np.array([[True, False, False],
                         [True, True, False],
                         [False, True, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, -1],
                          [ 0, -1],
                          [ 0,  0],
                          [ 1,  0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False],
                         [False, True, True],
                         [True,  True, False]]),
     'center': (1, 1),
     'offsets': np.array([[0, 0],
                          [0, 1],
                          [1, -1],
                          [1,  0]], dtype=np.int64)},
    {'mask': np.array([[False, True, False],
                         [False, True, True],
                         [False, False, True]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 0],
                          [ 0, 0],
                          [ 0, 1],
                          [ 1, 1]], dtype=np.int64)},
    {'mask': np.array([[True, True, False],
                         [False, True, True],
                         [False, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, -1],
                          [-1,  0],
                          [ 0,  0],
                          [ 0,  1]], dtype=np.int64)},
    {'mask': np.array([[False, True, False],
                         [True,  True, False],
                         [True,  False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 0],
                          [ 0, -1],
                          [ 0,  0],
                          [ 1, -1]], dtype=np.int64)},
    {'mask': np.array([[False, False, False],
                         [True, True, False],
                         [False, True, True]]),
     'center': (1, 1),
     'offsets': np.array([[ 0, -1],
                          [ 0,  0],
                          [ 1,  0],
                          [ 1,  1]], dtype=np.int64)},
    {'mask': np.array([[False, False, True],
                         [False, True, True],
                         [False, True, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 1],
                          [ 0, 0],
                          [ 0, 1],
                          [ 1, 0]], dtype=np.int64)},
    {'mask': np.array([[True, False, False],
                         [False, True, False],
                         [False, True, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, -1],
                          [ 0,  0],
                          [ 1,  0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False],
                         [False, True, True],
                         [True, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[0, 0],
                          [0, 1],
                          [1, -1]], dtype=np.int64)},
    {'mask': np.array([[False, True, False],
                         [False, True, False],
                         [False, False, True]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 0],
                          [ 0, 0],
                          [ 1, 1]], dtype=np.int64)},
    {'mask': np.array([[False, False, True],
                         [True, True, False],
                         [False, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 1],
                          [ 0, -1],
                          [ 0,  0]], dtype=np.int64)},
    {'mask': np.array([[True, False, False],
                         [False, True, False],
                         [False, True, True]]),
     'center': (1, 1),
     'offsets': np.array([[-1, -1],
                          [ 0,  0],
                          [ 1,  0],
                          [ 1,  1]], dtype=np.int64)},
    {'mask': np.array([[False, False, True],
                         [False, True, True],
                         [True, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 1],
                          [ 0,  0],
                          [ 0,  1],
                          [ 1, -1]], dtype=np.int64)},
    {'mask': np.array([[True, True, False],
                         [False, True, False],
                         [False, False, True]]),
     'center': (1, 1),
     'offsets': np.array([[-1, -1],
                          [-1,  0],
                          [ 0,  0],
                          [ 1,  1]], dtype=np.int64)},
    {'mask': np.array([[False, False, True],
                         [True, True, False],
                         [True, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 1],
                          [ 0, -1],
                          [ 0,  0],
                          [ 1, -1]], dtype=np.int64)},
    {'mask': np.array([[False, False, True],
                         [False, True, False],
                         [False, True, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 1],
                          [ 0,  0],
                          [ 1,  0]], dtype=np.int64)},
    {'mask': np.array([[True, False, False],
                         [False, True, True],
                         [False, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, -1],
                          [ 0,  0],
                          [ 0,  1]], dtype=np.int64)},
    {'mask': np.array([[False, True, False],
                         [False, True, False],
                         [ True, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 0],
                          [ 0, 0],
                          [ 1, -1]], dtype=np.int64)},
    {'mask': np.array([[False, False, False],
                         [True, True, False],
                         [False, False, True]]),
     'center': (1, 1),
     'offsets': np.array([[ 0, -1],
                          [ 0,  0],
                          [ 0,  1],
                          [ 1,  1]], dtype=np.int64)},
    {'mask': np.array([[True, False, False],
                         [False, True, False],
                         [False, False, True]]),
     'center': (1, 1),
     'offsets': np.array([[-1, -1],
                          [ 0,  0],
                          [ 1,  1]], dtype=np.int64)},
    {'mask': np.array([[False, False, True],
                         [False, True, False],
                         [True, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 1],
                          [ 0,  0],
                          [ 1, -1]], dtype=np.int64)},
    {'mask': np.array([[False, True, False],
                         [True, True, False],
                         [False, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 0],
                          [ 0, -1],
                          [ 0,  0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False],
                         [True, True, False],
                         [False, True, False]]),
     'center': (1, 1),
     'offsets': np.array([[0, -1],
                          [0,  0],
                          [1,  0]], dtype=np.int64)},
    {'mask': np.array([[False, False, False],
                         [False, True, True],
                         [False, True, False]]),
     'center': (1, 1),
     'offsets': np.array([[0, 0],
                          [0, 1],
                          [1, 0]], dtype=np.int64)},
    {'mask': np.array([[False, True, False],
                         [False, True, True],
                         [False, False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, 0],
                          [ 0, 0],
                          [ 0, 1]], dtype=np.int64)}
]

PATTERN_SHAPES = [
    {'mask': np.array([[True, False],
                         [False, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1, -1]], dtype=np.int64)},
    {'mask': np.array([[True, False],
                         [False, True]]),
     'center': (1, 1),
     'offsets': np.array([[-1, -1],
                          [ 0,  0]], dtype=np.int64)},
    {'mask': np.array([[False, True],
                         [True, False]]),
     'center': (1, 1),
     'offsets': np.array([[-1,  0],
                          [ 0, -1]], dtype=np.int64)}
]



# ---------------- Helper Functions ----------------

# ---------------- Helper Functions ----------------

def gaussian_kernel(size, sigma):
    """Creates a normalized 2D Gaussian kernel with maximum value 1."""
    ax = np.linspace(-(size-1)/2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.max(kernel)
    return kernel.astype(np.float32)

def add_kernel_to_heatmap(heatmap, center_y, center_x, kernel):
    """Adds the given kernel to the heatmap centered at (center_y, center_x), handling boundaries."""
    ksize = kernel.shape[0]
    half = ksize // 2
    h, w = heatmap.shape
    y1 = max(center_y - half, 0)
    y2 = min(center_y - half + ksize, h)
    x1 = max(center_x - half, 0)
    x2 = min(center_x - half + ksize, w)
    ky1 = 0 if center_y - half >= 0 else half - center_y
    ky2 = ksize - (center_y - half + ksize - h) if (center_y - half + ksize) > h else ksize
    kx1 = 0 if center_x - half >= 0 else half - center_x
    kx2 = ksize - (center_x - half + ksize - w) if (center_x - half + ksize) > w else ksize
    heatmap[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]

def safe_filename(s):
    """Converts a string into a filename-safe string."""
    return "".join(c if c.isalnum() else "_" for c in s)

def find_best_candidate_shape(image_array, x, y, tol, candidate_shapes, bad_color):
    """
    For each candidate shape, uses precomputed offsets to extract candidate colors
    from image_array and computes Euclidean distances to bad_color.
    Returns the candidate color (the most common one among those within tolerance)
    if one is found, otherwise returns None.
    """
    img_h, img_w, _ = image_array.shape
    best_candidate_color = None
    best_count = -1
    best_avg_distance = float('inf')
    for shape in candidate_shapes:
        offsets = shape.get('offsets')
        if offsets is None or len(offsets) == 0:
            continue
        candidate_positions = offsets + np.array([y, x])
        valid = (candidate_positions[:, 0] >= 0) & (candidate_positions[:, 0] < img_h) & \
                (candidate_positions[:, 1] >= 0) & (candidate_positions[:, 1] < img_w)
        candidate_positions = candidate_positions[valid]
        if candidate_positions.shape[0] == 0:
            continue
        colors = [tuple(image_array[pos[0], pos[1]]) for pos in candidate_positions]
        if not colors:
            continue
        distances = [math.sqrt(sum((int(a) - int(b)) ** 2 for a, b in zip(color, bad_color)))
                     for color in colors]
        count_close = sum(1 for d in distances if d <= tol)
        avg_distance = sum(distances) / len(distances) if distances else float('inf')
        if count_close > best_count or (count_close == best_count and avg_distance < best_avg_distance):
            best_count = count_close
            best_avg_distance = avg_distance
            best_candidate_color = Counter(colors).most_common(1)[0][0]
    return best_candidate_color

# ---------------- Global Filter Functions ----------------

def vectorized_allowed_mask(image_array, allowed_shapes, tol, alpha_mask=None, alpha_threshold=100):
    """
    Computes a boolean mask for the image where each pixel is marked as "good"
    if it matches any of the allowed shapes. If an alpha_mask is provided, then for each
    neighbor pixel used in the shape matching, if its alpha value is below alpha_threshold,
    the condition is automatically considered True.
    """
    h, w, _ = image_array.shape
    overall_good_mask = np.zeros((h, w), dtype=bool)
    tol_sq = tol * tol
    for shape in allowed_shapes:
        mask = shape['mask']
        cy, cx = shape['center']
        coords = np.argwhere(mask)
        offsets = coords - np.array([cy, cx])
        shape_good = np.ones((h, w), dtype=bool)
        for dy, dx in offsets:
            if dy >= 0:
                y_region = slice(0, h - dy)
                y_neighbor = slice(dy, h)
            else:
                y_region = slice(-dy, h)
                y_neighbor = slice(0, h + dy)
            if dx >= 0:
                x_region = slice(0, w - dx)
                x_neighbor = slice(dx, w)
            else:
                x_region = slice(-dx, w)
                x_neighbor = slice(0, w + dx)
            base = image_array[y_region, x_region].astype(np.int32)
            neighbor = image_array[y_neighbor, x_neighbor].astype(np.int32)
            diff_sq = np.sum((neighbor - base) ** 2, axis=-1)
            cond = diff_sq <= tol_sq
            if alpha_mask is not None:
                # Get the alpha values for the neighbor region.
                neighbor_alpha = alpha_mask[y_neighbor, x_neighbor]
                # If the neighbor pixel is transparent, force cond True.
                cond = cond | (neighbor_alpha < alpha_threshold)
            shape_good[y_region, x_region] &= cond
        overall_good_mask |= shape_good
    return overall_good_mask

def remove_intentional_features_with_patterns(bad_mask, pattern_shapes, blur_kernel):
    """
    Uses user-defined pattern shapes to detect intentional features.
    For each pattern shape, it uses a 2D convolution (via SciPy) to slide the pattern over bad_mask.
    When a full match is found, a blurred stamp (using blur_kernel) is added to a heatmap.
    Then, for each bad pixel, if any pattern heatmap value is between PATTERN_HEAT_MIN and PATTERN_HEAT_MAX,
    that pixel is removed from the bad_mask.
    Returns the updated bad_mask and a dictionary mapping pattern keys to their heatmaps.
    """
    h, w = bad_mask.shape
    pattern_heatmaps = {}
    new_bad_mask = bad_mask.copy()
    for pattern in pattern_shapes:
        p_mask = pattern['mask'].astype(np.uint8)
        p_h, p_w = p_mask.shape
        if np.all(p_mask == 0):
            continue
        pattern_key = str(p_mask.flatten().tolist())
        conv_result = convolve2d(bad_mask.astype(np.uint8), p_mask, mode='valid')
        full_match = (conv_result == np.sum(p_mask))
        heatmap = np.zeros((h, w), dtype=np.float32)
        match_indices = np.argwhere(full_match)
        for y, x in match_indices:
            center_y = y + p_h // 2
            center_x = x + p_w // 2
            add_kernel_to_heatmap(heatmap, center_y, center_x, blur_kernel)
        pattern_heatmaps[pattern_key] = heatmap
        condition = (heatmap >= PATTERN_HEAT_MIN) & (heatmap <= PATTERN_HEAT_MAX)
        new_bad_mask[condition] = False
    return new_bad_mask, pattern_heatmaps

# ---------------- Processing Functions ----------------

def process_iteration(image_array, allowed_shapes, candidate_shapes, pattern_shapes, alpha_mask=None):
    """
    Processes one filter iteration.
    After computing the allowed mask via shape matching (using the alpha_mask if available),
    the rest of the procedure is as before.
    """
    img_h, img_w, _ = image_array.shape
    total_pixels = img_h * img_w

    allowed_mask = vectorized_allowed_mask(image_array, allowed_shapes, TOLERANCE, alpha_mask=alpha_mask)
    bad_mask = ~allowed_mask
    cleaned_array = image_array.copy()
    changed_debug = np.zeros_like(image_array)
    bad_count = np.count_nonzero(bad_mask)
    initial_bad_mask = bad_mask.copy()

    # Compute dynamic blur size.
    dynamic_blur_size = int(np.sqrt(img_h * img_w) * 0.2)
    if dynamic_blur_size < 3:
        dynamic_blur_size = 3
    if dynamic_blur_size % 2 == 0:
        dynamic_blur_size += 1
    dynamic_blur_kernel = gaussian_kernel(dynamic_blur_size, dynamic_blur_size / 6)

    # Remove intentional features if enabled.
    if PATTERN_CORRECTION:
        bad_mask, pattern_heatmaps = remove_intentional_features_with_patterns(bad_mask, pattern_shapes, dynamic_blur_kernel)
    else:
        pattern_heatmaps = {}

    new_bad_count = np.count_nonzero(bad_mask)

    # Create debug image: blue for pixels that were initially bad but removed, red for remaining bad pixels.
    debug_bad = image_array.copy()
    blue_mask = (initial_bad_mask) & (~bad_mask)
    debug_bad[blue_mask] = [0, 0, 255]
    debug_bad[bad_mask] = [255, 0, 0]

    # ---- Candidate Correction for Bad Pixels ----
    bad_coords = np.argwhere(bad_mask)
    for y, x in bad_coords:
        bad_color = tuple(cleaned_array[y, x])
        current_tol = TOLERANCE
        candidate = None
        while candidate is None and current_tol <= 150:
            candidate = find_best_candidate_shape(cleaned_array, x, y, current_tol, candidate_shapes, bad_color)
            if candidate is None:
                current_tol += 5
        if candidate is None or candidate == bad_color:
            plus_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            neighbor_colors = []
            for dy, dx in plus_offsets:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= img_h or nx < 0 or nx >= img_w:
                    continue
                neighbor_colors.append(tuple(cleaned_array[ny, nx]))
            if neighbor_colors:
                candidate = Counter(neighbor_colors).most_common(1)[0][0]
        if candidate is not None:
            cleaned_array[y, x] = candidate
            changed_debug[y, x] = candidate

    return cleaned_array, debug_bad, changed_debug, pattern_heatmaps, bad_mask

def process_image_file(image, allowed_shapes, candidate_shapes, pattern_shapes):
    # For RGBA images, extract the raw RGB channels and the alpha channel without compositing.
    if image.mode == "RGBA":
        rgba_array = np.array(image)
        image_array = rgba_array[:, :, :3]
        alpha_channel = rgba_array[:, :, 3]
    else:
        alpha_channel = None
        image_array = np.array(image.convert("RGB"))

    total_pixels = image_array.shape[0] * image_array.shape[1]
    initial_allowed_mask = vectorized_allowed_mask(image_array, allowed_shapes, TOLERANCE, alpha_mask=alpha_channel)
    initial_bad_mask = ~initial_allowed_mask
    initial_bad_count = np.count_nonzero(initial_bad_mask)
    initial_bad_percent = (initial_bad_count / total_pixels) * 100
    if initial_bad_percent < MIN_FINAL_BAD_PERCENTAGE:
        final_cleaned = image_array
        final_bad = image_array
    else:
        iteration = 0
        prev_bad_percent = None
        current_image = image_array.copy()

        while iteration < NUM_ITERATIONS:
            iteration += 1
            # Contains debug arrays if needed
            current_image, debug_bad, changed_debug, pattern_heatmaps, bad_mask = process_iteration(
                current_image, allowed_shapes, candidate_shapes, pattern_shapes, alpha_mask=alpha_channel
            )
            current_bad_percent = (np.count_nonzero(bad_mask) / total_pixels) * 100
            
            if current_bad_percent < MIN_BAD_PERCENTAGE:
                break
            if prev_bad_percent is not None and abs(current_bad_percent - prev_bad_percent) < DELTA_BAD_PERCENTAGE:
                break
            prev_bad_percent = current_bad_percent

        final_cleaned = current_image
        final_bad = debug_bad

    # Reassemble final cleaned image with the original alpha channel if available.
    if alpha_channel is not None:
        final_cleaned_rgba = np.dstack((final_cleaned, alpha_channel))
    else:
        final_cleaned_rgba = final_cleaned

    return Image.fromarray(final_cleaned_rgba, mode="RGBA" if alpha_channel is not None else None), Image.fromarray(final_bad, mode="RGBA" if alpha_channel is not None else None)

def pixel_correction_filter(image: Image) -> Image:
    filtered_image, bad_pixels = process_image_file(image, ALLOWED_SHAPES, CORRECTION_SHAPES, PATTERN_SHAPES)
    return filtered_image, bad_pixels