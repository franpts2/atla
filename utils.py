import cv2
import numpy as np

def overlay_png(background, overlay, x, y):
    """
    Overlays a PNG with an alpha channel onto a background image.
    """

    # dimensions
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    # boundary check (so it doesn't crash at screen edges)
    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return background

    # cropping if the image is partially off-screen
    x1, x2 = max(x, 0), min(x + ov_w, bg_w)
    y1, y2 = max(y, 0), min(y + ov_h, bg_h)

    # parts of the overlay to use
    ov_x1, ov_x2 = x1 - x, x2 - x
    ov_y1, ov_y2 = y1 - y, y2 - y

    # extract regions
    overlay_crop = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    background_roi = background[y1:y2, x1:x2]

    # blend using Alpha channel
    if overlay_crop.shape[2] < 4:
        # If no alpha channel, just paste it
        background[y1:y2, x1:x2] = overlay_crop[:, :, :3]
    else:
        alpha = overlay_crop[:, :, 3] / 255.0
        for c in range(3):
            background_roi[:, :, c] = (alpha * overlay_crop[:, :, c] +
                                      (1.0 - alpha) * background_roi[:, :, c])
    
    return background