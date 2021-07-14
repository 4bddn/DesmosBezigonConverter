import cv2 as cv
import numpy as np


def mkbitmap(image, f=4, s=2, t=0.45):
    """Convert an image to a bilevel bitmap
    Imitates part of Peter Selinger's `mkbitmap(1)`.
    """
    
    # Grayscale (note: not color-corrected)
    image = np.mean(image, axis=2)
    
    # High-pass filter
    if f is not None:
        ksize = int(3 * f) | 1
        image -= cv.GaussianBlur(image, (ksize, ksize), f, cv.BORDER_DEFAULT)
        image += 128

    # Scaling
    shape = (int(image.shape[1] * s), int(image.shape[0] * s))
    image = cv.resize(image, shape, s, s, cv.INTER_CUBIC)

    # Threshold
    thresh, image = cv.threshold(image, 255 * t, 255, cv.THRESH_BINARY)

    return image


# Example usage
if __name__ == '__main__':
    fname = "image.png"
    image = cv.imread(fname)
    if image is not None:
        image = mkbitmap(image, f=None, t=0.5)
        cv.imwrite("bitmap.png", image)
    else:
        print("Cannot open", fname)
