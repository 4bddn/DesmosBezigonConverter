import cv2 as cv
import numpy as np


def mkbitmap(image, coloravg=True, lowpass=None, highpass=4, scale=2, threshold=0.45, debug=False):
    """Convert an image to a bilevel bitmap.
    Imitates part of Peter Selinger's `mkbitmap(1)`.
    """
    
    # Grayscale
    image = np.mean(image, axis=2) if coloravg else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # High-pass filter
    def gauss(img, f):
        ksize = int(3 * f) | 1
        return cv.GaussianBlur(img, (ksize, ksize), f, cv.BORDER_DEFAULT)
    image = (gauss(image, lowpass) if lowpass else image) \
          + (128 - gauss(image, highpass) if highpass else 0)

    if debug:
        cv.imshow('Filtered image', image)
        cv.waitKey()
    
    # Scaling
    shape = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    image = cv.resize(image, shape, scale, scale, cv.INTER_CUBIC)

    # Threshold
    thresh, image = cv.threshold(image, 255 * threshold, 255, cv.THRESH_BINARY)

    return image


# Example usage
if __name__ == '__main__':
    fname = "image.png"
    image = cv.imread(fname)
    if image is not None:
        image = mkbitmap(image, highpass=None, threshold=0.5)
        cv.imwrite("bitmap.png", image)
    else:
        print("Cannot open", fname)
