import Loader
import skimage.filters as filters
import numpy as np

THRESH_TRIANGLE = 1
THRESH_MEAN = 2
THRESH_OTSU = 3
THRESH_YEN = 4
THRESH_MIMIMUM = 5
THRESH_ISODATA = 6
THRESH_LI = 7


def _thresh(im, fun):
    return im > fun(im)


def apply_threshold_window(img_input, setValue, range: tuple):
    (min_val, max_val) = range
    img_input[img_input < max_val] = setValue
    return img_input


def apply_thresholding_algorithm(image, method: int = 1, plot: bool = False):
    name = 'Not Set'
    thresholded = None

    if method == 1:
        name = 'Triangle'
        thresholded = _thresh(image, filters.threshold_triangle)
    elif method == 2:
        name = 'Mean'
        thresholded = _thresh(image, filters.threshold_mean)
    elif method == 3:
        name = 'Otsu'
        thresholded = _thresh(image, filters.threshold_otsu)
    elif method == 4:
        name = 'Yen'
        thresholded = _thresh(image, filters.threshold_yen)
    elif method == 5:
        name = 'Minimum'
        thresholded = _thresh(image, filters.threshold_minimum)
    elif method == 6:
        name = 'Isodata'
        thresholded = _thresh(image, filters.threshold_isodata)
    elif method == 7:
        name = 'Li'
        thresholded = _thresh(image, filters.threshold_li())

    print("[DEBUG] Method '"+name+"' was selected for threshold")
    if plot:
        print("[DEBUG] Comparison with original image requested. Plotting.. ")
        Loader.hist_compare([image, thresholded], ["Original", name])

    return thresholded.astype(np.uint8)


def get_regions(img, threshImg):
    if img.shape != threshImg.shape:
        print("[ERROR] The image sizes doesn´t match")
        return None
    else:
        sizeX, sizeY = img.shape
        back = np.zeros(img.shape)
        front = np.zeros(img.shape)
        for i in np.arange(0, sizeX):
            for j in np.arange(0, sizeY):
                if threshImg[i][j] == 1:
                    front[i][j] = img[i][j]
                else:
                    back[i][j] = img[i][j]
        return back, front


def get_regions_grayscale(img, threshImg):

    if img.shape != threshImg.shape:
        print("[ERROR] The image sizes doesn´t match")
        return None
    else:
        sizeX, sizeY = img.shape
        back = np.zeros(img.shape)
        front = np.zeros(img.shape)
        for i in np.arange(0, sizeX):
            for j in np.arange(0, sizeY):
                if threshImg[i][j] > 245:
                    front[i][j] = img[i][j]
                else:
                    back[i][j] = img[i][j]
        return back, front
