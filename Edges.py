import cv2
import scipy.ndimage as nd
import Loader as ld
import numpy as np

def sobel(img, axis=-1):
    if axis == 0:
        print("[DEBUG] Calculating Sobel on X axis")
        return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    elif axis == 1:
        print("[DEBUG] Calculating Sobel on Y axis")
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    else:
        print("[ERROR] You must specify an axis to calculate Sobel: '0' -> X axis; '1' -> Y axis.   Skipping... ")
        return None


def difference_of_gaussian(img,sigma1: float, sigma2: float):
    print("[DEBUG] Calculating DoG with sigmas: s1: "+sigma1.__str__()+" s2: "+sigma2.__str__())
    s1 = cv2.GaussianBlur(img, (11, 11), sigma1)
    s2 = cv2.GaussianBlur(img, (11, 11), sigma2)

    return s2 - s1


def laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F)


def laplacian_of_gaussian(img, sig):
    return nd.gaussian_laplace(img, sigma=sig)


def canny(img, th1: int, th2: int, sigma: float = 0):
    if sigma != 0:
        img = cv2.GaussianBlur(img, (11, 11), sigma)
        ld.print_image(img)
    return cv2.Canny(img, th1, th2)


def applySobelY(image):

    sY = sobel(image, 1)
    out = np.zeros(image.shape)
    (shapeY, shapeX) = sY.shape
    for px in np.arange(0, shapeX):
        for py in np.arange(0, shapeY):
            if sY[py][px] > 0:
                out[py][px] = 255

    return out.astype("uint8")
