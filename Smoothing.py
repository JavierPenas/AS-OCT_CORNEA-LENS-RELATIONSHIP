import cv2
import scipy.ndimage.filters as filters


def median_filter(img, kSize: int):
    print("[DEBUG] Executing Median Filter Algorithm")
    return cv2.medianBlur(img, kSize)


def denoising_NlMeans(img):
    print("[DEBUG] Executing Non Local Means Denoising Algorithm")
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def gaussian(img, sigma: float):
    print("[DEBUG] Executing Gaussian filter with sigma: " + sigma.__str__())
    return filters.gaussian_filter(img, sigma)


def min_filter(img, size):
    print("[DEBUG] Executing Minimums filter with size: " + size.__str__())
    return filters.minimum_filter(img, size)


def max_filter(img, size):
    print("[DEBUG] Executing Maximums filter with size: " + size.__str__())
    return filters.maximum_filter(img, size)
