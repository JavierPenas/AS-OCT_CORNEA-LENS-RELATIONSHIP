import cv2
import numpy as np
from matplotlib import pyplot as plt

#ENVIROMENTAL VARIABLES
BASE_PATH = "/Users/javier/Documents/VA_PRACTICAS/Images/P2/"
#WINDOWS BASE PATH
#BASE_PATH = "C:/Users/Javier/DevData/P2/Images/"
GRAY = cv2.IMREAD_GRAYSCALE
IMAGE_TYPE_CLASSIFICATION = "A"

# LOADS IMAGE IN BASE_PATH+filename
def load_image(filename):
    image = cv2.imread(BASE_PATH+filename, GRAY)
    return image


# Prints the selected image
def print_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# Prints the histogram and accumulative histogram of an image
def hist_and_cumsum(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


def hist_compare(images_list: list, titles: list, hist: bool = False):
    if np.mod(len(images_list), 2) == 0:
        rows = int(len(images_list)/2)
    else:
        rows = int(np.ceil(len(images_list)/2))

    for i in np.arange(0, len(images_list)):
        plt.subplot(2, rows, i + 1)
        if hist:
            plt.hist(images_list[i].ravel(), 256, [0, 256])
            plt.xlim([0, 256])
            plt.tight_layout(h_pad=1.5)
        else:
            plt.imshow(images_list[i], 'gray')
            plt.xticks([]), plt.yticks([])
        plt.title(titles[i])

    plt.show()


#Prints historam of image
def histogram(img):
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    # plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


# Histogram equalization
def equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[img]


def inverse_img(img):
    (shapeY, shapeX) = img.shape
    inverse = np.zeros(img.shape)
    for x in np.arange(1, shapeX-5):
        for y in np.arange(1, shapeY-5):
            if img[y][x] == 0:
                inverse[y][x] = 255
            else:
                inverse[y][x] = 0
    #Dilate is here for removing detected imperfections on inverse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inverse = cv2.morphologyEx(inverse, cv2.MORPH_DILATE, kernel)
    return inverse


def bright_and_contrast(source, contrast, bright):
    new_image = cv2.convertScaleAbs(source, alpha=contrast, beta=bright)
    return new_image
