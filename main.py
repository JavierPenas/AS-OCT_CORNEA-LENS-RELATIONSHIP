import Loader
import cv2
import numpy as np
import Smoothing as smooth
import Thresholding as thresh
import Draw as dw
import Edges as edg

# A: NORMAL, B: XTREM NOISY
IMAGE_TYPE_CLASSIFICATION = "A"
V_EDGE_LIMIT = 400


#
# COMPARTIVA DE MÉTODOS PROBADOS PARA ELIMINAR
# EL RUIDO MOTEADO PRESENTE EN LAS IMAGENES DE EJEMPLO
#
def denoising_comparison(input_image, hist: bool = False):
    # NON LOCAL MEAN DENOISING
    nl_means_denoised_img = smooth.denoising_NlMeans(input_image)
    # MEDIAN FILTER DENOISING
    mean_denoised_img = smooth.median_filter(input_image, 9)
    mean_denoised_img = smooth.median_filter(input_image, 9)
    # GAUSSIAN DENOISING
    gaussian_denoised = smooth.gaussian(input_image, 1.5)
    # MINIMUM FILTER
    minimum_denoised = smooth.min_filter(input_image, (5, 5))
    # MAXIMUM FILTER
    maximum_denoised = smooth.max_filter(input_image, (5, 5))

    denoised_imgs = [img, gaussian_denoised, mean_denoised_img, nl_means_denoised_img, minimum_denoised, maximum_denoised]
    denoised_titles = ["Original", "Denoised Gaussian", "Median Filtered", "NL Means Filter", "Minimums Filter", "Maximums Filter"]
    Loader.hist_compare(denoised_imgs, denoised_titles, hist)

    return denoised_imgs, denoised_titles


#
# COMPARTIVA DE MÉTODOS PROBADOS PARA
# REALIZAR LA SEGMENTACION DE LA IMAGEN
#
def thresholding_comparison(input_image):

    # Thresholding algoritms precalculation for comparison
    triangle = thresh.apply_thresholding_algorithm(input_image, thresh.THRESH_TRIANGLE)
    mean = thresh.apply_thresholding_algorithm(input_image, thresh.THRESH_MEAN)
    otsu = thresh.apply_thresholding_algorithm(input_image, thresh.THRESH_OTSU)
    yen = thresh.apply_thresholding_algorithm(input_image, thresh.THRESH_YEN)
    minimum = thresh.apply_thresholding_algorithm(input_image, thresh.THRESH_MIMIMUM)
    isodata = thresh.apply_thresholding_algorithm(input_image, thresh.THRESH_ISODATA)
    # li = thresh.apply_thresholding_algorithm(img, thresh.THRESH_LI)

    thresholded_imgs = [input_image, triangle, mean, otsu, yen,
                        minimum, isodata]
    thresholded_titles = ["Original", "Triangle", "Mean",
                          "Otsu", "Yen", "Minimum", "Isodata"]
    Loader.hist_compare(thresholded_imgs, thresholded_titles)


def edges_comparison(input_image):
    log = edg.laplacian_of_gaussian(input_image, 2)
    dog = edg.difference_of_gaussian(input_image, 1.0, 2.5)
    sobelX = edg.sobel(input_image, 0)
    canny = edg.canny(input_image, 100, 200, sigma=1.5)
    Loader.hist_compare([log, dog, canny, sobelX], ["LoG", "DoG", "Canny", "Sobel"])


def fill_cornea(edge_image):

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
    res1 = cv2.morphologyEx(edge_image, cv2.MORPH_DILATE, k1)
    print("DILATED")
    #Loader.print_image(res1)
    enhance_black = smooth.min_filter(res1, 7)
    print("BLACk")
    #Loader.print_image(enhance_black)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    res2 = cv2.morphologyEx(enhance_black, cv2.MORPH_ERODE, k2)
    print("eroded")
    #Loader.print_image(res2)
    res3 = smooth.max_filter(res2, 3)
    res3 = smooth.gaussian(res3, 1.5)
    print("dilated")
    #Loader.print_image(res3)

    return res3


# def apply_to_all():
#     for i in np.arange(3, 13):
#
#         img = Loader.load_image("im"+i.astype(str)+".jpeg")
#         #TODO process image must return the output image
#         res = process_image(img)
#         print("res/img"+i.astype(str)+".jpeg")
#         cv2.imwrite(Loader.BASE_PATH + "res/img"+i.astype(str)+".jpeg", res)


def define_image_properties(image):
    (shapeY, shapeX) = image.shape
    white_region_size = 0
    black_region_size = 0
    total_size = shapeY * shapeX
    vertical_edge = 0
    for x in np.arange(10, shapeX - 10):
        for y in np.arange(10, shapeY - 10):
            if image[y][x] == 0:
                black_region_size = black_region_size + 1
            else:
                white_region_size = white_region_size + 1
                if image[y][x + 1] == 0 and image[y - 1][x + 1] == 0 \
                        and image[y + 1][x + 1] == 0:
                    vertical_edge = vertical_edge + 1
                if image[y][x - 1] == 0 and image[y - 1][x - 1] == 0 \
                        and image[y + 1][x - 1] == 0:
                    vertical_edge = vertical_edge + 1
    # print("IMAGE " + i.__str__() + " VALUES:")
    # print("BLACK REGION = ", black_region_size.__str__())
    # print("V EDGE = ", vertical_edge.__str__())
    # print("WHITE REGION = ", white_region_size.__str__())
    # print("TOTAL SIZE = ", total_size.__str__())
    percentage = (white_region_size * 100) / total_size
    # print("WHITE REGION PERCENTAGE = ", percentage.__str__())

    if vertical_edge > V_EDGE_LIMIT:
        return "B"
    else:
        return "A"


def draw_contours(edgeImage, outputCanvas):

    (z, contornos, jerarquia) = cv2.findContours(edgeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for c in contornos:
        if cv2.contourArea(c) < 1000:
            continue
        else:
            cnts.append(c)
    cv2.drawContours(outputCanvas, cnts, -1, (0, 0, 255), 3)

    return outputCanvas


if __name__ == '__main__':

    print("[DEBUG] Load image from local sources")
    # img = Loader.load_image("test/th.jpeg")
    # Loader.print_image(stretched)
    # Loader.print_image(smoothed_thresholded)
    # Loader.print_image(front)

    for i in np.arange(1, 13):
        img = Loader.load_image("im"+i.__str__()+".jpeg")
        print("Loaded image "+"im"+i.__str__()+".jpeg")
        # DENOISING IMAGE
        denoised_img = smooth.median_filter(img, 9)
        denoised_img = smooth.median_filter(denoised_img, 7)

        # thresholding_comparison(denoised_img)
        th_img = thresh.apply_thresholding_algorithm(denoised_img, thresh.THRESH_TRIANGLE)
        #Suavizamos los bordes marcados por la segmentacion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        stretched = cv2.morphologyEx(th_img, cv2.MORPH_ERODE, kernel)
        smoothed_thresholded = smooth.max_filter(stretched, 5)
        #Obtenemos las areas correspondientes a la imagen original, despues del segmentado
        back, front = thresh.get_regions(denoised_img, stretched)

        IMAGE_TYPE_CLASSIFICATION = define_image_properties(smoothed_thresholded)
        if IMAGE_TYPE_CLASSIFICATION == "A":
            # RESOLUTION 1
            print("Executing custom resolution, mode 1")
            # EDGE DETECTION
            #eq = Loader.equalization(front.astype("uint8"))
            eq = Loader.bright_and_contrast(front.astype("uint8"), 1.0, 20)
            eq = smooth.gaussian(eq, 1.5)
            #Loader.print_image(eq)
            edges = edg.laplacian_of_gaussian(eq, 2)
            #Loader.print_image(edges)
            normalized = fill_cornea(edges)
            #Loader.print_image(normalized)
            # Calculate distances in the cornea-lens region
            lineas = dw.find_vertical_lines(normalized)
            diferencias,posiciones,error = dw.calculate_differences(lineas)
            dw.draw_graph_distance(diferencias, posiciones)
            output_image = dw.lines_image(lineas, img, 7, error)

            contourned_img = draw_contours(normalized, output_image)

            print("Saved image by M1 " + "im" + i.__str__() + ".jpeg")
            cv2.imwrite(Loader.BASE_PATH+"/contoursM1/"+"im"+i.__str__()+".jpeg", contourned_img)
        else:
            # RESOLUTION 2

            eq = smooth.max_filter(front.astype("uint8"), 7)
            eq = Loader.bright_and_contrast(eq, 2.0, 25)
            ret, thresh1 = cv2.threshold(eq, 240, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
            dilated = cv2.morphologyEx(thresh1, cv2.MORPH_DILATE, kernel)

            #EDGE DETECTION
            out = edg.applySobelY(dilated)
            # Calculate distances in the cornea-lens region
            out_aux = dw.white_contour(out)
            lineas = dw.find_vertical_lines(out_aux)
            diferencias,posiciones,error = dw.calculate_differences(lineas)
            dw.draw_graph_distance(diferencias, posiciones)
            output_image = dw.lines_image(lineas, img, 7, error)
            contourned_img = draw_contours(out, output_image)
            Loader.print_image(contourned_img)

            print("Saved image " + "im" + i.__str__() + ".jpeg")
            cv2.imwrite(Loader.BASE_PATH+"/contoursM2/"+"im"+i.__str__()+".jpeg", contourned_img)

    print("[DEBUG] End of processing")



