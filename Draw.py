import numpy as np
from matplotlib import pyplot as plt
import cv2


def find_next_white(pos: tuple, img):
    (shapeY, shapeX) = img.shape
    n = []
    for y in np.arange(pos[0], shapeY - 5):
        if img[y][pos[0]] > 0:
            n.append((y, pos[0]))
    return n


def trim_error_measures(measures, positions):
    d2 = []
    p2 = []
    it2 = 0
    for d in measures:
        if d > (np.sum(measures) / len(measures) * 0.5):
            d2.append(d)
            p2.append(positions[it2])
        it2 = it2 + 1

    return d2, p2, (np.sum(d2) / len(d2) * 0.5)


def find_vertical_lines(edgeImage):
    inverse = edgeImage
    #inverse = Loader.inverse_img(edgeImage)
    #Loader.print_image(inverse)
    (shapeY, shapeX) = edgeImage.shape
    lineas = []
    for x in np.arange(1, shapeX - 5):
        y = 0
        # Busco el primer pixel blanco
        while y < shapeY and inverse[y][x] == 0 :
            y = y + 1
        # Estoy en el primer punto vertical blanco
        # Avanzo hacia arriba hasta el primer hueco negro
        while y < shapeY and  inverse[y][x] != 0:
            y = y + 1
        # Estoy en hueco inter-lente
        while y < shapeY and inverse[y][x] == 0:
            y = y + 1
        # Final de lente
        while y < shapeY and inverse[y][x] != 0:
            y = y + 1
        # He acabado la cornea, empieza lo que quiero guardar, blanco
        vertical = []
        while y < shapeY and inverse[y][x] == 0:
            vertical.append((y, x))
            y = y + 1
        # Acabo el trozo que me interesa, lo añado a lineas
        if len(vertical) != 0:
            lineas.append(vertical.copy())
    return lineas


def calculate_differences(lineas):
    diferencias = []
    posiciones = []
    paint = 0
    for linea in lineas:
        distancia = np.abs(linea[0][0] - linea[len(linea) - 1][0])
        posX = linea[0][1]
        diferencias.append(distancia)
        posiciones.append(posX)

    return trim_error_measures(diferencias, posiciones)


def lines_image(lines, img, intervalo, error):
    #DIUJA LAS LINEAS, DEJANDO ALGUNOS HUECOS

    output = img.copy()
    paint = 0
    margin = error*4
    for linea in lines:
        length = np.abs(linea[0][0] - linea[len(linea) - 1][0])
        if (paint == intervalo) and (length<margin):
            paint = 0
            for dot in linea:
                output[dot[0]][dot[1]] = 255
        paint = paint+1
    return output


def white_contour(edgeImage):
    out = np.zeros(edgeImage.shape)
    (z, contornos, jerarquia) = cv2.findContours(edgeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for c in contornos:
        if cv2.contourArea(c) < 1000:
            continue
        else:
            cnts.append(c)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 3)
    return out


def draw_graph_distance(measures, positions):
    plt.plot(positions, measures)
    plt.xticks(np.arange(min(positions)-2, max(positions), 100.0)), plt.yticks(np.arange(min(measures), max(measures), 1.0))
    plt.show()
