import numpy as np
import tensorflow as tf

def polynomial(x, coeff):
    poly = coeff[0]
    for i in range(1, len(coeff)):
        poly = poly + coeff[i] * x**i
    return result
def polycnn(image, filter):
    image_row, image_col = image.shape[0], image.shape[1]
    filter_row, filter_col = filter.shape[0], filter.shape[1]
    output_row, output_col = image_row - filter_row + 1, image_col - filter_col + 1
    output = np.empty(output_row, output_col)
    for x in range(output_row):
        for y in range(output_col):
            matrix = image[x : x + fiter_row, y : y + filter_col]
            matrix = np.reshape(matrix, np.size(matrix))
            filter = np.reshape(filter, np.size(filter))
            for a in matrix:
                p = polynomial(a, filter)
                output[x,y] = np.prod(matrix * filter)
    return output
def padding(image, size_of_padding):
    image_row, image_col = image.shape[0], image.shape[1]
    output_row, output_col = image_row - filter_row + (size_of_padding * 2) + 1, image_col - filter_col + (size_of_padding * 2) + 1
    output = np.zeros(output_row, output_col)
    output[size_of_padding:image_row + size_of_padding, size_of_padding:image_col + size_of_padding] =output
    return output
