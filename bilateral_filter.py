import time

import cv2
import numpy as np
import math


def neighborhood(coordinates, radius, xmax, ymax):
    neighbors = []
    x, y = coordinates
    for i in range(-1 * radius, radius + 1):
        for j in range(-1 * radius, radius + 1):
            n_x = -1*abs(abs(x+i)-xmax)+xmax
            n_y = -1*abs(abs(y+j)-ymax)+ymax
            neighbors.append((n_x, n_y))
    return neighbors


def g(x, sigma):
    # implement bottom-up dynamic programming
    return np.exp(-(x**2) / (2 * sigma**2))


def bilateral_filter(input_image, radius, sigma_intensity, sigma_space):
    # detect if input is 1-channel or 3-channel
    output_image = input_image.copy()
    lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    xmax, ymax = len(input_image), len(input_image[0])
    for x in range(len(input_image)):
        for y in range(len(input_image[1])):
            output_image[x, y] = pixel_response(input_image, (x, y), radius, sigma_intensity, sigma_space, xmax, ymax)
    return output_image


def distance(coordinates1, coordinates2):
    return math.hypot(coordinates1[0] - coordinates2[0], coordinates1[1] - coordinates2[1])


def pixel_response(input_image, p_coordinates, radius, sigma_intensity, sigma_space, xmax, ymax):
    numerator = 0
    divisor = 0
    # summation over the neighborhood of the input pixel
    for n_coordinates in neighborhood(p_coordinates, radius, xmax-1, ymax-1):
        # compute the numerator
        space_term = g(distance(p_coordinates, n_coordinates), sigma_space)
        intensity_term = g(cv2.absdiff(int(input_image[p_coordinates]), int(input_image[n_coordinates]))[0], sigma_intensity)
        numerator += space_term * intensity_term * input_image[n_coordinates]
        # compute the divisor
        divisor += space_term * intensity_term
    response = numerator / divisor
    return response


def main():
    img = cv2.imread("test1.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # timing_test(gray_img, 1, 1, 1)
    result_cv2 = cv2.bilateralFilter(gray_img, 5, 20, 20)
    result = bilateral_filter(gray_img, 2, 20, 20)
    cv2.imshow("Homebrew Bilateral, ", result)
    cv2.imshow("CV2 Bilateral", result_cv2)
    cv2.imshow("Originalpicc", img)
    cv2.waitKey()

flags = [i for i in dir(cv2) if i.startswith('COLOR_BGR2') and "lab" in i.lower()]
print(flags)
