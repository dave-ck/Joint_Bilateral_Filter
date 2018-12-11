from time import strftime
import cv2
import numpy as np
import math


def neighborhood(coordinates, radius, xmax, ymax):
    neighbors = []
    x, y = coordinates
    for i in range(-1 * radius, radius + 1):
        for j in range(-1 * radius, radius + 1):
            n_x = -1 * abs(abs(x + i) - xmax) + xmax
            n_y = -1 * abs(abs(y + j) - ymax) + ymax
            neighbors.append((n_x, n_y))
    return neighbors


def g(x, sigma):
    # implement bottom-up dynamic programming
    return np.exp(-(x ** 2) / (2 * sigma ** 2))


def bilateral_filter(input_image, radius, sigma_intensity, sigma_space):
    # detect if input is 1-channel or 3-channel
    output_image = input_image.copy()
    lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    x_max, y_max = len(input_image), len(input_image[0])
    for x in range(x_max):
        for y in range(y_max):
            # compute pixel response
            numerator = 0
            divisor = 0
            # summation over the neighborhood of the input pixel
            for n_coordinates in neighborhood((x, y), radius, x_max - 1, y_max - 1):
                # compute the numerator
                space_term = g(distance((x, y), n_coordinates), sigma_space)
                # use euclidean distance in CIE-LAB space, as paper suggested
                intensity_term = g(np.linalg.norm(cv2.absdiff(lab_image[(x, y)], lab_image[n_coordinates])),
                                   sigma_intensity)
                divisor_add = space_term * intensity_term
                numerator += divisor_add * input_image[n_coordinates]
                # compute the divisor
                divisor += divisor_add
            output_image[x, y] = numerator / divisor
    return output_image


def joint_bilateral_filter(flash_image, noflash_image, radius, sigma_intensity, sigma_space):
    # detect if input is 1-channel or 3-channel
    output_image = flash_image.copy()
    lab_image = cv2.cvtColor(flash_image, cv2.COLOR_BGR2LAB)
    x_max, y_max = len(flash_image), len(flash_image[0])
    for x in range(x_max):
        for y in range(y_max):
            # compute pixel response
            numerator = 0
            divisor = 0
            # summation over the neighborhood of the input pixel
            for n_coordinates in neighborhood((x, y), radius, x_max - 1, y_max - 1):
                # compute the numerator
                space_term = g(distance((x, y), n_coordinates), sigma_space)
                # use euclidean distance in CIE-LAB space, as paper suggested
                intensity_term = g(np.linalg.norm(cv2.absdiff(lab_image[(x, y)], lab_image[n_coordinates])),
                                   sigma_intensity)
                divisor_add = space_term * intensity_term
                numerator += divisor_add * noflash_image[n_coordinates]
                # compute the divisor
                divisor += divisor_add
            output_image[x, y] = numerator / divisor
    return output_image


def distance(coordinates1, coordinates2):
    # wrap in np.abs to ensure np array is input for np.linalg.norm
    coordinates1, coordinates2 = np.abs(coordinates1), np.abs(coordinates2)
    return np.linalg.norm(coordinates1-coordinates2)


def write_image(img):
    path = "outputs/"
    path += strftime("%d %b %Y %H:%M:%S")


def main():
    img = cv2.imread("test2.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # timing_test(gray_img, 1, 1, 1)
    result_cv2 = cv2.bilateralFilter(img, 5, 20, 20)
    result = bilateral_filter(img, 2, 20, 20)
    cv2.imshow("Homebrew Bilateral, ", result)
    cv2.imshow("CV2 Bilateral", result_cv2)
    cv2.imshow("Originalpicc", img)

    cv2.waitKey()



main()
