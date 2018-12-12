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
    # may need to revert to "return neighbors"
    return np.array([np.array(x) for x in neighbors])


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
        print("Working on row:", x)
        for y in range(y_max):
            # compute pixel response
            numerator = 0
            divisor = 0
            # summation over the neighborhood of the input pixel
            for n_coordinates in neighborhood((x, y), radius, x_max - 1, y_max - 1):
                # compute the numerator
                space_term = g(distance((x, y), n_coordinates), sigma_space)
                # use euclidean distance in CIE-LAB space, as paper suggested
                intensity_term = g(np.linalg.norm(cv2.absdiff(lab_image[(x, y)], lab_image[tuple(n_coordinates)])),
                                   sigma_intensity)
                divisor_add = space_term * intensity_term
                numerator += divisor_add * noflash_image[tuple(n_coordinates)]
                # compute the divisor
                divisor += divisor_add
            output_image[x, y] = numerator / divisor
    return output_image


def distance(coordinates1, coordinates2):
    # wrap in np.abs to ensure np array is input for np.linalg.norm
    coordinates1, coordinates2 = np.abs(coordinates1), np.abs(coordinates2)
    return np.linalg.norm(coordinates1 - coordinates2)


def main():
    img = cv2.imread("test2.png")
    flash_image = cv2.imread("test3b.png")
    noflash_image = cv2.imread("test3a.png")
    jbf = joint_bilateral_filter(flash_image, noflash_image, 10, 2.5, 500)
    cv2.imshow("JBF radius 10, intensity 2.5, space 500", jbf)
    cv2.imwrite("./outputs/jbf_10_2.5_500.png", jbf)
    cv2.waitKey()


def cropped_batch():
    flash_image = cv2.imread("test3b.png")
    noflash_image = cv2.imread("test3a.png")
    x = 120
    y = 20
    h = 160
    w = 160
    crop_flash = flash_image[y:y + h, x:x + w]
    crop_noflash = noflash_image[y:y + h, x:x + w]

    cv2.imwrite("./outputs/test.png", noflash_image)
    for radius in [10]:
        for color_sigma in [1, 5, 10, 2.5]:
            for space_sigma in [10, 20, 50, 100, 200]:
                print("Doing: colorSigma =", color_sigma, "spaceSigma =", space_sigma, "radius =", radius, "at:", strftime("%H:%M:%S"))
                joint_bilat = joint_bilateral_filter(crop_flash, crop_noflash, radius, color_sigma, space_sigma)
                cv2.imwrite("./outputs/cropped_joint_"+str(radius)+"_"+str(color_sigma)+"_"+str(space_sigma)+".png", joint_bilat)
                print("Done:", color_sigma, space_sigma)


main()
