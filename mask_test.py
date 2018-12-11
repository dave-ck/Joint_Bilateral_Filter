import numpy as np
import cv2
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

def distance(coordinates1, coordinates2):
    return math.hypot(coordinates1[0] - coordinates2[0], coordinates1[1] - coordinates2[1])

def gaussian_mask(sigma):
    radius = 0
    while g(radius, sigma) > 0.5:
        radius += 1
    size = 2*radius+1, 2*radius+1, 1
    mask = np.zeros(size, dtype=np.float32)
    home = (radius, radius)
    # update each element in mask to be the gaussianBoi
    total = 0
    for i in neighborhood(home, radius, 100, 100):
        val = g(distance(home, i), sigma)
        mask[i] = val
        total += val
        print(val)
        print(i)
    for i in mask:
        i /= total
    return mask

mask = gaussian_mask(1)
print(mask)
