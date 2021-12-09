import numpy as np
import matplotlib.pyplot as plt
import math
from ppp.convolution import convolution
 
 
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
 
def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()
 
    return kernel_2D
 
 
def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)
 
 
if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="Path to the image")
    # args = vars(ap.parse_args())
 
    # image = cv2.imread(args["image"])
    
    nx, ny = 500, 500
    x, y = np.linspace(0, .5, nx+1), np.linspace(0, .4, ny+1)
    Lx, Ly = x[-1]-x[0], y[-1]-y[0]
    h, L = .5, .1
    # H0, ΔH = (1+h)/2, (1-h)/2
    X, Y = np.meshgrid(x, y)
    # H = np.sin(np.pi*(X-x[0])/Lx) * (H0 + 
    #         ΔH * np.tanh(((Y-y[0])-L)/(1*L)))
    
    H = np.ones((ny+1, nx+1))
    H[Y < .1] = .1
    H[(Y < .15) & (.2 < X) & (X < .3)] = .1
 
    print(gaussian_blur(H, 25, verbose=True))
