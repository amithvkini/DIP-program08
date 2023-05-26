import numpy as np
import cv2
from matplotlib import pyplot as plt

def homomorphic_filter(image, cutoff_freq, gamma_l, gamma_h):
    image_log = np.log1p(np.array(image, dtype="float"))
    image_fft = np.fft.fft2(image_log)

    rows, cols = image_fft.shape
    crow, ccol = rows // 2, cols // 2

    # Construct the high-pass filter
    high_pass_filter = np.ones((rows, cols))
    high_pass_filter[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 0

    # Apply the filter in the frequency domain
    filtered_fft = image_fft * high_pass_filter

    # Perform inverse Fourier Transform
    filtered_image = np.fft.ifft2(filtered_fft)
    filtered_image = np.exp(np.real(filtered_image)) - 1

    # Apply gamma correction
    filtered_image = filtered_image ** gamma_l * (255 ** (1 - gamma_l))
    filtered_image = np.uint8(filtered_image)

    return filtered_image

image = cv2.imread('image.jpg', 0)
filtered_image = homomorphic_filter(image, cutoff_freq=30, gamma_l=0.5, gamma_h=1.5)

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(filtered_image, cmap='gray')
plt.title('Homomorphic Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()