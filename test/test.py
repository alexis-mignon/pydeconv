from scipy.misc import imread, imshow
from pydeconv.utils import convolve2d
from pydeconv import recover_image
import numpy as np

img = imread("image.png").astype("float")
k = imread("psf.png").astype("float")
k/=k.sum()

img_blurred = convolve2d(img, k)
imshow(img_blurred)

img_rec = recover_image(img_blurred, k, verbose=True)
print img.shape, img_blurred.shape, img_rec.shape
imshow(np.hstack([img, img_blurred, img_rec]))




