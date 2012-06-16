from scipy.misc import imread, imshow, imsave
from pydeconv.utils import convolve2d, project_simplex
from pydeconv import recover_image, estimate_psf
import numpy as np
import sys

img = imread("image.png").astype("float")
k = imread("psf.png").astype("float")
k/=k.sum()

img_blurred = convolve2d(img, k)

psf0 = project_simplex(np.identity(k.shape[0]))

print np.abs(img - img_blurred).mean()
img_rec = recover_image(img_blurred, img_blurred, psf0, verbose=True, w0 = 50, lambda1=1e1, lambda2=20, a=3.0e-1, maxiter=5, t=5)
imshow(np.hstack([img, img_blurred, img_rec]))
print np.abs(img - img_rec).mean()
imsave("img_rec.png",img_rec)
img_rec=imread("img_rec.png")

print np.abs(k - psf0).mean()
psf_est = estimate_psf(img_blurred, img_rec, psf0, verbose=True, w0 = 50, lambda1=1e1, lambda2=20, a=3.0e-1, cutoff = 1e-6, maxiter=5, t=5, method="")
print np.abs(k - psf_est).mean()
imshow(psf_est)











