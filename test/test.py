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

def score(i,j):
    return ((i-j)**2).mean()
    
print "start real:", score(img, img_blurred)
print "obj_start:", score(convolve2d(img, psf0), img_blurred)
print "img sum", img.sum(), img_blurred.sum(), project_simplex(img, img.sum()).sum()
img_rec = recover_image(img_blurred, img_blurred, psf0,
    verbose=True, w0 = 50,
    lambda1=1.0, lambda2=20, a=1.0e+3, maxiter=5, t=5, method="gd", alpha_0=1e-6)
imshow(np.hstack([img, img_blurred, img_rec]))
print "conserv:", img_rec.sum()/img_blurred.sum()
print "real:", score(img, img_rec)
print "obj", score(convolve2d(img_rec, psf0), img_blurred)


psf_est = estimate_psf(img_blurred, img_rec, psf0, maxiter=5,
    verbose=True, w0 = 50.0, method="gd", alpha_0=1e-5)

print "obj", score(convolve2d(img_rec, psf_est), img_blurred)
img_rec = recover_image(img_blurred, img_rec, psf_est,
    verbose=True, w0 = 50,
    lambda1=1e-4, lambda2=20, a=3.0e-1, maxiter=5, t=5, method="gd", alpha_0=1e-6)
imshow(np.hstack([img, img_blurred, img_rec]))
print "real:", score(img, img_rec)
print "obj", score(convolve2d(img_rec, psf_est), img_blurred)

#~ 
#~ print np.abs(k - psf0).mean()
#~ psf_est = estimate_psf(img_blurred, img_rec, psf0,
    #~ verbose=True, w0 = 50, lambda1=1e1,
    #~ lambda2=20, a=3.0e-1,
    #~ cutoff = 1e-6, maxiter=5, t=5, method="gd", alpha_0=1e-6)
#~ print np.abs(k - psf_est).mean()
#~ imshow(psf_est)











