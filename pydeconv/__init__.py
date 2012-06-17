from objective import ObjFunc

def recover_image(image, image0, psf, w0 = 50, lambda1 = 1e-3, lambda2 = 10., a = 3.0, t=5, **optargs):
    func = ObjFunc(image, w0, lambda1, lambda2, a, t=t)
    func.set_psf(psf)
    return func.optimize_latent(image0, **optargs)

def estimate_psf(image, image0, psf0, w0 = 50, **optargs):
    func = ObjFunc(image, w0)
    func.set_latent(image0)
    print "sumL", func._L.sum()
    return func.optimize_psf(psf0, **optargs)

def unblur(image, psf0, w0 = 50, lambda1 = 1e-3, lambda2 = 10., a = 3.0, t=5, maxiter=20, callback=None, **optargs):
    func = ObjFunc(image, w0, lambda1, lambda2, a, t=t)

    psf_est = psf0
    img_rec = image

    for it in range(maxiter):
        func.set_psf(psf_est)
        img_rec = func.optimize_latent(img_rec, maxiter=3, **optargs)
        print "latent estimated..."
        func.set_latent(img_rec)
        psf_est = func.optimize_psf(psf_est, maxiter=5, alpha_0=1e-5,**optargs)
        print "psf estimated..."
        callback(it, img_rec, psf_est)
        print 

    return img_rec, psf_est

