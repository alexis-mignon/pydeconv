from objective import ObjFunc

def recover_image(image, image0, psf, w0 = 50, lambda1 = 1e-3, lambda2=10., a=1.0, b=3.0, t=5, **optargs):
    func = ObjFunc(image, w0, lambda1, lambda2, a, t=t)
    func.set_psf(psf)
    return func.optimize_latent(image0, **optargs)

def estimate_psf(image, image0, psf0, w0 = 50, **optargs):
    func = ObjFunc(image, w0)
    func.set_latent(image0)
    print "sumL", func._L.sum()
    return func.optimize_psf(psf0, **optargs)

def unblur(image, psf0, image0 = None, w0 = 50, lambda1 = 1e-3, lambda2=10., a=1.0, b=3.0,
           t=5, maxiter=20, callback=None,
           niter_latent=3, niter_psf=5, **optargs):
    print "entry"
    func = ObjFunc(image, w0, lambda1, lambda2, a, b, t=t)
    
    psf_est = psf0
    img_rec = image if image0 is None else image0

    for it in range(maxiter):
        print "here"
        func.set_psf(psf_est)
        img_rec = func.optimize_latent(img_rec, maxiter=niter_latent, **optargs)
        print "latent estimated..."
        func.set_latent(img_rec)
        psf_est = func.optimize_psf(psf_est, maxiter=niter_psf, alpha_0=1e-5,**optargs)
        print "psf estimated..."
        if callback is not None: callback(it, img_rec, psf_est)
        print 

    return img_rec, psf_est

