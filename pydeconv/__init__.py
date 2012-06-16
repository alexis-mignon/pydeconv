from objective import ObjFunc

def recover_image(image, image0, psf, w0 = 50, lambda1 = 1e-3, lambda2 = 10., a = 3.0, t=5, **optargs):
    func = ObjFunc(image, w0, lambda1, lambda2, a, t=t)
    func.set_psf(psf)
    return func.optimize_latent(image0, **optargs)

def estimate_psf(image, image0, psf0, w0 = 50, lambda1 = 1e-3, lambda2 = 10., a = 3.0, cutoff=1e-5, t=5, **optargs):
    func = ObjFunc(image, w0, lambda1, lambda2, a, t=t)
    func.set_latent(image0)
    return func.optimize_psf(psf0, **optargs)
