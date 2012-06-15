from objective import ObjFunc

def recover_image(image, kernel, w0 = 50, lambda1 = 1e-3, lambda2 = 10., a = 3.0, **optargs):
    func = ObjFunc(image, w0, lambda1, lambda2, a)
    func.set_psf(kernel)
    return func.optimize_latent(image, **optargs)
