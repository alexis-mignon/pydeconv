import numpy as np
from scipy.optimize import fmin_cg, brent
from scipy.misc import imshow
import utils

def steepest_descent(func, X0, grad, maxiter = 100, verbose = False, project=None, alpha_0=1.0, gtol = 1e-5,  **args):
    def func_one(X,dir_):
        def inner(alpha):
            if project is None:
                return func(X + alpha*dir_)
            else:
                px = project(X + alpha*dir_)
                f = func(px)
                #~ print "q", alpha, f , px.sum(), dir_.sum()
                return func(project(X + alpha*dir_))
        return inner

    def print_info(iter, objective, gnorm):
        print "iter:", iter, ", objective:", objective, ", ||grad||_inf:", gnorm

    failed = False
    fold = np.inf
    f = func(X0)
    g = grad(X0)
    X = X0
    alpha = alpha_0
    gnorm = np.abs(g).max()
    iter = 0
    if verbose:
        print_info(iter, f, gnorm)

    inner_maxiter0 = 3
    inner_maxiter = inner_maxiter0
    while True:
        iter +=1
        func_ = func_one(X,-g)
        alpha = np.abs(alpha)
        alpha, fopt = brent(func_, brack=(0,alpha), full_output=True, maxiter=inner_maxiter, **args)[:2]
        if fopt >= f: alpha = 0.0
        if alpha == 0.0:
            if not failed:
                failed = True
                alpha = alpha_0
                inner_maxiter = 100
                continue
            else:
                inner_maxiter = inner_maxiter0
            if verbose:
                print "Could not optimize in the descent direction"
            break
        if project is None:
            X = X - alpha * g
        else:
            X = project(X - alpha * g)
        fold = f
        f = fopt
        g = grad(X)

        gnorm = np.abs(g).max()
        if verbose: print_info(iter,f,gnorm)
        if gnorm < gtol :
            if verbose:
                print "gradient convergence reached"
            break
        if iter >= maxiter:
            if verbose:
                print "maximum number of iterations reached"
            break
    return X

class ObjFunc(object):
    def __init__(self, I0, w0, lambda1=1e-3, lambda2=10.,  a=1.0, b=3.0, cutoff = 1e-6, t=5):
        self.w0 = w0
        self.w1 = self.w0/2
        self.w2 = self.w1/4
        self.t = t

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.a = a
        self.b = b
        self.cutoff = cutoff

        self.I0 = I0
        self.I0_sum = self.I0.sum(0).sum(0)
        
        self._dxI0 = utils.dx(self.I0)
        self._dyI0 = utils.dy(self.I0)
        self._dxxI0 = utils.dx_b(self._dxI0)
        self._dyyI0 = utils.dy_b(self._dyI0)
        self._dxyI0 = utils.dx_b(self._dyI0)
        
        self._dxL = np.zeros(self.I0.shape)
        self._dyL = np.zeros(self.I0.shape)
        self._dxxL = np.zeros(self.I0.shape)
        self._dyyL = np.zeros(self.I0.shape)
        self._dxyL = np.zeros(self.I0.shape)
        
        self._J = np.zeros(self.I0.shape)
        self._dxJ = np.zeros(self.I0.shape)
        self._dyJ = np.zeros(self.I0.shape)
        self._dxxJ = np.zeros(self.I0.shape)
        self._dyyJ = np.zeros(self.I0.shape)
        self._dxyJ = np.zeros(self.I0.shape)


        self._M = None
        self._simplex_normal = None

    @property
    def M(self):
        if self._M is None:
            if self._P is None:
                raise ValueError("The kernel has not been set")
            var_k = np.ones(self._P.shape)/self._P.size
            I0_m = utils.convolve2d(self.I0, var_k)
            I02_m = utils.convolve2d(self.I0**2, var_k)
            var = (I02_m - I0_m**2)
            if var.ndim == 3:
                var = I02_m.mean(-1)
            self._M = (var < self.t*self.t).astype("uint8")
            print "smooth parts", self._M.mean()
        return self._M

    def projectP(self, P):
#~ 
        #~ pcut = P.max() * self.cutoff
        #~ if pcut <0 :
            #~ raise ValueError("P was all negative!")
        #~ P = P * (P > pcut)
        #~ P /= P.sum()
        #~ return P
        return utils.project_simplex(P)

    def projectL(self, L):
        if L.ndim == 3:
            pL = np.zeros(L.shape,"float")
            for i in range(L.shape[2]):
                pL[...,i] = utils.project_simplex(L[...,i], self.I0_sum[i])
            return pL
        else:
            return utils.project_simplex(L, self.I0_sum)

    def __call__(self, L, P, psf_only=False):
        # compute
        utils.convolve2d(L, P, output=self._J)
        utils.dx(self._J, output=self._dxJ)
        utils.dy(self._J, output=self._dyJ)
        utils.dx(L, output=self._dxL)
        utils.dy(L, output=self._dyL)
        utils.dx_b(self._dxJ, output=self._dxxJ)
        utils.dy_b(self._dyJ, output=self._dyyJ)
        utils.dx_b(self._dyJ, output=self._dxyJ)
        # enegery for data compatibility

        R = self._J - self.I0
        dxR = self._dxJ - self._dxI0
        dyR = self._dyJ - self._dyI0
        dxxR = self._dxxJ - self._dxxI0
        dyyR = self._dyyJ - self._dyyI0
        dxyR = self._dxyJ - self._dxyI0

        E = self.w0 * utils.norm2(R)
        E += self.w1 * utils.norm2(dxR)
        E += self.w1 * utils.norm2(dyR)
        #~ E += self.w2 * utils.norm2(dxxR)
        #~ E += self.w2 * utils.norm2(dyyR)
        #~ E += self.w2 * utils.norm2(dxyR)

        if not psf_only:
            # energy for global prior
            E += self.lambda1 * utils.global_prior(self._dxL, self.a, self.b)
            E += self.lambda1 * utils.global_prior(self._dyL, self.a, self.b)
            # energy for local prior
            E += self.lambda2 * utils.local_prior(self._dxL, self._dxI0, self.M)
            E += self.lambda2 * utils.local_prior(self._dyL, self._dyI0, self.M)

        return E/self.I0.size

    def grad_L(self, L, P):
        # compute
        utils.convolve2d(L, P, output=self._J)
        utils.dx(self._J, self._dxJ)
        utils.dy(self._J,self._dyJ)
        
        utils.dx(L, self._dxL)
        utils.dy(L, self._dyL)
        
        utils.dx_b(self._dxJ, self._dxxJ)
        utils.dy_b(self._dyJ, self._dyyJ)
        utils.dx_b(self._dyJ, self._dxyJ)

        R = self._J - self.I0
        dxR = self._dxJ - self._dxI0
        dyR = self._dyJ - self._dyI0
        dxxR = self._dxxJ - self._dxxI0
        dyyR = self._dyyJ - self._dyyI0
        dxyR = self._dxyJ - self._dxyI0
        # enegery for data compatibility

        
        dxP = utils.dx(P)
        dyP = utils.dy(P)
        dxxP = utils.dx_b(dxP)
        dyyP = utils.dy_b(dyP)
        dxyP = utils.dx_b(dyP)

        dL = np.zeros(L.shape)
        
        dL += self.w0 * utils.grad_L(P, R)
        dL += self.w1 * utils.grad_L(dxP, dxR)
        dL += self.w1 * utils.grad_L(dyP, dyR)
        #~ dL += self.w2 * utils.grad_L(dxxP, dxxR)
        #~ dL += self.w2 * utils.grad_L(dyyP, dyyR)
        #~ dL += self.w2 * utils.grad_L(dxyP, dxyR)

        dL += self.lambda1 * utils.grad_global_prior_x(self._dxL, self.a, self.b)
        dL += self.lambda1 * utils.grad_global_prior_y(self._dyL, self.a, self.b)

        dL += self.lambda2 * utils.grad_local_prior_x(self._dxL, self._dxI0, self.M)
        dL += self.lambda2 * utils.grad_local_prior_y(self._dyL, self._dyI0, self.M)
        
        return (dL - dL.mean()) /self.I0.size 
        
    def grad_P(self, L, P):
        # compute
        utils.convolve2d(L, P ,output=self._J)
        utils.dx(self._J, self._dxJ)
        utils.dy(self._J,self._dyJ)
        utils.dx(L, self._dxL)
        utils.dy(L, self._dyL)
        #~ 
        #~ utils.dx_b(self._dxL, self._dxxL)
        #~ utils.dy_b(self._dyL, self._dyyL)
        #~ utils.dx_b(self._dyL, self._dxyL)
        #~ 
        #~ utils.dx_b(self._dxJ, self._dxxJ)
        #~ utils.dy_b(self._dyJ, self._dyyJ)
        #~ utils.dx_b(self._dyJ, self._dxyJ)
        
        R = self._J - self.I0
        dxR = self._dxJ - self._dxI0
        dyR = self._dyJ - self._dyI0
        #~ dxxR = self._dxxJ - self._dxxI0
        #~ dyyR = self._dyyJ - self._dyyI0
        #~ dxyR = self._dxyJ - self._dxyI0

        dP = self.w0 * utils.grad_P(P.shape, L, R)
        dP += self.w1 * utils.grad_P(P.shape, self._dxL, dxR)
        dP += self.w1 * utils.grad_P(P.shape, self._dyL, dyR)
        #~ dP += self.w2 * utils.grad_P(P.shape, self._dxxL, dxxR)
        #~ dP += self.w2 * utils.grad_P(P.shape, self._dyyL, dyyR)
        #~ dP += self.w2 * utils.grad_P(P.shape, self._dxyL, dxyR)
        
        #~ if self._simplex_normal is None:
            #~ self._simplex_normal = np.ones(P.shape)/np.sqrt(P.size)
        #~ dPnorm = np.dot(dP.flatten(), self._simplex_normal.flatten()) * self._simplex_normal
        dP -= dP.mean()
        return dP/self.I0.size

    def set_psf(self, P):
        self._P = self.projectP(P)

    def set_latent(self, L):
        self._L = self.projectL(L)

    def eval_L(self, L):
        if self._P is None:
            raise ValueError("the kernel is not set")
        return self(L, self._P)

    def eval_P(self, P):
        if self._L is None:
            raise ValueError("the latent image is not set")
        return self(self._L, P, psf_only=True)

    def eval_grad_L(self, L):
        if self._P is None:
            raise ValueError("the kernel is not set")
        return self.grad_L(L, self._P)        

    def eval_grad_P(self, P):
        if self._L is None:
            raise ValueError("the latent image is not set")
        return self.grad_P(self._L, P)        

    def projectXL(self, X):
        return self.L_to_X(self.projectL(self.X_to_L(X)))
    
    def projectXP(self, X):
        return utils.project_simplex(X)

    def X_to_P(self, X):
        return X.reshape(self._P.shape)

    def P_to_X(self, P):
        return P.flatten()

    def X_to_L(self, X):
        return X.reshape(self.I0.shape)

    def L_to_X(self, L):
        return L.flatten()

    def _eval_XP(self, X):
        return self.eval_P(self.X_to_P(X))

    def _eval_XL(self, X):
        return self.eval_L(self.X_to_L(X))

    def _eval_grad_XP(self, X):
        return self.P_to_X(self.eval_grad_P(self.X_to_P(X)))

    def _eval_grad_XL(self, X):
        return self.L_to_X(self.eval_grad_L(self.X_to_L(X)))

    def optimize_latent(self, L0=None, method="gd", verbose=True, **args):
        if L0 is not None:
            self.set_latent(L0)
        else:
            L0 = self._L
        X0 = self.L_to_X(L0)

        if method=="cg":
            if verbose:
                def callback(X):
                    print "objective:",self._eval_XL(X), "||grad||_inf:", np.abs(self._eval_grad_XL(X)).max()
            else:
                callback = None
        
            Xopt = fmin_cg(self._eval_XL, X0, self._eval_grad_XL, callback=callback, **args)
        else:
            Xopt = steepest_descent(self._eval_XL, X0, self._eval_grad_XL, project=self.projectXL, verbose=verbose, **args)

        Lopt = self.X_to_L(Xopt)
        return Lopt

    def optimize_psf(self, P0=None, method="gd", verbose=True, **args):
        if P0 is not None:
            self.set_psf(P0)
        else:
            P0 = self._P
        X0 = self.P_to_X(P0)
        
        if method=="cg":
            if verbose:
                def callback(X):
                    print "objective:",self._eval_XP(X), "||grad||_inf:", np.abs(self._eval_grad_XP(X)).max()
            else:
                callback = None
        
            Xopt = fmin_cg(self._eval_XP, X0, self._eval_grad_XP, callback=callback, **args)
        else:
            Xopt = steepest_descent(self._eval_XP, X0, self._eval_grad_XP, project=self.projectXP, verbose=verbose, **args)
        
        Popt = self.X_to_P(Xopt)
        return Popt
