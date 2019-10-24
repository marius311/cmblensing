import os, copy
import numpy as np
import pylab as pl

class camb_clfile(object):
    # loader for standard CAMB _scalCls and _lensedCls.dat files.
    def __init__(self, tfname, lmax=None):
        tarray = np.loadtxt(tfname)
        lmin   = int(tarray[0, 0])
        assert(lmin in [1,2])

        if lmax == None:
            lmax = np.shape(tarray)[0]+1
            assert(tarray[-1, 0] == lmax)
        assert( (np.shape(tarray)[0]+1) >= lmax )

        ncol = np.shape(tarray)[1]
        ell  = np.arange(lmin, lmax+1, dtype=np.float)

        self.lmax = lmax
        self.ls   = np.concatenate( [ np.arange(0, lmin), ell ] )
        if ncol == 5:
            self.cltt = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),1]*2.*np.pi/ell/(ell+1.)       ] )
            self.clee = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),2]*2.*np.pi/ell/(ell+1.)        ] )
            self.clbb = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),3]*2.*np.pi/ell/(ell+1.)        ] )
            self.clte = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),4]*2.*np.pi/ell/(ell+1.)        ] )

        elif ncol == 6:
            tcmb  = 2.726*1e6 #uK

            self.cltt = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),1]*2.*np.pi/ell/(ell+1.)       ] )
            self.clee = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),2]*2.*np.pi/ell/(ell+1.)       ] )
            self.clte = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),3]*2.*np.pi/ell/(ell+1.)       ] )
            self.clpp = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),4]/ell**4/tcmb**2     ] )
            self.cltp = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),5]/ell**3/tcmb ] )

    def copy(self, lmax=None):
        if (lmax == None):
            return copy.deepcopy(self)
        else:
            assert( lmax <= self.lmax )
            ret      = copy.deepcopy(self)
            ret.lmax = lmax
            ret.ls   = np.arange(0, lmax+1)
            for k, v in self.__dict__.items():
                if k[0:2] == 'cl':
                    setattr( ret, k, copy.deepcopy(v[0:lmax+1]) )
            return ret

    def plot(self, spec='cl', p=pl.plot, t=lambda l:1., **kwargs):
        p( self.ls, t(self.ls) * getattr(self, spec), **kwargs )
