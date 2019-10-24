import os
import numpy as np
import ctypes as ct

import spec
import wignerd

class gauss_legendre_quadrature:
    # Gauss-Legendre and Wigner-d matrix
    # code from libkms_ist, by Kendrick Smith.
    """
    self.npoints = number of points in quadrature
    self.zvec = list of points in [-1,1]
    self.wvec = integration weights for each point
    """

    def __init__(self, npoints):
        self.npoints = npoints
        self.zvec, self.wvec = wignerd.init_gauss_legendre_quadrature(npoints)

    def cf_from_cl(self, s1, s2, cl):
        "This computes (sum_l C_l d^l_{ss'}), not (sum_l (2l+1)/(4pi) C_l d^l_{ss'})"

        lmax = len(cl)-1

        if np.iscomplexobj(cl):
            #FIXME: convert to 1 cf_from_cl call for potential 2x speed boost.
            return (wignerd.wignerd_cf_from_cl( s1, s2, 1, self.npoints, lmax, self.zvec, cl.real ) +
                    wignerd.wignerd_cf_from_cl( s1, s2, 1, self.npoints, lmax, self.zvec, cl.imag ) * 1.j)
        else:
            return (wignerd.wignerd_cf_from_cl( s1, s2, 1, self.npoints, lmax, self.zvec, cl ))

    def cl_from_cf(self, lmax, s1, s2, cf):
        "This computes (int f d^l_{ss'}), not 2pi (int f d^l_{ss'})"

        if np.iscomplexobj(cf):
            #FIXME: convert to 1 cl_from_cf call for potential 2x speed boost.
            return (wignerd.wignerd_cl_from_cf( s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf.real ) +
                    wignerd.wignerd_cl_from_cf( s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf.imag ) * 1.j)
        else:
            return (wignerd.wignerd_cl_from_cf( s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf ))

def calc_clbb(lmax_b, clee, clpp):
    lmax_e = len(clee) - 1
    lmax_p = len(clee) - 1
    
    assert( lmax_b >= 2 )
    assert( lmax_e >= 2 )
    assert( lmax_p >= 1 )
    assert( lmax_b <= (lmax_e+lmax_p-1) )

    gl = gauss_legendre_quadrature( int((lmax_e + lmax_p + lmax_b)*0.5)+1 )

    ls     = np.arange(0, lmax_e+1, dtype=np.double)
    cl_e33 = (2*ls+1)/(4*np.pi)*clee*(ls-2)*(ls+3)
    cl_e31 = (2*ls+1)/(4*np.pi)*clee*np.sqrt( (ls-1)*(ls+2)*(ls-2)*(ls+3) )
    cl_e11 = (2*ls+1)/(4*np.pi)*clee*(ls-1)*(ls+2)

    ls     = np.arange(0, lmax_p+1, dtype=np.double)
    cl_p11 = (2*ls+1)/(4*np.pi)*ls*(ls+1)*clpp

    zeta_e33_p = gl.cf_from_cl( 3,  3, cl_e33 )
    zeta_e33_m = gl.cf_from_cl( 3, -3, cl_e33 )
    zeta_e31_p = gl.cf_from_cl( 3,  1, cl_e31 )
    zeta_e31_m = gl.cf_from_cl( 3, -1, cl_e31 )
    zeta_e11_p = gl.cf_from_cl( 1,  1, cl_e11 )
    zeta_e11_m = gl.cf_from_cl( 1, -1, cl_e11 )

    zeta_p11_p = gl.cf_from_cl( 1,  1, cl_p11 )
    zeta_p11_m = gl.cf_from_cl( 1, -1, cl_p11 )

    clbb_out_p = gl.cl_from_cf( lmax_b, 2,  2, 
                                    zeta_e33_p*zeta_p11_p + 
                                    2.*zeta_e31_p*zeta_p11_m + 
                                    zeta_e11_p*zeta_p11_p )
    clbb_out_m = gl.cl_from_cf( lmax_b, 2, -2, 
                                    zeta_e33_m*zeta_p11_m + 
                                    2.*zeta_e31_m*zeta_p11_p +
                                    zeta_e11_m*zeta_p11_m )

    return np.pi/4. * (clbb_out_p - clbb_out_m)

def calc_nlpp(lmax_p, clbb, clee, nleb):
    lmax_e = len(clee) - 1
    lmax_b = len(clbb) - 1

    assert( lmax_b >= 2 )
    assert( lmax_e >= 2 )
    assert( lmax_p >= 1 )
    assert( lmax_b <= (lmax_e+lmax_p-1) )
    gl = gauss_legendre_quadrature( int((lmax_e + lmax_p + lmax_b)*0.5)+1 )

    ls = np.arange(0, lmax_e+1, dtype=np.double)
    cl_en = clee**2/(clee+nleb)
    cl_en33 = (2*ls+1)/(4*np.pi)*cl_en*(ls-2)*(ls+3)
    cl_en31 = (2*ls+1)/(4*np.pi)*cl_en*np.sqrt( (ls-1)*(ls+2)*(ls-2)*(ls+3) )
    cl_en11 = (2*ls+1)/(4*np.pi)*cl_en*(ls-1)*(ls+2)

    ls = np.arange(0, lmax_b+1, dtype=np.double)
    cl_bn22 = (2*ls+1)/(4*np.pi)*(1./(clbb+nleb))

    zeta_en33_p = gl.cf_from_cl( 3,  3, cl_en33 )
    zeta_en33_m = gl.cf_from_cl( 3, -3, cl_en33 )
    zeta_en31_p = gl.cf_from_cl( 3, 1, cl_en31 )
    zeta_en31_m = gl.cf_from_cl( 3, -1, cl_en31 )
    zeta_en11_p = gl.cf_from_cl( 1, 1, cl_en11 )
    zeta_en11_m = gl.cf_from_cl( 1, -1, cl_en11 )

    zeta_bn22_p = gl.cf_from_cl( 2, 2, cl_bn22 )
    zeta_bn22_m = gl.cf_from_cl( 2, -2, cl_bn22 )

    nlpp_out_p = gl.cl_from_cf( lmax_p, 1, 1,
                                zeta_en33_p*zeta_bn22_p -
                                2.*zeta_en31_m*zeta_bn22_m +
                                zeta_en11_p*zeta_bn22_p)

    nlpp_out_m =  gl.cl_from_cf( lmax_p, 1, -1,
                                 zeta_en33_m*zeta_bn22_m -
                                 2.*zeta_en31_p*zeta_bn22_p +
                                 zeta_en11_m*zeta_bn22_m)

    return 1./(np.pi/4.*(ls* (ls+1) ) * ( nlpp_out_p - nlpp_out_m ))
