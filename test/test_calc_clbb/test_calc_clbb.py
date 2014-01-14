import os, sys
import numpy as np

import pylab as pl

sys.path.append("../../")
import cmblensing

# init
lmax = 4096

cl_unl = cmblensing.spec.camb_clfile("inputs/wmap7_lcdm/bestfit_scalCls.dat", lmax=lmax)
cl_len = cmblensing.spec.camb_clfile("inputs/wmap7_lcdm/bestfit_lensedCls.dat", lmax=lmax)

# calculate clbb
clbb   = cmblensing.calc_clbb(lmax, cl_unl.clee, cl_unl.clpp)

# make plots
ls = np.arange(0, lmax+1, dtype=np.double)

pl.subplot(211)
pl.plot(ls, 1.e4*ls*cl_len.clbb, color='k', label=r'$C_l^{BB, {\rm CAMB}}$')
pl.plot(ls, 1.e4*ls*clbb, color='m', ls='--', label=r'$C_l^{BB}$')
pl.ylabel(r'$10^4 l C_l^{BB}$  $[\mu K^2]$')
pl.legend(loc='upper right')
pl.setp(pl.gca().get_legend().get_frame(), visible=False)

pl.subplot(212)
pl.plot( ls, 100.*(clbb / cl_len.clbb - 1.), color='k' )
pl.ylabel(r'$100 (C_l^{BB} / C_l^{BB, {\rm CAMB}}-1)$  $[\%]$')
pl.ylim(-5., 5.)

pl.ion()
pl.show()
