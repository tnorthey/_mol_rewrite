"""
Test functions for x.py module

The following are tested:
$ grep "def " modules/x.py 
  def __init__(self):
  def read_iam_coeffs(self):
  def atomic_factor(self, atom_number, qvector):
  def compton_spline(self, atomic_numbers, qvector):
  def iam_calc_xray(
  def iam_calc_electron(
  def iam_calc_2d(self, atomic_numbers, xyz, qvector):
  def jq_atomic_factors_calc(self, atomic_numbers, qvector):
  def compton_spline_calc(self, atomic_numbers, qvector):
  def Imol_calc(self, atomic_factor_arr, xyz, qvector):
"""

import numpy as np
import os

# my own modules
import modules.mol as mol
import modules.x as xray

# create class objects
m = mol.Xyz()
x = xray.Xray()

#############################
### Initialise some stuff ###
#############################
# read test.xyz (perfectly linear H-O-H with exactly 1 Angstrom OH bonds)
xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/test.xyz")
natoms = len(atomlist)
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]

# qvector
qlen = 241
qvector = np.linspace(1e-9, 24, qlen, endpoint=True)
#################################
### End Initialise some stuff ###
#################################

def test_read_iam_coeffs():
    aa, bb, cc = x.read_iam_coeffs()
    # check it reads H coeffs correctly
    assert aa[0, 0] == 0.489918 
    assert bb[0, 0] == 20.6593 
    assert cc[0] == 0.001305

def test_compton_spline():
    compton_array = x.compton_spline(atomic_numbers, qvector)
    # test if splined compton factor == 0 for oxygen at q = 0
    assert round(compton_array[0, 0], 1) == 0, "splined compton factor != 0 at q = 0"
    # test if splined compton factor == 1 for one of the hydrogens at q = 24
    assert round(compton_array[1, -1], 1) == 1, "splined compton factor != 1 for H at q = 24"

def test_atomic_factor():
    atom_number = 6  # atom_number = 1 is hydrogen, etc.
    atom_factor = x.atomic_factor(atom_number, qvector)
    # test if C atomic scattering factor f(q=0) = Nel = 6  (within rounding)
    assert round(atom_factor[0], 2) == 6.0, "C  atomic factor (q = 0) != 6"

# I use this for the IAM calc x-ray and electron tests
compton_array = x.compton_spline(atomic_numbers, qvector)  # compton factors
inelastic = True

def test_iam_calc_xray():
    iam, atomic, molecular, compton = x.iam_calc_xray(atomic_numbers, xyz, qvector, inelastic, compton_array)
    # test if H2O I(q=0) = Nel**2 = 10**2  (within rounding)
    assert round(iam[0], 0) == 100.0, "H2O I_total(q = 0) != 100"
    # test if H2O I(q=24)_inelastic = Nel = 10  (within rounding)
    assert round(compton[-1], 0) == 10.0, "H2O inelastic scattering term (q = 24) != 10"

def test_iam_calc_electron():
    iam, atomic, molecular, compton = x.iam_calc_electron(atomic_numbers, xyz, qvector, inelastic, compton_array)
    # test if H2O I(q=0) = 0  (within rounding)
    assert round(iam[0], 0) == 0.0, "H2O I_total(q = 0) != 0"
    # test if H2O I(q=24)_inelastic = Nel = 10  (within rounding)
    #assert round(compton[-1], 0) == 10.0, "H2O inelastic scattering term (q = 24) != 10"

