"""
Test gd.py module

The following functions are tested:
$ grep "def " modules/gd.py 
    def dImoldx(self, xyz, fqi, qvector):
    #def d2Imoldxi2(self, xyz, fqi, qvector):
    #def d2Idij(self, xyz, fqi, qvector):
    #def create_d2chi2_matrix():
    #def eigen_(square_array):
    #def gradient_descent_cartesian(
"""

import numpy as np

# my own modules
import modules.mol as mol
import modules.x as xray
import modules.gd as gd

# create class objects
m = mol.Xyz()
x = xray.Xray()
g = gd.G()

#############################
### Initialise some stuff ###
#############################
# qvector
qlen = 241
qvector = np.linspace(1e-9, 24, qlen, endpoint=True)

start_xyz_file = "xyz/test.xyz"
_, _, atomlist, starting_xyz = m.read_xyz(start_xyz_file)
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
print(starting_xyz)

natoms = len(atomlist)
atomic_factor_array = np.zeros((natoms, qlen))  # array of atomic factors
for i in range(natoms):
    atomic_factor_array[i, :] = x.atomic_factor(atomic_numbers[i], qvector)

#################################
### End Initialise some stuff ###
#################################

def test_dImoldx():
    """test dImol/dx function"""

    dImoldx = g.dImoldx(starting_xyz, atomic_factor_array, qvector)
    print(dImoldx.shape)
    print(dImoldx)
    assert dImoldx.shape == (qlen, natoms, 3)
    # more assertations.. I have not proved it actually works correctly!
    # (15 Nov: likely not correct as last row is all zeros always)

#test_dImoldx()

