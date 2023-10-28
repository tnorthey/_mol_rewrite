"""
Test functions for mol.py module
"""

import numpy as np
import os

# my own modules
import modules.mol as mol
import modules.x as xray

# create class objects
m = mol.Xyz()
x = xray.Xray()

# read test.xyz (perfectly linear H-O-H with exactly 1 Angstrom OH bonds)
xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/test.xyz")
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]

print(xyz)

# qvector
qlen = 241
qvector = np.linspace(0.1, 24, qlen, endpoint=True)

def test_read_xyz():
    assert xyzheader == 3, "xyzheader should be 3"
    assert comment.__contains__("test"), "comment should be 'test'"
    assert atomlist[0] == "O", "1st atom should be O"
    assert atomic_numbers[0] == 8, "1st atomic charge should be 8"
    assert xyz[0, 0] == 0.0, "Upper left coordinate should be 0.0"
#test_read_xyz()


def test_write_xyz():
    fname = "out.xyz"
    comment = "test"
    m.write_xyz(fname, comment, atomlist, xyz)
    with open(fname) as out:
        assert out.readline() == "3\n", "1st line of out.xyz != 3"
        assert out.readline() == "test\n", "2nd line of out.xyz != 'test'"
    os.remove(fname)
#test_write_xyz()

def test_distances_array():
    dist_array = m.distances_array(xyz)
    assert dist_array[1, 2] == 2, "distance between hydrogens != 2"
#test_distances_array()

def test_periodic_table():
    h = m.periodic_table("H")
    he = m.periodic_table("He")
    c = m.periodic_table("C")
    assert h == 1, "H should have atom number 1"
    assert he == 2, "He should have atom number 2"
    assert c == 6, "C should have atom number 2"
#test_periodic_table()


def test_atomic_factor():
    atom_number = 6  # atom_number = 1 is hydrogen, etc.
    atom_factor = x.atomic_factor(atom_number, qvector)
    #assert round(atom_factor[0], 3) == 1.0, "H  atomic factor (q = 0) != 1"
    #assert (
    #    round(x.atomic_factor(2, qvector)[0], 3) == 2.0
    #), "He atomic factor (q = 0) != 2"
    #np.savetxt('atomic_factor.dat', np.column_stack((qvector, atom_factor)))
test_atomic_factor()

def test_iam_calc_xray():
    compton_array = x.compton_spline(atomic_numbers, qvector)  # compton factors
    inelastic = True
    iam, atomic, molecular, compton = x.iam_calc_xray(atomic_numbers, xyz, qvector, inelastic, compton_array)
    #np.savetxt('iam.dat', np.column_stack((qvector, iam)))
    assert round(iam[0], 0) == 100.0, "H2O molecular factor (q = 0) != 100"
    assert round(iam[-1], 0) == 10.0, "H2O molecular factor (q = 24) != 10"
#test_iam_calc()


