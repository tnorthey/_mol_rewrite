"""
Test functions for mol.py module.

Test for the following functions:
$ grep "def " ../modules/mol.py 
  def __init__(self):
  def periodic_table(self, element):
  def atomic_mass(self, element):
  def read_xyz(self, fname):
  def write_xyz(self, fname, comment, atoms, xyz):
  def read_xyz_traj(self, fname, ntsteps):
  def write_xyz_traj(self, fname, atoms, xyz_traj):
  def distances_array(self, xyz):

Not tested yet:
  def rmsd_atoms(self, xyz, xyz_, indices):
  def rmsd_kabsch(self, xyz, xyz_, indices):
  def mapd_function(self, xyz, xyz_, indices, bond_print):
  def mapd_distances(self, rij, rij_, bond_print):
"""

import numpy as np
import os
# my own modules
import modules.mol as mol

# create class objects
m = mol.Xyz()


def test_periodic_table():
    h = m.periodic_table("H")
    he = m.periodic_table("He")
    c = m.periodic_table("C")
    assert h == 1, "H should have atom number 1"
    assert he == 2, "He should have atom number 2"
    assert c == 6, "C should have atom number 6"

def test_atomic_mass():
    """
    H        1.0079
    He       4.0026
    C        12.0107
    """
    h = m.atomic_mass("H")
    he = m.atomic_mass("He")
    c = m.atomic_mass("C")
    assert h == 1.0079, "H"
    assert he == 4.0026, "He"
    assert c == 12.0107, "C"

# read test.xyz (perfectly linear H-O-H with exactly 1 Angstrom OH bonds)
xyzheader, comment, atomlist, xyz = m.read_xyz("xyz/test.xyz")
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]

def test_read_xyz():
    assert xyzheader == 3, "xyzheader should be 3"
    assert comment.__contains__("test"), "comment should be 'test'"
    assert atomlist[0] == "O", "1st atom should be O"
    assert atomic_numbers[0] == 8, "1st atomic charge should be 8"
    assert xyz[0, 0] == 0.0, "Upper left coordinate should be 0.0"

def test_write_xyz():
    fname = "out.xyz"
    comment = "test"
    m.write_xyz(fname, comment, atomlist, xyz)
    with open(fname) as out:
        assert out.readline() == "3\n", "1st line of out.xyz != 3"
        assert out.readline() == "test\n", "2nd line of out.xyz != 'test'"
    os.remove(fname)  # delete the file

def test_distances_array():
    dist_array = m.distances_array(xyz)
    assert dist_array[1, 2] == 2, "distance between hydrogens != 2"

