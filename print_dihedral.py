import os
import sys
import numpy as np

# my modules
import modules.mol as mol

# create class objects
m = mol.Xyz()

xyz_fullpath = sys.argv[1] 
file_name = os.path.basename(xyz_fullpath)
ftarg = file_name[5:15]

xyzheader, comment, atomlist, xyz = m.read_xyz(xyz_fullpath)
p0 = np.array(xyz[0, :])
p1 = np.array(xyz[1, :])
p4 = np.array(xyz[4, :])
p5 = np.array(xyz[5, :])
dihedral = m.new_dihedral(np.array([p0, p1, p4, p5]))
#print('%s %9.8f' % (ftarg, dihedral))
print('%9.8f' % dihedral)

