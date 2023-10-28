import numpy as np

######
class Xyz:
    """methods to manipulate molecular coordinates (xyz)"""

    def __init__(self):
        pass

    def periodic_table(self, element):
        """Outputs atomic number for each element in the periodic table"""
        with open("data/pt.txt") as pt_file:
            for line in pt_file:
                if line.split()[0] == element:
                    return int(line.split()[1])

    def atomic_mass(self, element):
        """Outputs atomic mass for each element in the periodic table"""
        with open("data/atomic_masses.txt") as am_file:
            for line in am_file:
                if line.split()[0] == element:
                    return line.split()[1]

    # read/write xyz files

    def read_xyz(self, fname):
        """Read a .xyz file"""
        with open(fname, "r") as xyzfile:
            xyzheader = int(xyzfile.readline())
            comment = xyzfile.readline()
        xyzmatrix = np.loadtxt(fname, skiprows=2, usecols=[1, 2, 3])
        atomarray = np.loadtxt(fname, skiprows=2, dtype=str, usecols=[0])
        return xyzheader, comment, atomarray, xyzmatrix

    def write_xyz(self, fname, comment, atoms, xyz):
        """Write .xyz file"""
        natom = len(atoms)
        xyz = xyz.astype("|S10")  # convert to string array (max length 10)
        atoms_xyz = np.append(np.transpose([atoms]), xyz, axis=1)
        np.savetxt(
            fname,
            atoms_xyz,
            fmt="%s",
            delimiter=" ",
            header=str(natom) + "\n" + comment,
            footer="",
            comments="",
        )
        return

    def read_xyz_traj(self, fname, ntsteps):
        """Read a .xyz trajectory file"""
        with open(fname, "r") as xyzfile:
            natoms = int(xyzfile.readline())
            comment = xyzfile.readline()
            xyztraj = np.zeros((natoms, 3, ntsteps))
            atomarray = []
            for line in range(natoms):
                atomarray.append(xyzfile.readline().split()[0])
                print(atomarray)
        with open(fname, "r") as xyzfile:
            for t in range(ntsteps):
                print("read_xyz_traj: reading frame: %i" % t)
                natoms = int(xyzfile.readline())
                comment = xyzfile.readline()
                for line in range(natoms):
                    xyztraj[line, :, t] = xyzfile.readline().split()[1:4]
        return natoms, comment, atomarray, xyztraj

    def write_xyz_traj(self, fname, atoms, xyz_traj):
        """converts xyz_traj array to traj.xyz"""
        natom = len(atoms)
        atoms_xyz_traj = np.empty((1, 4))
        for t in range(xyz_traj.shape[2]):
            comment = "iteration: %i" % t
            xyz = xyz_traj[:, :, t]
            xyz = xyz.astype("|S14")  # convert to string array (max length 14)
            tmp = np.array([[str(natom), "", "", ""], [comment, "", "", ""]])
            atoms_xyz = np.append(np.transpose([atoms]), xyz, axis=1)
            atoms_xyz = np.append(tmp, atoms_xyz, axis=0)
            atoms_xyz_traj = np.append(atoms_xyz_traj, atoms_xyz, axis=0)
        atoms_xyz_traj = atoms_xyz_traj[1:, :]  # remove 1st line of array
        print("writing %s..." % fname)
        np.savetxt(
            fname,
            atoms_xyz_traj,
            fmt="%s",
            delimiter=" ",
            header="",
            footer="",
            comments="",
        )
        return

    ### distances array

    def distances_array(self, xyz):
        """Computes matrix of distances from xyz"""
        natom = xyz.shape[0]  # number of atoms
        dist_array = np.zeros((natom, natom))  # the array of distances
        for i in range(natom):
            dist_array[i, i] = 0
            for j in range(i + 1, natom):
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                dist_array[i, j] = dist
                dist_array[j, i] = dist  # opposite elements are equal
        return dist_array

    def rmsd_atoms(self, xyz, xyz_, indices):
        """RMSD between xyz and xyz_ for atom indices"""
        natoms = len(indices)
        rmsd = 0.0
        for i in range(natoms):
            rmsd += np.sum((xyz[indices[i], :] - xyz_[indices[i], :]) ** 2)
        rmsd = (rmsd / natoms) ** 0.5
        return rmsd

    def rmsd_kabsch(self, xyz, xyz_, indices):
        """RMSD between xyz and xyz_ for atom indices"""
        # first rotate xyz to have max coincidence with xyz_
        estimated_rotation, rmsd_ = R.align_vectors(xyz[indices, :], xyz_[indices, :])
        xyz_rotated = np.dot(xyz, estimated_rotation.as_matrix())
        # xyz_rotated = xyz
        rmsd = np.sqrt(
            ((((xyz_rotated[indices, :] - xyz_[indices, :]) ** 2)) * 3).mean()
        )
        return rmsd, estimated_rotation

    def mapd_function(self, xyz, xyz_, indices, bond_print):
        """calculate MAPD as defined in Yong et al. Faraday Disc. (2021)"""
        # MAPD is calculated between structures xyz and xyz_
        # indices: calculates MAPD for specified atomic indices
        nind = len(indices)
        mapd = 0
        for i in range(nind):
            for j in range(i + 1, nind):
                rij = np.linalg.norm(xyz[indices[i], :] - xyz[indices[j], :])
                rij_ = np.linalg.norm(xyz_[indices[i], :] - xyz_[indices[j], :])
                delta = np.abs(rij - rij_)
                if bond_print:
                    print("r_{%i, %i}^' : %10.9f" % (i, j, rij))
                    print("r_{%i, %i}^0 : %10.9f" % (i, j, rij_))
                    print("  |\delta_{%i, %i}|  : %10.9f" % (i, j, delta))
                mapd += delta / rij_
        mapd *= 100 / (nind * (nind - 1) / 2)
        return mapd

    def mapd_distances(self, rij, rij_, bond_print):
        """calculate MAPD as defined in Yong et al. Faraday Disc. (2021)"""
        # MAPD is calculated between distance arrays rij, rij_
        nind = rij.shape[0]
        mapd = 0
        for i in range(nind):
            for j in range(i + 1, nind):
                r1 = rij[i, j]
                r0 = rij_[i, j]
                delta = np.abs(r1 - r0)
                if bond_print:
                    print("r_{%i, %i}^' : %10.9f" % (i, j, r1))
                    print("r_{%i, %i}^0 : %10.9f" % (i, j, r0))
                    print("  |\delta_{%i, %i}|  : %10.9f" % (i, j, delta))
                mapd += delta / r0
        mapd *= 100 / (nind * (nind - 1) / 2)
        return mapd

### End Molecule class section

