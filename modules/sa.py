import numpy as np
from numpy.random import random_sample as random
from numpy import linalg as LA
# my modules
import modules.mol as mol
import modules.x as xray

# create class objects
m = mol.Xyz()
x = xray.Xray()

#############################
class Annealing:
    """Gradient descent functions"""

    def __init__(self):
        pass

    def read_nm_displacements(self, fname, natoms):
        """read_nm_displacements: Reads displacement vector from file=fname e.g. 'normalmodes.txt'
        Inputs: 	natoms (int), total number of atoms
        Outputs:	displacements, array of displacements, size: (nmodes, natoms, 3)"""
        if natoms == 2:
            nmodes = 1
        elif natoms > 2:
            nmodes = 3 * natoms - 6
        else:
            print("ERROR: natoms. Are there < 2 atoms?")
            return False
        with open(fname, "r") as xyzfile:
            tmp = np.loadtxt(fname)
        displacements = np.zeros((nmodes, natoms, 3))
        for i in range(3 * natoms):
            for j in range(nmodes):
                if i % 3 == 0:  # Indices 0,3,6,...
                    dindex = int(i / 3)
                    displacements[j, dindex, 0] = tmp[i, j]  # x coordinates
                elif (i - 1) % 3 == 0:  # Indices 1,4,7,...
                    displacements[j, dindex, 1] = tmp[i, j]  # y coordinates
                elif (i - 2) % 3 == 0:  # Indices 2,5,8,...
                    displacements[j, dindex, 2] = tmp[i, j]  # z coordinates
        return displacements

    def uniform_factors(self, nmodes, displacement_factors):
        """uniformly random displacement step along each mode"""
        factors = np.zeros(nmodes)
        for j in range(nmodes):
            # random factors in range [-a, a]
            a = displacement_factors[j]
            factors[j] = 2 * a * random.random_sample() - a
        return factors

    def displacements_from_wavenumbers(self, wavenumbers, step_size, exponential=False):
        nmodes = len(wavenumbers)
        displacement_factors = np.zeros(nmodes)
        for i in range(nmodes):  # initial factors are inv. prop. to wavenumber
            if wavenumbers[i] > 0:
                if exponential:
                    displacement_factors[i] = np.exp(wavenumbers[0] / wavenumbers[i])
                else:
                    displacement_factors[i] = wavenumbers[0] / wavenumbers[i]
            else:
                displacement_factors[i] = 0.0
        displacement_factors *= step_size  # adjust max size of displacement step
        return displacement_factors

    def simulate_trajectory(
        self, starting_xyz, displacements, wavenumbers, nsteps, step_size
    ):
        """creates a simulated trajectory by randomly moving along normal modes"""
        natom = starting_xyz.shape[0]
        nmodes = len(wavenumbers)
        modes = list(range(nmodes))
        displacement_factors = self.displacements_from_wavenumbers(
            wavenumbers, step_size
        )
        xyz = starting_xyz  # start at starting xyz
        xyz_traj = np.zeros((natom, 3, nsteps))
        for i in range(nsteps):
            factors = self.uniform_factors(
                nmodes, displacement_factors
            )  # random factors
            xyz = nm.nm_displacer(xyz, displacements, modes, factors)
            xyz_traj[:, :, i] = xyz
        return xyz_traj

    def atomic_pre_molecular(self, atomic_numbers, qvector, aa, bb, cc, electron_mode=False):
        """both parts of IAM equation that don't depend on atom-atom distances"""
        # compton factors for inelastic effect
        compton_array = x.compton_spline(atomic_numbers, qvector)
        natoms = len(atomic_numbers)
        qlen = len(qvector)
        atomic_total = np.zeros(qlen)  # total atomic factor
        atomic_factor_array = np.zeros((natoms, qlen))  # array of atomic factors
        compton = np.zeros(qlen)
        for k in range(natoms):
            compton += compton_array[k, :]
            atomfactor = np.zeros(qlen)
            for j in range(qlen):
                for i in range(4):
                    atomfactor[j] += aa[atomic_numbers[k] - 1, i] * np.exp(
                        -bb[atomic_numbers[k] - 1, i] * (0.25 * qvector[j] / np.pi) ** 2
                    )
            atomfactor += cc[atomic_numbers[k] - 1]
            atomic_factor_array[k, :] = atomfactor
            if electron_mode:
                atomic_total += (atomic_numbers[k] - atomfactor) ** 2
            else:
                atomic_total += atomfactor ** 2
        nij = int(natoms * (natoms - 1) / 2)
        pre_molecular = np.zeros((nij, qlen))
        k = 0
        for i in range(natoms):
            for j in range(i + 1, natoms):
                if electron_mode:
                    pre_molecular[k, :] = np.multiply(
                            (atomic_numbers[i] - atomic_factor_array[i, :]),
                            (atomic_numbers[j] - atomic_factor_array[j, :]),
                        )
                else:
                    pre_molecular[k, :] = np.multiply(
                        atomic_factor_array[i, :], atomic_factor_array[j, :]
                    )

                k += 1
        return compton, atomic_total, pre_molecular

    def simulated_annealing_modes_ho(
        self,
        atomlist,
        starting_xyz,
        reference_xyz,
        displacements,
        mode_indices,
        target_function,
        qvector,
        step_size_array,
        ho_indices,
        starting_temp=0.2,
        nsteps=10000,
        inelastic=True,
        af=1,  # HO factor
        pcd_mode=False,
        q_mode=False,
        electron_mode=False,
    ):
        """simulated annealing minimisation to target_function"""
        ##=#=#=# DEFINITIONS #=#=#=##
        ## start.xyz, reference.xyz ##
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
        compton_array = x.compton_spline(atomic_numbers, qvector)
        reference_iam, _, _, _ = x.iam_calc(
            atomic_numbers, reference_xyz, qvector, electron_mode, inelastic, compton_array
        )
        natoms = starting_xyz.shape[0]  # number of atoms
        nmodes = displacements.shape[0]  # number of displacement vectors
        modes = list(range(nmodes))  # all modes
        ## q-vector, atomic, and pre-molecular IAM contributions ##
        qlen = len(qvector)  # length of q-vector
        aa, bb, cc = x.read_iam_coeffs()
        compton, atomic_total, pre_molecular = self.atomic_pre_molecular(
            atomic_numbers, qvector, aa, bb, cc, electron_mode,
        )
        ##=#=#=# END DEFINITIONS #=#=#=#

        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##
        nho_indices = len(ho_indices)  # number of HO indices
        r0_arr = np.zeros(nho_indices)  # array of starting xyz bond-lengths
        for i in range(nho_indices):
            r0_arr[i] = np.linalg.norm(
                starting_xyz[ho_indices[0][i], :] - starting_xyz[ho_indices[1][i], :]
            )

        total_harmonic_contrib = 0
        total_xray_contrib = 0
        ##=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=##

        ##=#=#=# INITIATE LOOP VARIABLES #=#=#=#=#
        xyz = starting_xyz
        i, c = 0, 0
        chi2, chi2_best = 1e9, 1e10
        chi2_array = np.zeros(nsteps)
        # mdisp = displacements * step_size  # array of molecular displacements
        mdisp = displacements
        ##=#=#=# END INITIATE LOOP VARIABLES #=#=#
        while i < nsteps:
            i += 1  # count steps
            # print(i)

            ##=#=#=#=# TEMPERATURE #=#=#=#=#=#=#=#=##
            tmp = 1 - i / nsteps  # this is prop. to how far the molecule moves
            temp = starting_temp * tmp  # this is the probability of going uphill
            ##=#=#=# END TEMPERATURE #=#=#=#=#=#=#=##

            ##=#=#=# DISPLACE XYZ RANDOMLY ALONG ALL DISPLACEMENT VECTORS #=#=#=##
            summed_displacement = np.zeros(mdisp[0, :, :].shape)
            for n in mode_indices:
                summed_displacement += (
                    mdisp[n, :, :] * step_size_array[n] * tmp * (2 * random() - 1)
                )
            xyz_ = xyz + summed_displacement  # save a temporary displaced xyz: xyz_
            ##=#=#=# END DISPLACE XYZ RANDOMLY ALONG ALL DISPLACEMENT VECTORS #=#=#=##

            ##=#=#=# IAM CALCULATION #=#=#=##
            ## this takes 84% of the run time ... ##
            ## can it be optimised further? ##
            molecular = np.zeros(qlen)  # total molecular factor
            k = 0
            for ii in range(natoms):
                for jj in range(ii + 1, natoms):  # j > i
                    qdij = qvector * LA.norm(xyz_[ii, :] - xyz_[jj, :])
                    molecular += pre_molecular[k, :] * np.sin(qdij) / qdij
                    k += 1
            iam_ = atomic_total + 2 * molecular
            if inelastic:
                iam_ += compton
            ##=#=#=# END IAM CALCULATION #=#=#=##

            ##=#=#=# PCD & CHI2 CALCULATIONS #=#=#=##
            if pcd_mode:
                predicted_function_ = 100 * (iam_ / reference_iam - 1)
            elif q_mode:
                predicted_function_ = iam_ * qvector ** 1.0
            else:
                predicted_function_ = iam_

            ### x-ray part of chi2
            xray_contrib = (
                #np.sum((predicted_function_ - target_function) ** 2 / target_function) / qlen
                np.sum((predicted_function_ - target_function) ** 2) / qlen
            )
            ### harmonic oscillator part of chi2
            harmonic_contrib = 0
            for iho in range(nho_indices):
                r = LA.norm(xyz_[ho_indices[0][iho], :] - xyz_[ho_indices[1][iho], :])
                harmonic_contrib += af * (r - r0_arr[iho]) ** 2

            ### combine x-ray and harmonic contributions
            chi2_ = xray_contrib + harmonic_contrib
            ##=#=#=# END PCD & CHI2 CALCULATIONS #=#=#=##

            ##=#=#=# ACCEPTANCE CRITERIA #=#=#=##
            if chi2_ < chi2 or temp > random():
                c += 1  # count acceptances
                chi2, xyz = chi2_, xyz_  # update chi2 and xyz
                # save chi2 to graph
                chi2_array[c - 1] = chi2
                if chi2 < chi2_best:
                    # store values corresponding to chi2_best
                    chi2_best, xyz_best, predicted_best = chi2, xyz, predicted_function_
                    chi2_xray_best = xray_contrib
                total_harmonic_contrib += harmonic_contrib
                total_xray_contrib += xray_contrib
            ##=#=#=# END ACCEPTANCE CRITERIA #=#=#=##
        # remove ending zeros from chi2_array
        chi2_array = chi2_array[:c]
        # print ratio of contributions to chi2
        total_contrib = total_xray_contrib + total_harmonic_contrib
        xray_ratio = total_xray_contrib / total_contrib
        harmonic_ratio = total_harmonic_contrib / total_contrib
        print("xray contrib ratio: %f" % xray_ratio)
        print("harmonic contrib ratio: %f" % harmonic_ratio)
        # end function
        return chi2_best, predicted_best, xyz_best, chi2_array, chi2_xray_best

    
