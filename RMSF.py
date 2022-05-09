"""
SAME AS:

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.tests.datafiles import GRO, XTC, TPR

u = mda.Universe(GRO, XTC)
average = align.AverageStructure(u, u, select='protein and name CA', ref_frame=0, verbose = True).run()
ref = average.results.universe

aligner = align.AlignTraj(u, ref, select='protein and name CA', in_memory=True,  verbose = True).run()

c_alphas = u.select_atoms('protein and name CA')
R = rms.RMSF(c_alphas).run()

Reference: https://www.mdanalysis.org
"""

import os
#otherise numpy uses more than one core
#change before loading numpy
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

import MDAnalysis as mda
from MDAnalysis.analysis import rms, base
from mpi4py import MPI
import numpy as np
from itertools import chain
import functools 
import MDAnalysis.lib.qcprot as qcp
from MDAnalysis.tests.datafiles import GRO, XTC

def second_order_moments(S1, S2):
    T = S1[0] + S2[0]
    mu = (S1[0]*S1[1] + S2[0]*S2[1])/T
    M = S1[2] + S2[2] + (S1[0] * S2[0]/T) * (S2[1] - S1[1])**2
    S = T, mu, M
    return S

def get_rotation_matrix(ref_coordinates, mobile_coordinates, n_atoms):

    rotation_matrix = np.zeros(9, dtype=np.float64)
    reshaped_matrix = rotation_matrix.reshape(3, 3)

    qcp.CalcRMSDRotationalMatrix(ref_coordinates, mobile_coordinates, n_atoms, rotation_matrix, weights=None)
    reshaped_matrix[:, :] = rotation_matrix.reshape(3, 3)

    return reshaped_matrix

if __name__ == '__main__':
   
    #define variables
    universe = mda.Universe(GRO, XTC)
    reference = universe.copy() 

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
 
    ref_frame = 0 

    n_frames = universe.trajectory.n_frames
    n_blocks = size
    n_frames_per_block = n_frames // n_blocks
    blocks = [range(i * n_frames_per_block, (i + 1) * n_frames_per_block) for i in range(n_blocks-1)]
    blocks.append(range((n_blocks - 1) * n_frames_per_block, n_frames))

    start = blocks[rank].start
    stop = blocks[rank].stop

    print("Process:%3d --> Frames: %10d -- %10d" % (rank, start, stop))

    #calculate average structure
    mobile_atoms = universe.select_atoms("protein and name CA")
    ref_atoms = reference.select_atoms("protein and name CA")
    
    current_frame = reference.universe.trajectory.ts.frame
    
    try:
        reference.universe.trajectory[ref_frame]
        ref_com = ref_atoms.center_of_mass().astype(np.float64)
        ref_coordinates = ref_atoms.positions.astype(np.float64) - ref_com
    finally:
        reference.universe.trajectory[current_frame]        

    pos = np.zeros((universe.trajectory.n_atoms, 3))

    for i, frame in enumerate(range(start, stop)):
        ts = universe.trajectory[frame]
        
        mobile_com = mobile_atoms.center_of_mass().astype(np.float64)
        mobile_coordinates = mobile_atoms.positions.astype(np.float64) - mobile_com

        reshaped_matrix = get_rotation_matrix(ref_coordinates, mobile_coordinates, mobile_atoms.n_atoms)

        ts.positions[:] -= mobile_com
        ts.positions[:] = np.dot(ts.positions, reshaped_matrix)
        ts.positions += ref_com

        pos += ts.positions

    pos = pos.flatten()

    comm.Barrier()

    #calculate RMSF
    positions = np.zeros(universe.atoms.n_atoms * 3)
    comm.Allreduce(sendbuf = pos, recvbuf = positions, op = MPI.SUM)
    positions = positions / float(n_frames)
    
    reference = mda.Universe(GRO, positions.reshape((1, -1, 3)))
    
    ref_atoms = reference.select_atoms("protein and name CA")
    ref_com = ref_atoms.center_of_mass().astype(np.float64)
    ref_coordinates = ref_atoms.positions.astype(np.float64) - ref_com

    sumsquares = np.zeros((universe.select_atoms("protein and name CA").n_atoms, 3))
    mean = sumsquares.copy()

    for k, frame in enumerate(range(start, stop)):
        ts = universe.trajectory[frame]
    
        mobile_atoms = universe.select_atoms("protein and name CA")
        mobile_com = mobile_atoms.center_of_mass().astype(np.float64)
        mobile_coordinates = mobile_atoms.positions.astype(np.float64) - mobile_com
    

        reshaped_matrix = get_rotation_matrix(ref_coordinates, mobile_coordinates, mobile_atoms.n_atoms) 

        ts.positions[:] -= mobile_com
        ts.positions[:] = np.dot(ts.positions, reshaped_matrix)
        ts.positions += ref_com
    
        sumsquares += (k / (k + 1.0)) * (universe.select_atoms("protein and name CA").positions.astype(np.float64) - mean) ** 2
        mean = (k * mean + universe.select_atoms("protein and name CA").positions.astype(np.float64)) / (k + 1)    

    S = [stop - start, mean, sumsquares]
    comm.Barrier()
    MPI.Op.Create(second_order_moments, commute=True)
    Data = comm.reduce(S, root = 0, op = second_order_moments)    
 
    if rank == 0:
        RMSF = np.sqrt(Data[2].sum(axis = 1) / Data[0])
        #Do something with RMSF
    
    exit(0)
