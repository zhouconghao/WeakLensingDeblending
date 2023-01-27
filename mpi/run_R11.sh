#!/usr/bin/bash
#SBATCH --partition=cpuq
#SBATCH --account=cpuq
#SBATCH --nodes=10
#SBATCH --ntasks=20
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=2

# sample in halomass bins
# R11 vs mass bin jackknife

pwd; hostname; date
echo "Running program on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS total tasks, with each node getting $SLURM_NTASKS_PER_NODE running on cores."

pwd; hostname; date
echo "Running program on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS total tasks, with each node getting $SLURM_NTASKS_PER_NODE running on cores."

source ~/.bashrc
conda activate newmpi
echo "Sucessfully load conda environment."

module purge
module load mpich
module load slurm
which python
cd /data/groups/jeltema/zhou/lsst_shear/WeakLensingDeblending

mpiexec -n 20 python -u /data/groups/jeltema/zhou/lsst_shear/WeakLensingDeblending/mpi/R11.py

echo "Finished job!"