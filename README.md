# Tight_Binding-Parallel
A simple implementation of the Tight-Binding potential for metallic alloys with a Second Moment Approximation.

Can be compiled with nvcc (requires CUDA 2.0 or higher), for example

nvcc -O3 -D_FORCE_INLINES -lm --ptxas-options=-v -o TBSMA.x TBSMA_energy_4.0.cu

To run, simply do

./TBSMA.x

On a folder with coord_z.xyz file. 3 sample files have been added, just do

cp coord_XXXXX coord_z.xyz
