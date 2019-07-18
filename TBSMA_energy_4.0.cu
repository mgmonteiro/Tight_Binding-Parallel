#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<cuda.h>
#include<time.h>
//#include<limits.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
                ________________________
                |                       |_____    __
                |  TBSMA-Energy vs4.1   |     |__|  |_________
                |________________       |     |::|  |        /
   /\__/\       |                \._____|_____|::|__|      <
  ( o_o  )_     |                             \::/  \._______\
   (u--u   \_)  |
    (||___   )==\
  ,dP"/b/=( /P"/b\
  |8 || 8\=== || 8
  `b,  ,P  `b,  ,P
    """`     """`

**************************************************************************************************
Author: Maxwel Gama Monteiro Junior
Contact: maxweljr@gmail.com

Description: Obtains the cohesive energy for a system of N elements (.xyz coordinate file format). 
Uses the Second Moment approximation of the Tight Binding scheme (spherical coordinates!).
**************************************************************************************************

*/



//Kernels = functions from the GPU -> prefixes __global__ or __device__, with device being the GPU itself whenever it is mentioned, and "host" being the CPU
__constant__ double neighbor_fives = 2.60;

//Double approximated reciprocal square root function (drsqrt) - DISCLAIMER use at your own risk. Uses half the cycles of original sqrt() function.

    __device__ __forceinline__ double drsqrt (double a)
    {
      double y, h, l, e;
      unsigned int ilo, ihi, g, f;
      int d;

      ihi = __double2hiint(a);
      ilo = __double2loint(a);
      if (((unsigned int)ihi) - 0x00100000U < 0x7fe00000U){
        f = ihi | 0x3fe00000;
        g = f & 0x3fffffff;
        d = g - ihi;
        a = __hiloint2double(g, ilo); 
        y = rsqrt (a);
        h = __dmul_rn (y, y);
        l = __fma_rn (y, y, -h);
        e = __fma_rn (l, -a, __fma_rn (h, -a, 1.0));
        /* Round as shown in Peter Markstein, "IA-64 and Elementary Functions" */
        y = __fma_rn (__fma_rn (0.375, e, 0.5), e * y, y);
        d = d >> 1;
        a = __hiloint2double(__double2hiint(y) + d, __double2loint(y));
      } else if (a == 0.0) {
        a = __hiloint2double ((ihi & 0x80000000) | 0x7ff00000, 0x00000000);
      } else if (a < 0.0) {
        a = __hiloint2double (0xfff80000, 0x00000000);
      } else if (isinf (a)) {
        a = __hiloint2double (ihi & 0x80000000, 0x00000000);
      } else if (isnan (a)) {
        a = a + a;
      } else {
        a = a * __hiloint2double (0x7fd00000, 0);
        y = rsqrt (a);
        h = __dmul_rn (y, y);
        l = __fma_rn (y, y, -h);
        e = __fma_rn (l, -a, __fma_rn (h, -a, 1.0));
        /* Round as shown in Peter Markstein, "IA-64 and Elementary Functions" */
        y = __fma_rn (__fma_rn (0.375, e, 0.5), e * y, y);
        a = __hiloint2double(__double2hiint(y) + 0x1ff00000,__double2loint(y));
      }
      return a;
    }

//Energy kernel

__global__ void potential_energy(int atoms, double *x_, double *y_, double *z_, double *a0_, double *r0_, double *a_, double *mass_, double *qui_ , double *q_, double *p_, int *lbl, double *epot1, double *epot2)

{

int n = threadIdx.x + blockIdx.x * blockDim.x;
double temp1_ = 0.0;
double temp2_ = 0.0;
extern __shared__ double cache2[];
extern __shared__ double cache1[];


while(n < atoms)
{


double xx_;
double yy_;
double zz_;
double expa_;
double expr_;
double atrac_ = 0.0;
double repul_ = 0.0;
double r_;
double qui2_;
double rij_;
double rcut_;
double x = x_[n];
double y = y_[n];
double z = z_[n];
int k = 0;


	for(int counter = n + 1; counter < atoms; counter++)
	{

if (lbl[n]  ==  lbl[counter])
{

if((lbl[n] == 28) && (lbl[counter] == 28)) k = 0; 	//Nickel
if((lbl[n] == 29) && (lbl[counter] == 29)) k = 1; 	//Copper
if((lbl[n] == 45) && (lbl[counter] == 45)) k = 2; 	//Rhodium
if((lbl[n] == 46) && (lbl[counter] == 46)) k = 3; 	//Palladium
if((lbl[n] == 47) && (lbl[counter] == 47)) k = 4; 	//Silver
if((lbl[n] == 77) && (lbl[counter] == 77)) k = 5; 	//Iridium
if((lbl[n] == 78) && (lbl[counter] == 78)) k = 6; 	//Platinum
if((lbl[n] == 79) && (lbl[counter] == 79)) k = 7; 	//Gold
if((lbl[n] == 13) && (lbl[counter] == 13)) k = 8; 	//Aluminium
if((lbl[n] == 82) && (lbl[counter] == 82)) k = 9;	//Lead

}

if (lbl[n] != lbl[counter])
{

if((lbl[n] == 79) && (lbl[counter] == 29)) k = 10;	//Au - Cu
if((lbl[n] == 29) && (lbl[counter] == 79)) k = 10;

if((lbl[n] == 28) && (lbl[counter] == 13)) k = 11;	//Ni - Al
if((lbl[n] == 13) && (lbl[counter] == 28)) k = 11;

if((lbl[n] == 79) && (lbl[counter] == 47)) k = 12;	//Au - Ag
if((lbl[n] == 47) && (lbl[counter] == 79)) k = 12;

}


//Pure gold mode

//k = 7;
	

	rcut_= neighbor_fives * r0_[k];
		
		xx_ = (x - x_[counter]);
		yy_ = (y - y_[counter])*(y - y_[counter]);
		zz_ = (z - z_[counter])*(z - z_[counter]);
	
//	rij_ = __fma_rn(xx_,xx_,yy_);
//	rij_ = __dadd_rn(rij_, zz_);
	  
//	rij_ = drsqrt(rij_);

	rij_ = drsqrt(xx_*xx_ + yy_ + zz_);

	if (1.0/rij_ < rcut_)
	
	{
	
		r_ = (1.0/rij_)/r0_[k];
	//	r_ = __fma_rn(1.0/rij_,1.0/r0_[k],-1.0); 	
		r_ = r_ - 1.0;

	//Attractive + repulsive term
	
	expr_ = exp( - p_[k] * r_ );
	expa_ = exp( -2.000 * q_[k] * r_);

	repul_ = repul_ + (a_[k] * expr_);
	qui2_ = qui_[k] * qui_[k];
	atrac_ = atrac_ + (qui2_ * expa_);
	//atrac_ = __fma_rn(qui2_,expa_, atrac_);	

	}

	
	}

temp1_ += repul_; 
temp2_ -= 1.0/drsqrt(atrac_);
n += blockDim.x * gridDim.x;

}

cache1[threadIdx.x] = temp1_;
cache2[threadIdx.x] = temp2_;

__syncthreads();



int u = blockDim.x/2;

while(u != 0)
	{
		if (threadIdx.x < u)
			{
			cache1[threadIdx.x] += cache1[threadIdx.x + u];
			cache2[threadIdx.x] += cache2[threadIdx.x + u];
			}
	__syncthreads();
	u /= 2;
	}

if (threadIdx.x == 0){
 epot2[blockIdx.x] = cache1[0];
 epot1[blockIdx.x] = cache2[0];
}

}
//#####################################PROGRAM STARTS HERE##################################

//
// fn = Distance among nearest neighbors
// rij = Distance between atoms
// rcut = Fifth neighbors distance on lattice

//Some variables have the "host_" or "h_" prefixes - these are stored within the regular memory bandwidth
//Variables with the "dev_" prefix are to have their values stored and processed inside the GPU's physical memory (device memory)


int main()

{

//cudaSetDeviceFlags(cudaDeviceMapHost); //Mapping host memory is as slow as slow can get - use only if acessing data ONCE PER THREAD or if dataset is HUGE
				       //And there is absolutely no way to spread it across chunks - D E P R E C A T E D flag!

//####################################\\Variables//#####################################

const int t=13;

unsigned int natom;

double *host_x,*host_y,*host_z, *host_atrac, *host_repul; //coordinates and partial energy

double *dev_x, *dev_y, *dev_z; //data transfers to the GPU

double *h_a0,*h_a,*h_p,*h_q,*h_qui,*h_mass,*h_r0;

double *dev_a0, *dev_a, *dev_p, *dev_q, *dev_qui, *dev_mass, *dev_r0;

double *dev_atrac, *dev_repul; //, *dev_expa, *dev_expr, *dev_atrac, *dev_repul, *dev_r;//only used on device

//double *dev_rcut, *dev_rij, *dev_xx, *dev_yy, *dev_zz, *dev_qui2;//only used on device


double total_energy = 0.0, total_atrac = 0.0, total_repul = 0.0;

int i = 0, j = 0, k = 0;;

int *lbl, *dev_lbl;

char comment[70];

//#######################################\\Reading of the input file//#####################################


FILE *finp, *fout;
finp=fopen("coord_z.xyz","r");
fout=fopen("energy_output.dat","w");


clock_t start_t = clock();


fscanf(finp,"%u\n",&natom);
fscanf(finp,"%s\n",&comment);


//dynamic allocation of the host and device parameters (for transfers between CPU and GPU)

//size variables - contains the size of the memory bandwidth to be used by programs

size_t size_coord = sizeof(double)*natom;
size_t size_lbl = sizeof(int)*natom;
size_t size_pr = sizeof(double)*t;

	int deviceCount;
	cudaGetDeviceCount (&deviceCount);
	if (deviceCount <1)
	{
		printf("CUDA supporting video card not detected. Go eat a sandwich or something.\n");
		exit(1);
	}

//is it larger than 4GB

if(size_coord >= 4e+9)
{

printf("Large vector mode - SLOW\n");

cudaMallocHost((void**)&h_a0,size_pr);
cudaMallocHost((void**)&h_a,size_pr);
cudaMallocHost((void**)&h_p,size_pr);
cudaMallocHost((void**)&h_q,size_pr);
cudaMallocHost((void**)&h_qui,size_pr);
cudaMallocHost((void**)&h_mass,size_pr);
cudaMallocHost((void**)&h_r0,size_pr);

cudaMalloc((void**)&dev_a0,size_pr);
cudaMalloc((void**)&dev_a,size_pr);
cudaMalloc((void**)&dev_p,size_pr);
cudaMalloc((void**)&dev_q,size_pr);
cudaMalloc((void**)&dev_qui,size_pr);
cudaMalloc((void**)&dev_mass,size_pr);
cudaMalloc((void**)&dev_r0,size_pr);



//gpuErrchk( cudaMalloc((void**)&dev_lbl, size_lbl) );
cudaMallocHost((void**)&lbl,size_lbl);

host_x = (double*) malloc(size_coord);
host_y = (double*) malloc(size_coord);
host_z = (double*) malloc(size_coord);

if (host_x == NULL || host_y == NULL || host_z == NULL || lbl == NULL) printf("CPU ALLOCATION FAILED"); //Is there enough system memory

cudaHostRegister(host_x, size_coord, cudaHostRegisterMapped);

cudaHostRegister(host_y, size_coord, cudaHostRegisterMapped);

cudaHostRegister(host_z, size_coord, cudaHostRegisterMapped);

cudaHostRegister(lbl, size_lbl, cudaHostRegisterMapped);


if (host_x == NULL || host_y == NULL || host_z == NULL || lbl == NULL) printf("CPU ALLOCATION FAILED"); //Is there unmapped memory left for proper system operations


//Syntax is: DEVICE pointer to the HOST pointer assigning mapped memory to DEVICE of number X (yes, a mess)

cudaHostGetDevicePointer((void **) &dev_x, (void *) host_x, 0);
cudaHostGetDevicePointer((void **) &dev_y, (void *) host_y, 0);
cudaHostGetDevicePointer((void **) &dev_z, (void *) host_z, 0);
cudaHostGetDevicePointer((void **) &dev_lbl, (void *) lbl,  0);

if (dev_x == NULL || dev_y == NULL || dev_z == NULL || dev_lbl == NULL) printf("GPU ALLOCATION FAILED"); //Is it possible for this system to handle 64-bit pointer MAGIC


}

else if(size_coord < 4e+9){

//host variables

cudaMallocHost((void**)&lbl,size_lbl);

cudaMallocHost((void**)&host_x,size_coord);
cudaMallocHost((void**)&host_y,size_coord);
cudaMallocHost((void**)&host_z,size_coord);


cudaMallocHost((void**)&h_a0,size_pr);
cudaMallocHost((void**)&h_a,size_pr);
cudaMallocHost((void**)&h_p,size_pr);
cudaMallocHost((void**)&h_q,size_pr);
cudaMallocHost((void**)&h_qui,size_pr);
cudaMallocHost((void**)&h_mass,size_pr);
cudaMallocHost((void**)&h_r0,size_pr);

//device variables

gpuErrchk( cudaMalloc((void**)&dev_lbl, size_lbl) );
gpuErrchk( cudaMalloc((void**)&dev_x,size_coord)  );
gpuErrchk( cudaMalloc((void**)&dev_y,size_coord)  );
gpuErrchk( cudaMalloc((void**)&dev_z,size_coord)  );

cudaMalloc((void**)&dev_a0,size_pr);
cudaMalloc((void**)&dev_a,size_pr);
cudaMalloc((void**)&dev_p,size_pr);
cudaMalloc((void**)&dev_q,size_pr);
cudaMalloc((void**)&dev_qui,size_pr);
cudaMalloc((void**)&dev_mass,size_pr);
cudaMalloc((void**)&dev_r0,size_pr);




}

//scanning through input file, coordZ.xyz, and assigning values of "pot" to each corresponding atom label "lbl"

while (fscanf(finp,"%d %lf %lf %lf\n", &lbl[j], &host_x[j], &host_y[j], &host_z[j]) == 4)
	{

	j=j+1;

	}


clock_t finish_t = clock();

double cpu_time_used = ((double) (finish_t - start_t)) / CLOCKS_PER_SEC;

printf("\nTotal Time for reading file: %.5lf s\n", cpu_time_used);


//####################################################################
/*Parameters from: F. Cleri and V. Rossato, Phys. Rev. B, 48, 22 (1993).
Halliday, Resnick and Walker, Fourth Edition, 1996.*/

//Value of "t" is representing the total number of elements or alloys to give parameters, so this list has t "a0" arrays, t "r0" arrays, etc

//Note:Arrays are counted starting from the zero element (i.e, mass[0] is the mass of element 01 - Nickel, mass[1] is the mass of Copper, and so on)


//01 (Ni) Nickel

h_a0[0] = 3.523;	h_r0[0] = h_a0[0]/sqrt(2.0);	h_a[0] = 0.0376; h_p[0] = 16.999; h_q[0] = 1.189; h_qui[0] = 1.070; h_mass[0] = 58.6934;

//02 (Cu) Copper

h_a0[1] = 3.615;	h_r0[1] = h_a0[1]/sqrt(2.0);	h_a[1] = 0.0855; h_p[1] = 10.960; h_q[1] = 2.278; h_qui[1] = 1.224; h_mass[1] = 63.546;

//03 (Rh) Rhodium

h_a0[2] = 3.803;	h_r0[2] = h_a0[2]/sqrt(2.0);	h_a[2] = 0.0629; h_p[2] = 18.450; h_q[2] = 1.867; h_qui[2] = 1.660; h_mass[0] = 102.90550;

//04 (Pe) Palladium

h_a0[3] = 3.887;	h_r0[3] = h_a0[3]/sqrt(2.0);	h_a[3] = 0.1746; h_p[3] = 10.867; h_q[3] = 3.742; h_qui[3] = 1.718; h_mass[3] = 106.42;

//05 (Ag) Silver

h_a0[4] = 4.085;	h_r0[4] = h_a0[4]/sqrt(2.0);	h_a[4] = 0.1028; h_p[4] = 10.928; h_q[4] = 3.139; h_qui[4] = 1.178; h_mass[4] = 107.8682;

//06 (Ir) Iridium

h_a0[5] = 3.839;	h_r0[5] = h_a0[5]/sqrt(2.0);	h_a[5] = 0.1156; h_p[5] = 16.980; h_q[5] = 2.691; h_qui[5] = 2.289; h_mass[5] =192.217;

//07 (Pt) Platinum

h_a0[6] = 3.924;	h_r0[6] = h_a0[6]/sqrt(2.0);	h_a[6] = 0.2975; h_p[6] = 10.612; h_q[6] = 4.004; h_qui[6] = 2.695; h_mass[6] = 195.084;

//08 (Au) Gold

h_a0[7] = 4.0790;	h_r0[7] = h_a0[7]/sqrt(2.0);	h_a[7] = 0.20610; h_p[7] = 10.22900; h_q[7] = 4.0360; h_qui[7] = 1.7900; h_mass[7] = 196.9665690;

//09 (Al) Aluminium

h_a0[8] = 4.050;	h_r0[8] = h_a0[8]/sqrt(2.0);	h_a[8] = 0.1221; h_p[8] = 8.612; h_q[8] = 2.615; h_qui[8] = 1.316; h_mass[8] = 26.9815386;

//10 (Pb) Lead

h_a0[9] = 4.951;	h_r0[9] = h_a0[9]/sqrt(2.0);	h_a[9] = 0.0980; h_p[9] = 9.576; h_q[9] = 3.648; h_qui[9] = 0.914; h_mass[9] = 207.2;

//11 (Au-Cu) Alloy: Gold and Copper

h_a0[10] = 3.736;	h_r0[10] = h_a0[10]/sqrt(2.0);	h_a[10] = 0.169; h_p[10] = 9.890; h_q[10] = 2.940; h_qui[10] = 1.600; h_mass[10] = 180.66;

//12 (Ni - Al) Alloy: Nickel and Aluminium

h_a0[11] = 3.567;	h_r0[11] = h_a0[11]/sqrt(2.0);	h_a[11] = 0.0563; h_p[11] = 14.997; h_q[11] = 1.2823; h_qui[11] = 1.2349; h_mass[11] = 42.835;

//13 (Au-Ag) Alloy: Gold and Silver

h_a0[12] = 4.069;	h_r0[12] = h_a0[12]/sqrt(2.0);	h_a[12] = 0.067; h_p[12] = 15.630; h_q[12] = 3.580; h_qui[12] = 1.404; h_mass[12] = 196.05;

/*####################\\Kernel launch, thread specifications, memory management//####################*/

//Assigning 'k' values (parameters)

/*
for(j=0;j<natom;j++)
	{
	for (i = j + 1; i < natom; i++)
{

if((lbl[j] == 28) && (lbl[i] == 28)) k = 0; 	//Nickel
if((lbl[j] == 29) && (lbl[i] == 29)) k = 1; 	//Copper
if((lbl[j] == 45) && (lbl[i] == 45)) k = 2; 	//Rhodium
if((lbl[j] == 46) && (lbl[i] == 46)) k = 3; 	//Palladium
if((lbl[j] == 47) && (lbl[i] == 47)) k = 4; 	//Silver
if((lbl[j] == 77) && (lbl[i] == 77)) k = 5; 	//Iridium
if((lbl[j] == 78) && (lbl[i] == 78)) k = 6; 	//Platinum
if((lbl[j] == 79) && (lbl[i] == 79)) k = 7; 	//Gold
if((lbl[j] == 13) && (lbl[i] == 13)) k = 8; 	//Aluminium
if((lbl[j] == 82) && (lbl[i] == 82)) k = 9;	//Lead

if((lbl[j] == 79) && (lbl[i] == 29)) k = 10;	//Au - Cu
if((lbl[j] == 29) && (lbl[i] == 79)) k = 10;

if((lbl[j] == 28) && (lbl[i] == 13)) k = 11;	//Ni - Al
if((lbl[j] == 13) && (lbl[i] == 28)) k = 11;

if((lbl[j] == 79) && (lbl[i] == 47)) k = 12;	//Au - Ag
if((lbl[j] == 47) && (lbl[i] == 79)) k = 12;
}
	}
*/
//Filling arrays that are stored on GPU physical memory (every "dev_" variable)

	
	

	//Prepare clocking function
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	
	cudaEventRecord(start, 0);

cudaMemcpyAsync(dev_a0, h_a0, size_pr,cudaMemcpyHostToDevice);
cudaMemcpyAsync(dev_a, h_a, size_pr,cudaMemcpyHostToDevice);
cudaMemcpyAsync(dev_p, h_p, size_pr,cudaMemcpyHostToDevice);
cudaMemcpyAsync(dev_q, h_q, size_pr,cudaMemcpyHostToDevice);
cudaMemcpyAsync(dev_qui, h_qui, size_pr,cudaMemcpyHostToDevice);
cudaMemcpyAsync(dev_mass, h_mass, size_pr,cudaMemcpyHostToDevice);
cudaMemcpyAsync(dev_r0, h_r0, size_pr,cudaMemcpyHostToDevice);

if(size_coord < 4e+9)
{
cudaMemcpyAsync(dev_x,host_x,size_coord,cudaMemcpyHostToDevice);
cudaMemcpyAsync(dev_y,host_y,size_coord,cudaMemcpyHostToDevice);
cudaMemcpyAsync(dev_z,host_z,size_coord,cudaMemcpyHostToDevice);
cudaMemcpyAsync(dev_lbl, lbl, size_lbl,cudaMemcpyHostToDevice);
}



//host variables are no longer needed:

cudaFreeHost(h_a0);
cudaFreeHost(h_p);
cudaFreeHost(h_q);
cudaFreeHost(h_qui);
cudaFreeHost(h_mass);
cudaFreeHost(h_r0);


if(size_coord < 4e+9)
{
cudaFreeHost(host_x);
cudaFreeHost(host_y);
cudaFreeHost(host_z);
cudaFreeHost(lbl);
}



//********Launching Kernel***********


//launch specifications

const size_t thread = 512;
const size_t block = 512;

size_t size_block = block * sizeof(double); 
	

//allocating result vectors

cudaMallocHost((void**)&host_atrac,size_block);
cudaMallocHost((void**)&host_repul,size_block);

cudaMalloc((void**)&dev_atrac,size_block);
cudaMalloc((void**)&dev_repul,size_block);


/*   Next is the kernel function: it executes a "grid" number of threads simultaneously across a "block" number of blocks. In other words, a thread "i" on block "j" will execute at the same time as a thread "x" executes on another block "y".   */



potential_energy<<<block,thread,size_block>>>(natom, dev_x, dev_y, dev_z, dev_a0, dev_r0, dev_a, dev_mass, dev_qui, dev_q, dev_p, dev_lbl, dev_atrac, dev_repul); 

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop); // Time it took to calculate

// Clean up:
cudaEventDestroy(start);
cudaEventDestroy(stop);


//Copying partial results (potential of each atom) and getting rid of dummy variables
if(size_coord > 4e+9)
{
cudaHostUnregister(dev_x);
cudaHostUnregister(dev_y);
cudaHostUnregister(dev_z);
cudaHostUnregister(dev_lbl);
}
else
{
cudaFree(dev_x);
cudaFree(dev_y);
cudaFree(dev_z);
cudaFree(dev_lbl);
}
cudaFree(dev_a0);
cudaFree(dev_r0);
cudaFree(dev_a);
cudaFree(dev_mass);
cudaFree(dev_qui);
cudaFree(dev_q);
cudaFree(dev_p);


/*
cudaFree(dev_xx);
cudaFree(dev_yy);
cudaFree(dev_zz);
cudaFree(dev_expa);
cudaFree(dev_atrac);
cudaFree(dev_repul);
cudaFree(dev_r);
cudaFree(dev_qui2);
cudaFree(dev_rij);
cudaFree(dev_rcut);
*/


cudaMemcpyAsync(host_atrac, dev_atrac, size_block,cudaMemcpyDeviceToHost);
cudaMemcpyAsync(host_repul, dev_repul, size_block, cudaMemcpyDeviceToHost);

cudaFree(dev_atrac);
cudaFree(dev_repul);

//Final answer:

for (i = 0; i < block; i++)
{
	total_atrac += host_atrac[i];
	total_repul += host_repul[i];
}

total_energy = total_atrac + total_repul;


printf("\n>>>Ending Simulation\n");
printf("Total energy: \n %.8lf \n\n average energy per atom:\n %.8lf \n\n",total_energy,total_energy/natom);
printf("\n\nprocess completed! \n\n It took the GPU %.8f seconds to do it\n\n",elapsedTime/1000);
printf("======================================================================~\n");

fprintf(fout, "%.12le %.12le %.12le %.12le %.12le %.12le", total_energy, total_energy/natom, total_atrac, total_atrac/natom, total_repul, total_repul/natom);

fclose(finp);
fclose(fout);

cudaFreeHost(host_atrac);
cudaFreeHost(host_repul);
return 0;
}
