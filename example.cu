#define __GPU
#define __NOPROTO

#include "fargo3d.h"

#define Sxj(i) Sxj_s[(i)]
#define Syj(i) Syj_s[(i)]
#define Szj(i) Szj_s[(i)]
#define Sxk(i) Sxk_s[(i)]
#define Syk(i) Syk_s[(i)]
#define Szk(i) Szk_s[(i)]
#define InvVj(i) InvVj_s[(i)]

CONSTANT(real, Sxj_s, 1098);
CONSTANT(real, Syj_s, 1098);
CONSTANT(real, Szj_s, 1098);
CONSTANT(real, Sxk_s, 1098);
CONSTANT(real, Syk_s, 1098);
CONSTANT(real, Szk_s, 1098);
CONSTANT(real, InvVj_s, 1098);

__global__ void UpdateDensityY_kernel(real dt,  real* qb,real* vy,real* rho_s,int pitch,int stride,int size_x,int size_y,int size_z) {

  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int ll;
  int llyp;

#ifdef X 
i = threadIdx.x + blockIdx.x * blockDim.x;
#else 
i = 0;
#endif 
#ifdef Y 
j = threadIdx.y + blockIdx.y * blockDim.y;
#else 
j = 0;
#endif 
#ifdef Z 
k = threadIdx.z + blockIdx.z * blockDim.z;
#else 
k = 0;
#endif

#ifdef Z
if(k>=0 && k<size_z) {
#endif
#ifdef Y
if(j>=0 && j<size_y) {
#endif
#ifdef X
if(i<size_x) {
#endif
	ll = l;
	llyp = lyp;

	qb[ll] += (vy[ll]*rho_s[ll]*SurfY(j,k) - vy[llyp]*rho_s[llyp] *SurfY(j+1,k)) * dt * InvVol(j,k);
#ifdef X 
 } 
 #endif
#ifdef Y 
 } 
 #endif
#ifdef Z 
 } 
 #endif
}

extern "C" void UpdateDensityY_gpu(real dt, Field *Q) {


  INPUT(Q);
  INPUT(Vy_temp);
  INPUT(DensStar);
  OUTPUT(Q);

dim3 block (BLOCK_X, BLOCK_Y, BLOCK_Z);
dim3 grid ((Nx+2*NGHX+block.x-1)/block.x,((Ny+2*NGHY)+block.y-1)/block.y,((Nz+2*NGHZ)+block.z-1)/block.z);

#ifdef BIGMEM
#define xmin_d &Xmin_d
#define ymin_d &Ymin_d
#define zmin_d &Zmin_d
#define Sxj_d &Sxj_d
#define Syj_d &Syj_d
#define Szj_d &Szj_d
#define Sxk_d &Sxk_d
#define Syk_d &Syk_d
#define Szk_d &Szk_d
#define InvVj_d &InvVj_d
#endif

CUDAMEMCPY(Sxj_s, Sxj_d, sizeof(real)*(Ny+2*NGHY), 0, cudaMemcpyDeviceToDevice);
CUDAMEMCPY(Syj_s, Syj_d, sizeof(real)*(Ny+2*NGHY), 0, cudaMemcpyDeviceToDevice);
CUDAMEMCPY(Szj_s, Szj_d, sizeof(real)*(Ny+2*NGHY), 0, cudaMemcpyDeviceToDevice);
CUDAMEMCPY(Sxk_s, Sxk_d, sizeof(real)*(Nz+2*NGHZ), 0, cudaMemcpyDeviceToDevice);
CUDAMEMCPY(Syk_s, Syk_d, sizeof(real)*(Nz+2*NGHZ), 0, cudaMemcpyDeviceToDevice);
CUDAMEMCPY(Szk_s, Szk_d, sizeof(real)*(Nz+2*NGHZ), 0, cudaMemcpyDeviceToDevice);
CUDAMEMCPY(InvVj_s, InvVj_d, sizeof(real)*(Ny+2*NGHY), 0, cudaMemcpyDeviceToDevice);


cudaFuncSetCacheConfig(UpdateDensityY_kernel, cudaFuncCachePreferL1 );
UpdateDensityY_kernel<<<grid,block>>>(dt,Q->field_gpu,Vy_temp->field_gpu,DensStar->field_gpu,Pitch_gpu,Stride_gpu,Nx+2*NGHX,Ny+2*NGHY-1,Nz+2*NGHZ);

check_errors("UpdateDensityY_kernel");

}
