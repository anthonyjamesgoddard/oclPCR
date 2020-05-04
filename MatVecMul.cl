#define BLOCK_SIZE 8
__kernel void
matrixVectorMul(__global float* C, 
          __global float* A, 
          __global float* B, int wA)
{

	float Csub=0;
	
    __local float As[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE];

    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
 
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
 

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
	
    int bBegin = 0;
    int bStep  = BLOCK_SIZE ;
 
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) 
    { 
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[tx]     = B[b + tx];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
		for(int k=0;k<BLOCK_SIZE;k++)
			Csub += As[ty][k] * Bs[k];
			
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int c =  BLOCK_SIZE * by;
    C[c +  ty] = Csub;

}
