// This combination of kernels deals with PCR-Thomas

inline void thomas(__local float* a,__local float* b,__local float* c,__local float* d,
							__local float* x,int sizeSmallerSystem, int stride, int thid)
{
	c[thid] = c[thid] / b[thid];
	d[thid] = d[thid] / b[thid];
	int startLocationSystem = stride + thid;
	float tmp;
	for (int k = startLocationSystem; k < sizeSmallerSystem; k += stride)
	{
		tmp = (b[k] - a[k] * c[k - stride]);
		c[k] = c[k] / tmp;
		d[k] = (d[k] - d[k - stride] * a[k]) / tmp;
	}
	
	// Backward substitution
	int endLocationSystem = sizeSmallerSystem - stride + thid;
	x[endLocationSystem] = d[endLocationSystem];
	for (int k = endLocationSystem - stride; k >= 0; k -= stride)
	{
		x[k] = d[k] - c[k]*x[k + stride];
	}
}

__kernel void modified_pcr(__global float *a_d, __global float *b_d, __global float *c_d, __global float *d_d, __global float *x_d, 
									 __local float *shared, int system_size, int num_systems, int iterations)
{
	int thid = get_local_id(0);
    int blid = get_group_id(0);

	int delta = 1;

	// These are pointers in the __private address space 
	// that point to an array of floats in the __local 
	// address space.
	
	__local float* a = shared; // Enough shared memory is allocated for all 5 vectors
	__local float* b = &a[system_size+1];
	__local float* c = &b[system_size+1];
	__local float* d = &c[system_size+1];
	__local float* x = &d[system_size+1];

	a[thid] = a_d[thid + blid * system_size];
	b[thid] = b_d[thid + blid * system_size];
	c[thid] = c_d[thid + blid * system_size];
	d[thid] = d_d[thid + blid * system_size];
  
	float aNew, bNew, cNew, dNew;
  
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// parallel cyclic reduction
	for (int j = 0; j < iterations; j++)
	{
		int i = thid;

		int iRight = i+delta;
		iRight = iRight & (system_size-1);

		int iLeft = i-delta;
		iLeft = iLeft & (system_size-1);

		float tmp1 = a[i] / b[iLeft];
		float tmp2 = c[i] / b[iRight];

		bNew = b[i] - c[iLeft] * tmp1 - a[iRight] * tmp2;
		dNew = d[i] - d[iLeft] * tmp1 - d[iRight] * tmp2;
		aNew = -a[iLeft] * tmp1;
		cNew = -c[iRight] * tmp2;

		barrier(CLK_LOCAL_MEM_FENCE);
        
		b[i] = bNew;
 		d[i] = dNew;
		a[i] = aNew;
		c[i] = cNew;	
    
	    delta *= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Here we want to call Thomas

	if(thid <delta)
		thomas(a,b,c,d,x,system_size,delta,thid);
	barrier(CLK_LOCAL_MEM_FENCE);
	
    x_d[thid + blid * system_size] = x[thid];
}
