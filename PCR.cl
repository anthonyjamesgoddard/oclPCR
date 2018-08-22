// This kernel carries our parallel cyclic reduction

__kernel void pcr(__global float *a_d, __global float *b_d, __global float *c_d, __global float *d_d, __global float *x_d,__local float *shared, int system_size, int num_systems, int iterations)
{
	int thid = get_local_id(0);
    int blid = get_group_id(0);

	int s = 1;

	__local float* a = shared;
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
	
	for (int j = 0; j < iterations; j++)
	{
		int i = thid;

		int iRight = i+s;
		iRight = iRight & (system_size-1);

		int iLeft = i-s;
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
    
	    s *= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (thid < s)
	{
		int addr1 = thid;
		int addr2 = thid + s;
		float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
		x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
		x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
	}
    
	barrier(CLK_LOCAL_MEM_FENCE);
    
    x_d[thid + blid * system_size] = x[thid];
}
