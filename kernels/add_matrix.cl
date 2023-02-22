__kernel void add_matrix(const int rows, const int cols, __global int* A, __global int* B, __global int* out)
{	
	// get work item id (both dimensions)
	int i = get_global_id(0);
	int j = get_global_id(1);

	// matrix addition 
	if ((i < rows) && (j < cols))
		out[i * cols + j] = A[i * cols + j] + B[i * cols + j];
}