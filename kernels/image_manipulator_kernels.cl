__kernel void convert_grayscale(__global unsigned char* img_data, __global unsigned char* out)
{	
	const int idx = get_global_id(0);

    // Check whether the index is a multiple of four (RGBA, we skip A)
    if (idx % 4 == 0) 
    {
        // Add the current value with the next three values and divide by 3
        unsigned char sum = (img_data[idx] + img_data[idx + 1] + img_data[idx + 2]) / 3;

        // Store the averaged value in the output array at the corresponding index
        out[idx / 4] = sum;
    }
}

__kernel void apply_moving_filter(const int height, const int width, const int GAUSSIAN_SIZE, __global unsigned char* img_data, __global float* gaussian_filter_matrix, __global unsigned char* out)
{	
	int i = get_global_id(0);
	int j = get_global_id(1);

	if ((i < height) && (j < width))
    {
		float sum = 0.0f;

        for (int k = 0; k < GAUSSIAN_SIZE; k++)
        {
            for (int l = 0; l < GAUSSIAN_SIZE; l++)
            {
                int x = j + l - GAUSSIAN_SIZE / 2;
                int y = i + k - GAUSSIAN_SIZE / 2;

                if (x < 0 || x >= width || y < 0 || y >= height)
                {
                    continue;
                }
                sum += gaussian_filter_matrix[k * GAUSSIAN_SIZE + l] * (float)img_data[y * width + x];
            }
        }
        out[i * width + j] = (unsigned char)sum;
    }
}