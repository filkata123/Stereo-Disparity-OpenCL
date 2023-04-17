__kernel void convert_grayscale(__read_only image2d_t input_img, __write_only image2d_t out_image)
{	
	const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    const uint4 pixel = read_imageui(input_img, coord);
    // Add the current value with the next two values and divide by 3
    const unsigned char sum = (pixel.x + pixel.y + pixel.z) / 3;

    // Store the averaged value in the output iamge at the corresponding index
    write_imageui(out_image, coord, sum);    
}

__kernel void resize_image(const int resize_factor, __read_only image2d_t input_img, __global unsigned char* out_image)
{	
    const int j = get_global_id(0); // width index
	const int i = get_global_id(1); // height index
    // iterate through a resize_factor * resize_factor box and take its average as new pixel value
    int sum = 0;
    for (int k = i * resize_factor; k < (i + 1) * resize_factor; k++) {
        for (int l = j * resize_factor; l < (j + 1) * resize_factor; l++) {
            // get only x value of pixel, as we are using one channel for a grayscale scalar
            sum += read_imageui(input_img, (int2)(l, k)).x;
        }
    }

    // add pixel to output buffer
    out_image[i * get_global_size(0) + j] = (unsigned char)(sum / (resize_factor * resize_factor));
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
__kernel void calc_zncc(const int half_window_size, const char is_left_image,
    const __global unsigned char* left_image, const __global unsigned char* right_image,  __global int* disparity_map)
{	
    // TODO: add 3rd dimension, add local size of max_disparity, add a memblock right before zncc comparison - then add mutex so as to find biggest value
    // bestZncc and bestDisp become local for the work group
    const int x = get_global_id(0); // width index
	const int y = get_global_id(1); // height index
    const int d = get_local_id(0); // disparity index

    const int width = get_global_size(0);
    const int height = get_global_size(1);
    //int max_disp = get_global_size(2);

    __local atomic_int best_disp;
    __local atomic_float best_ZNCC;
    if (d == 0) 
    {
        atomic_store(&best_ZNCC, -INFINITY);
        atomic_store(&best_disp, -1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float zncc = 0.0;
    // handle borders | keep best_disp at 0, so borders will be black
    if (!(y >= height - half_window_size || x >= width - half_window_size ||
        y <= half_window_size || x <= half_window_size))
    {
        
        float numerator = 0.0, denominator1 = 0.0, denominator2 = 0.0;
        float left_mean = 0.0, right_mean = 0.0;

        // calculate mean for each window - changes for different disparities, as the right_mean is calculated based on the disparity
        int avg_count = 0;
        for (int win_y = -half_window_size; win_y < half_window_size; win_y++)
        {
            for (int win_x = -half_window_size; win_x < half_window_size; win_x++)
            {
                // don't allow pixel to go to previous row
                if (d > x + win_x)
                {
                    continue;
                }

                int left_pixel_index = (y + win_y) * width + (x + win_x);
                int right_pixel_index = (y + win_y) * width + (x + win_x - is_left_image * d);
                if (right_pixel_index >= width * height ||
                    right_pixel_index <= 0)
                {
                    continue;
                }

                left_mean += left_image[left_pixel_index];
                right_mean += right_image[right_pixel_index];
                avg_count++;
            }

        }
        left_mean = native_divide(left_mean, avg_count);
        right_mean = native_divide(right_mean, avg_count);

        for (int win_y = -half_window_size; win_y < half_window_size; win_y++)
        {
            for (int win_x = -half_window_size; win_x < half_window_size; win_x++)
            {
                // don't allow pixel to go to previous row
                if (d > x + win_x)
                {
                    continue;
                }

                int left_pixel_index = (y + win_y) * width + (x + win_x);
                int right_pixel_index = (y + win_y) * width + (x + win_x - is_left_image * d);
                if (right_pixel_index >= width * height ||
                    right_pixel_index <= 0)
                {
                    continue;
                }

                // calculate zncc value for each window
                numerator += (left_image[left_pixel_index] - left_mean) * (right_image[right_pixel_index] - right_mean);
                denominator1 += pown(left_image[left_pixel_index] - left_mean, 2);
                denominator2 += pown(right_image[right_pixel_index] - right_mean, 2);

            }

        }
        

        float denominator = native_sqrt(denominator1) * native_sqrt(denominator2);
        if (denominator != 0) {
            zncc = native_divide(numerator, denominator);
        }
        
    }

    barrier(CLK_LOCAL_MEM_FENCE);
   
    if (zncc > atomic_load(&best_ZNCC))
    {
        atomic_exchange(&best_ZNCC, zncc);
        atomic_exchange(&best_disp, d);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (d == 0) {
        // add pixel to output buffer
        disparity_map[y * width + x] = atomic_load(&best_disp);
    }
}

__kernel void cross_check(const int cross_diff,
    const __global int* left_image, const __global int* right_image,  __global int* cross_checked_image)
{	
    const int x = get_global_id(0); // width index
	const int y = get_global_id(1); // height index

    const int width = get_global_size(0);
    const int height = get_global_size(1);

    // Get the disparity values for the current pixel in both directions
    int disp_left = left_image[y * width + x];
    int disp_right = right_image[y * width + x];

    // Check if the disparity values agree | abs used to account for rounding errors
    if (abs(disp_left - disp_right) <= cross_diff) {
        // If the disparities agree, use the left disparity value as the final disparity for the pixel
        cross_checked_image[y * width + x] = disp_left;
    }
    else {
        // Otherwise, mark the pixel as invalid
        cross_checked_image[y * width + x] = 0;
    }
    
}

__kernel void occlusion_filling(const int n_count, __local int* neighbours,
    __global int* cross_checked_image)
{	
    const int x = get_global_id(0); // width index
	const int y = get_global_id(1); // height index

    const int width = get_global_size(0);
    const int height = get_global_size(1);

    // handle borders | keep bestDisp at 0, so borders will stay black
    if (!(y >= height - (n_count / 2) || x >= width - (n_count / 2) ||
        y <= n_count / 2 || x <= n_count / 2))
    {
        // Check if the current pixel is marked as invalid
        if (cross_checked_image[y * width + x] == 0) {

            // Initialize the list of valid disparity values in the n-neighborhood of the current pixel
            int i = 0;

            // Loop over the n-neighbors of the current pixel
            for (int dy = -n_count / 2; dy <= n_count / 2; dy++) {
                for (int dx = -n_count / 2; dx <= n_count / 2; dx++) {
                    // Skip the center pixel
                    if (!(dx == 0 && dy == 0))
                    {
                        // Get the disparity value for the current neighbor
                        int neighbor_disp = cross_checked_image[(y + dy) * width + (x + dx)];

                        // If the neighbor is valid, add its disparity value to the list
                        if (neighbor_disp > 0) {
                            neighbours[i] = neighbor_disp;
                            i++;
                        }
                    }
                }
            }

            // If at least one valid disparity value was found in the n-neighborhood,
            // set the current pixel's disparity value to the median of the valid values
            if (i > 0) {
                cross_checked_image[y * width + x] = neighbours[(i + 1) / 2];
            }
        }
    }
}

__kernel void normalize_to_char(const int n_disp,
    const __global int* filled_image, __global unsigned char* norm_image)
{	
    const int i = get_global_id(0);

    norm_image[i] = (unsigned char)(((float)(filled_image[i])) / n_disp * 255);
}