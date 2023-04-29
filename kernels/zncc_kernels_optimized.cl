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
    const int2 idx = (int2)(get_global_id(0), get_global_id(1)); // (width, height) indexes
    // iterate through a resize_factor * resize_factor box and take its average as new pixel value
    int sum = 0;
    for (int k = idx.y * resize_factor; k < (idx.y + 1) * resize_factor; k++) {
        for (int l = idx.x * resize_factor; l < (idx.x + 1) * resize_factor; l++) {
            // get only x value of pixel, as we are using one channel for a grayscale scalar
            sum += read_imageui(input_img, (int2)(l, k)).x;
        }
    }

    // add pixel to output buffer
    out_image[idx.y * get_global_size(0) + idx.x] = (unsigned char)(sum / (resize_factor * resize_factor));
}

__kernel void calc_zncc(const int half_window_size, const char is_left_image,
    const __global unsigned char* left_image, const __global unsigned char* right_image, __global int* disparity_map, const int max_disparity)
{	
    const int2 idx = (int2)(get_global_id(0), get_global_id(1)); // (width, height) indexes

    const int width = get_global_size(0);
    const int height = get_global_size(1);

    int best_disp = 0;
    float best_ZNCC = -100.0;

    // handle borders | keep best_disp at 0, so borders will be black
    if (!(idx.y >= height - half_window_size || idx.x >= width - half_window_size ||
        idx.y <= half_window_size || idx.x <= half_window_size))
    {
        // go over all disparity values
        for (int d = 0; d < max_disparity; d++)
        {
            float zncc = 0.0;
            float numerator = 0.0, denominator1 = 0.0, denominator2 = 0.0;
            float2 means = (0.0, 0.0); // left_mean, right_mean;

            // calculate mean for each window - changes for different disparities, as the right_mean is calculated based on the disparity
            int avg_count = 0;
            for (int win_y = -half_window_size; win_y < half_window_size; win_y++)
            {
                for (int win_x = -half_window_size; win_x < half_window_size; win_x++)
                {
                    // don't allow pixel to go to previous row
                    if (d > idx.x + win_x)
                    {
                        continue;
                    }

                    // calculate pixel indices for the current window position
                    int left_pixel_index = (idx.y + win_y) * width + (idx.x + win_x);
                    int right_pixel_index = (idx.y + win_y) * width + (idx.x + win_x - is_left_image * d);
                    if (right_pixel_index >= width * height ||
                        right_pixel_index <= 0)
                    {
                        continue;
                    }

                    means.x += left_image[left_pixel_index];
                    means.y += right_image[right_pixel_index];
                    avg_count++;
                }

            }
            means = native_divide(means, avg_count);

            // calculate numerator and denominators used for zncc calculation
            for (int win_y = -half_window_size; win_y < half_window_size; win_y++)
            {
                for (int win_x = -half_window_size; win_x < half_window_size; win_x++)
                {
                    // don't allow pixel to go to previous row
                    if (d > idx.x + win_x)
                    {
                        continue;
                    }

                    // calculate pixel indices for the current window position
                    int left_pixel_index = (idx.y + win_y) * width + (idx.x + win_x);
                    int right_pixel_index = (idx.y + win_y) * width + (idx.x + win_x - is_left_image * d);
                    if (right_pixel_index >= width * height ||
                        right_pixel_index <= 0)
                    {
                        continue;
                    }

                    // calculate numerator and denominators and sum
                    numerator += (left_image[left_pixel_index] - means.x) * (right_image[right_pixel_index] - means.y);
                    denominator1 += pown(left_image[left_pixel_index] - means.x, 2);
                    denominator2 += pown(right_image[right_pixel_index] - means.y, 2);

                }

            }

            float denominator = native_sqrt(denominator1) * native_sqrt(denominator2);
            if (denominator != 0) {
                // calculate zncc value
                zncc = native_divide(numerator, denominator);

                // compare current zncc value to the current best value and update
                // value and disparity that led to it if new zncc value is better
                if (zncc > best_ZNCC)
                {
                    best_ZNCC = zncc;
                    best_disp = d;
                }
            }
        }
    }
   
    // add pixel to output buffer
    disparity_map[idx.y * width + idx.x] = best_disp;
}

__kernel void cross_check(const int cross_diff,
    const __global int* left_image, const __global int* right_image,  __global int* cross_checked_image)
{	
    const int2 idx = (int2)(get_global_id(0), get_global_id(1)); // (width, height) indexes

    const int width = get_global_size(0);
    const int height = get_global_size(1);

    // Get the disparity values for the current pixel in both directions
    int disp_left = left_image[idx.y * width + idx.x];
    int disp_right = right_image[idx.y * width + idx.x];

    // Check if the disparity values agree | abs used to account for rounding errors
    if (abs(disp_left - disp_right) <= cross_diff) {
        // If the disparities agree, use the left disparity value as the final disparity for the pixel
        cross_checked_image[idx.y * width + idx.x] = disp_left;
    }
    else {
        // Otherwise, mark the pixel as invalid
        cross_checked_image[idx.y * width + idx.x] = 0;
    }
    
}

__kernel void occlusion_filling(const int n_count,
    __global int* cross_checked_image)
{	
    const int2 idx = (int2)(get_global_id(0), get_global_id(1)); // (width, height) indexes

    const int width = get_global_size(0);
    const int height = get_global_size(1);

    // handle borders | keep bestDisp at 0, so borders will stay black
    if (!(idx.y >= height - (n_count / 2) || idx.x >= width - (n_count / 2) ||
        idx.y <= n_count / 2 || idx.x <= n_count / 2))
    {
        // Check if the current pixel is marked as invalid
        if (cross_checked_image[idx.y * width + idx.x] == 0) {

            // Initialize the list of valid disparity values in the n-neighborhood of the current pixel
            int i = 0;

            // NEIGHBOUR_SIZE passed as a -D define when building
            int neighbours[NEIGHBOUR_SIZE];

            // Loop over the n-neighbors of the current pixel
            for (int dy = -n_count / 2; dy <= n_count / 2; dy++) {
                for (int dx = -n_count / 2; dx <= n_count / 2; dx++) {
                    // Skip the center pixel
                    if (!(dx == 0 && dy == 0))
                    {
                        // Get the disparity value for the current neighbor
                        int neighbor_disp = cross_checked_image[(idx.y + dy) * width + (idx.x + dx)];

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
                cross_checked_image[idx.y * width + idx.x] = neighbours[(i + 1) / 2];
            }
        }
    }
}

__kernel void normalize_to_char(const int n_disp,
    const __global int* filled_image, __global unsigned char* norm_image)
{	
    const int i = get_global_id(0);

    // normalize disparity image to grayscale (char)
    norm_image[i] = (unsigned char)(((float)(filled_image[i])) / n_disp * 255);
}