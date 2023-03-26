__kernel void convert_grayscale(__read_only image2d_t input_img, __write_only image2d_t out_image)
{	
	const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    uint4 pixel = read_imageui(input_img, coord);
    
    // Add the current value with the next two values and divide by 3
    unsigned char sum = (pixel.x + pixel.y + pixel.z) / 3;

    // Store the averaged value in the output iamge at the corresponding index
    write_imageui(out_image, coord, sum);    
}

__kernel void resize_image(const int resize_factor, __read_only image2d_t input_img, __write_only image2d_t out_image)
{	
	int i = get_global_id(1);
    int j = get_global_id(0);

    // iterate through a resize_factor * resize_factor box and take its average as new pixel value
    int sum = 0;
    for (int k = i * resize_factor; k < (i + 1) * resize_factor; k++) {
        for (int l = j * resize_factor; l < (j + 1) * resize_factor; l++) {
            // get only x value of pixel, as we are using one channel for a grayscale scalar
            sum += read_imageui(input_img, (int2)(l, k)).x;
        }
    }

    // add pixel to new image
    write_imageui(out_image, (int2)(j, i), sum / (resize_factor * resize_factor));
}