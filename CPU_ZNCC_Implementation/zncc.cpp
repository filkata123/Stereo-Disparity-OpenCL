#include <lodepng.h>

#include <algorithm>
#include <iostream>
#include <math.h> 
#include <vector>



void zncc_disparity(const std::vector<unsigned char> left_img, const std::vector<unsigned char> right_img, const int& width, const int& height, const int& win_size, const int& max_disp, std::vector<double>& disp_map)
{
    // Allocate memory for the disparity map
    disp_map.resize(width * height, 0.0);

    // Loop over all pixels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            // Check if the current pixel is near the border
            if (x < win_size / 2 || x >= width - win_size / 2 || y < win_size / 2 || y >= height - win_size / 2) {
                disp_map[y * width + x] = max_disp;
                continue;
            }

            // Initialize the maximum correlation and the best disparity for the current pixel
            double max_corr = -1.0;
            int best_disp = 0;

            // Loop over all possible disparities up to the maximum disparity
            for (int disp = 0; disp <= max_disp && x - disp >= win_size / 2; disp++) {

                // Compute the correlation between the two windows centered at the current pixel and its corresponding pixel in the other 
                double sum1 = 0.0, sum2 = 0.0, sum12 = 0.0;

                // Loop over all pixels inside the window
                for (int dy = -win_size / 2; dy <= win_size / 2; dy++) {
                    for (int dx = -win_size / 2; dx <= win_size / 2; dx++) {

                        // Compute the pixel coordinates in the two images for the current window position and disparity
                        int x1 = x + dx, y1 = y + dy;
                        int x2 = x - disp + dx, y2 = y + dy;

                        // Compute the pixel intensities in the two windows
                        unsigned char p1 = left_img[y1 * width + x1];
                        unsigned char p2 = right_img[y2 * width + x2];

                        // Compute the sums for mean and correlation calculations
                        sum1 += p1;
                        sum2 += p2;
                        sum12 += p1 * p2;
                    }
                }

                // Compute the means of the pixel intensities in the two windows
                double mean1 = sum1 / (win_size * win_size);
                double mean2 = sum2 / (win_size * win_size);

                // Compute the standard deviations of the pixel intensities in the two windows
                double std1 = 0.0, std2 = 0.0, cov = 0.0;
                for (int dy = -win_size / 2; dy <= win_size / 2; dy++) {
                    for (int dx = -win_size / 2; dx <= win_size / 2; dx++) {

                        // Compute the pixel coordinates in the two images for the current window position and disparity
                        int x1 = x + dx, y1 = y + dy;
                        int x2 = x - disp + dx, y2 = y + dy;

                        // Compute the pixel intensities in the two windows
                        unsigned char p1 = left_img[y1 * width + x1];
                        unsigned char p2 = right_img[y2 * width + x2];

                        // Compute the sums for standard deviation and correlation calculations
                        std1 += (p1 - mean1) * (p1 - mean1);
                        std2 += (p2 - mean2) * (p2 - mean2);
                        cov += (p1 - mean1) * (p2 - mean2);
                    }
                }

                std1 = sqrt(std1 / (win_size * win_size));
                std2 = sqrt(std2 / (win_size * win_size));
                cov = cov / (win_size * win_size);

                // Compute the correlation between the two windows using the ZNCC formula
                double correlation = (cov - mean1 * mean2) / (std1 * std2);

                // If the correlation is higher than the current maximum, update the maximum and the corresponding disparity
                if (correlation > max_corr) {
                    max_corr = correlation;
                    best_disp = disp;
                }
            }

            disp_map[y * width + x] = best_disp;
        }
    }
}

void cross_check(const std::vector<unsigned char> left_img, const std::vector<unsigned char> right_img, const int& width, const int& height, const int& win_size, const int& max_disp, std::vector<double>& cross_disp_map)
{
    // Allocate memory for the disparity maps for both directions
    std::vector<double> disp_map_left(width * height, 0.0);
    std::vector<double> disp_map_right(width * height, 0.0);

    // Compute the disparity map for the left image
    zncc_disparity(left_img, right_img, width, height, win_size, max_disp, disp_map_left);

    // Compute the disparity map for the right image
    zncc_disparity(right_img, left_img, width, height, win_size, max_disp, disp_map_right);

    // Loop over all pixels inside the image boundary
    for (int y = win_size / 2; y < height - win_size / 2; y++) {
        for (int x = win_size / 2; x < width - win_size / 2; x++) {

            // Get the disparity values for the current pixel in both directions
            double disp_left = disp_map_left[y * width + x];
            double disp_right = disp_map_right[y * width + x - disp_left];

            // Check if the disparity values agree | abs used to account for rounding errors
            if (disp_right >= 0 && std::abs(disp_left - (-disp_right)) <= 8.0) {
                // If the disparities agree, use the left disparity value as the final disparity for the pixel
                cross_disp_map[y * width + x] = disp_left;
            }
            else {
                // Otherwise, mark the pixel as invalid
                cross_disp_map[y * width + x] = 0;
            }
        }
    }
}

void occlusion_filling(const std::vector<double>& disp_map, const int& width, const int& height, std::vector<double>& disp_map_filled)
{
    // Allocate memory for the filled disparity map
    disp_map_filled.resize(width * height);

    // Copy the input disparity map to the output disparity map
    std::copy(disp_map.begin(), disp_map.end(), disp_map_filled.begin());

    // Loop over all pixels inside the image boundary
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            // Check if the current pixel is marked as invalid
            if (disp_map[y * width + x] < 0) {

                // Initialize the list of valid disparity values in the 8-neighborhood of the current pixel
                std::vector<double> neighbors;

                // Loop over the 8-neighbors of the current pixel
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        // Skip the center pixel
                        if (dx == 0 && dy == 0) continue;

                        // Get the disparity value for the current neighbor
                        double neighbor_disp = disp_map[(y + dy) * width + x + dx];

                        // If the neighbor is valid, add its disparity value to the list
                        if (neighbor_disp >= 0) {
                            neighbors.push_back(neighbor_disp);
                        }
                    }
                }

                // If at least one valid disparity value was found in the 8-neighborhood,
                // set the current pixel's disparity value to the median of the valid values
                if (!neighbors.empty()) {
                    double median_disp = neighbors[neighbors.size() / 2];
                    disp_map_filled[y * width + x] = median_disp;
                }
            }
        }
    }
}

void normalize_to_char(const std::vector<double>& disp_map, const int& width, const int& height, const int& ndisp, std::vector<unsigned char>& norm_vec)
{
    // Allocate memory for the normalized disparity map
    norm_vec.resize(width * height);

    // Loop over all pixels and normalize the disparity values
    for (int i = 0; i < width * height; i++) {
        norm_vec[i] = static_cast<unsigned char>(disp_map[i] / ndisp * 255);
    }
}

int main()
{   // from calib.txt - downsized
    // each pixel in the downsampled image corresponds to a larger area in the original image
    // reducing the resolution of the images reduces the maximum disparity that can be reliably estimated
    // Disparity value represents the number of pixels that one image point is shifted relative to the other image point,
    // so the maximum disparity value is directly related to the image resolution.
    // To account for this reduction in resolution, we need to adjust the maximum disparity value by the same factor that we used to downsample the image
    int ndisp = 260 * 16; 
    int win_size = 11;

    // setup inputs and outputs
    const char* leftImgName = "../img/im0_out.png";
    const char* rightImgName = "../img/im1_out.png";

    const char* leftImgNameOut = "../img/im0_disparity.png";
    const char* rightImgNameOut = "../img/im1_disparity.png";

    // create containers for raw images
    std::vector<unsigned char> leftImage;
    std::vector<unsigned char> rightImage;
    unsigned int width, height;

    // decode images
    unsigned int error = lodepng::decode(leftImage, width, height, leftImgName, LCT_GREY, 8);
    if (error) std::cout << "decoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

    error = lodepng::decode(rightImage, width, height, rightImgName, LCT_GREY, 8);
    if (error) std::cout << "decoder error second image: " << error << ": " << lodepng_error_text(error) << std::endl;

    // apply zncc
    std::vector<double> leftImageDisparity(width * height);
    //std::vector<double> rightImageDisparity(width * height);
    zncc_disparity(leftImage, rightImage, width, height, win_size, ndisp, leftImageDisparity);
    //zncc_disparity(rightImage, leftImage, width, height, win_size, ndisp, leftImageDisparity);

    std::vector<unsigned char> leftImageDisparityNormalized(width * height);
    //std::vector<unsigned char> rightImageDisparityNormalized(width * height);
    normalize_to_char(leftImageDisparity, width, height, ndisp, leftImageDisparityNormalized);
    //normalize_to_char(rightImageDisparity, width, height, ndisp, rightImageDisparityNormalized);

    // encode resized and grayscaled images (im*_out)
    error = lodepng::encode(leftImgNameOut, leftImageDisparityNormalized, width, height, LCT_GREY, 8);
    if (error) std::cout << "encoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

    //error = lodepng::encode(rightImgNameOut, rightImageDisparityNormalized, width, height, LCT_GREY, 8);
    //if (error) std::cout << "encoder error second image: " << error << ": " << lodepng_error_text(error) << std::endl;
}