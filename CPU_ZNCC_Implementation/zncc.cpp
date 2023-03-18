#include <lodepng.h>

#include <algorithm>
#include <iostream>
#include <math.h> 
#include <vector>
#include <Windows.h>


void GrayScaleImageConversion(const std::vector<unsigned char>& image, unsigned int width, unsigned int height, std::vector<unsigned char>& imageGray)
{
    // iterate over every four values, as input is 4 channeled RGBA
    char channel = 4;
    for (size_t i = 0; i < image.size(); i += channel)
    {
        // Add up R, G and B values and divide to get grayscale value.
        // We don't care about the A value, so it is not used.
        imageGray[i / channel] = (image[i + 0] + image[i + 1] + image[i + 2]) / 3;
    }
}

void ResizeImage(std::vector<unsigned char> image, unsigned int width, unsigned int height, unsigned int resizeFactor, std::vector<unsigned char>& imageResized)
{
    // Divide image into resizeFactor x resizeFactor blocks and then use the average value of said blocks as the value of the new pixel
    imageResized.resize((width * height) / (resizeFactor * resizeFactor));

    int newWidth = width / resizeFactor;
    int newHeight = height / resizeFactor;

    for (int i = 0; i < newHeight; ++i)
    {
        for (int j = 0; j < newWidth; ++j)
        {
            int sum = 0;
            for (int k = i * resizeFactor; k < (i + 1) * resizeFactor; k++) {
                for (int l = j * resizeFactor; l < (j + 1) * resizeFactor; l++) {
                    sum += image[k * width + l];
                }
            }
            imageResized[i * (newWidth) + j] = sum / (resizeFactor * resizeFactor);
        }
    }
}


// Apply ZNCC algorithm for a given window size and max disparity
void CalcZNCC(const std::vector<unsigned char>& leftImage,
    const std::vector<unsigned char>& rightImage,
    int width, int height,
    int windowSize, int maxDisparity,
    std::vector<int>& disparityMap,
    char isLeftImage = 1
    ) 
{
    int imgSize = width * height;

    int halfWindowSize = (windowSize - 1) / 2;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int bestDisp = 0;
            float bestZNCC = -100.0;
            bool isBorderPixel = false;

            // handle borders | keep bestDisp at 0, so borders will be black
            if (y >= height - halfWindowSize || x >= width - halfWindowSize ||
                y <= halfWindowSize || x <= halfWindowSize)
            {
                isBorderPixel = true;
            }

            if (!isBorderPixel)
            {
                for (int d = 0; d < maxDisparity; d++)
                {
                    float zncc = 0.0;
                    float numerator = 0.0, denominator1 = 0.0, denominator2 = 0.0;
                    float leftMean = 0.0, rightMean = 0.0;

                    // calculate mean for each window - future kernel - changes for different disparities, as the rightmean is calculated based on the disparity
                    int avgCount = 0;
                    for (int winY = -halfWindowSize; winY < halfWindowSize; winY++)
                    {
                        for (int winX = -halfWindowSize; winX < halfWindowSize; winX++)
                        {
                            // don't allow pixel to go to previous row
                            if (d > x + winX)
                            {
                                continue;
                            }

                            int leftPixelIndex = (y + winY) * width + (x + winX);
                            int rightPixelIndex = (y + winY) * width + (x + winX - isLeftImage * d);
                            if (rightPixelIndex >= imgSize ||
                                rightPixelIndex <= 0)
                            {
                                continue;
                            }

                            leftMean += leftImage[leftPixelIndex];
                            rightMean += rightImage[rightPixelIndex];
                            avgCount++;
                        }

                    }
                    leftMean = leftMean / avgCount;
                    rightMean = rightMean / avgCount;

                    for (int winY = -halfWindowSize; winY < halfWindowSize; winY++)
                    {
                        for (int winX = -halfWindowSize; winX < halfWindowSize; winX++)
                        {
                            // don't allow pixel to go to previous row
                            if (d > x + winX)
                            {
                                continue;
                            }

                            int leftPixelIndex = (y + winY) * width + (x + winX);
                            int rightPixelIndex = (y + winY) * width + (x + winX - isLeftImage * d);
                            if (rightPixelIndex >= imgSize ||
                                rightPixelIndex <= 0)
                            {
                                continue;
                            }

                            // calculate zncc value for each window
                            numerator += (leftImage[leftPixelIndex] - leftMean) * (rightImage[rightPixelIndex] - rightMean);
                            denominator1 += pow(leftImage[leftPixelIndex] - leftMean, 2);
                            denominator2 += pow(rightImage[rightPixelIndex] - rightMean, 2);

                        }

                    }

                    float denominator = sqrt(denominator1) * sqrt(denominator2);
                    if (denominator == 0) {
                        break;
                    }

                    zncc = numerator / denominator;
                    if (zncc > bestZNCC)
                    {
                        bestZNCC = zncc;
                        bestDisp = d;
                    }
                }
            }

            disparityMap[y * width + x] = bestDisp;
        }
    }
}

void CrossCheck(const std::vector<int>& dispMapLeft, const std::vector<int>& dispMapRight, const int& width, const int& height, const int& crossDiff, std::vector<int>& crossDispMap)
{
    // Loop over all pixels inside the image boundary
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width ; x++) {

            // Get the disparity values for the current pixel in both directions
            int dispLeft = dispMapLeft[y * width + x];
            int dispRight = dispMapRight[y * width + x];

            // Check if the disparity values agree | abs used to account for rounding errors
            if (std::abs(dispLeft - dispRight) <= crossDiff) {
                // If the disparities agree, use the left disparity value as the final disparity for the pixel
                crossDispMap[y * width + x] = dispLeft;
            }
            else {
                // Otherwise, mark the pixel as invalid
                crossDispMap[y * width + x] = 0;
            }
        }
    }
}

void OcclusionFilling(const std::vector<int>& dispMap, const int& width, const int& height, const int& nCount, std::vector<int>& dispMapFilled)
{
    // Copy the input disparity map to the output disparity map
    std::copy(dispMap.begin(), dispMap.end(), dispMapFilled.begin());

    // Loop over all pixels inside the image boundary
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            // handle borders | keep bestDisp at 0, so borders will stay black
            if (y >= height - (nCount / 2) || x >= width - (nCount / 2)  ||
                y <= nCount / 2 || x <= nCount / 2)
            {
                continue;
            }

            // Check if the current pixel is marked as invalid
            if (dispMap[y * width + x] == 0) {

                // Initialize the list of valid disparity values in the n-neighborhood of the current pixel
                std::vector<int> neighbors;

                // Loop over the n-neighbors of the current pixel
                for (int dy = -nCount / 2; dy <= nCount / 2; dy++) {
                    for (int dx = -nCount / 2; dx <= nCount/ 2; dx++) {
                        // Skip the center pixel
                        if (dx == 0 && dy == 0) continue;

                        // Get the disparity value for the current neighbor
                        int neighbor_disp = dispMap[(y + dy) * width + (x + dx)];

                        // If the neighbor is valid, add its disparity value to the list
                        if (neighbor_disp > 0) {
                            neighbors.push_back(neighbor_disp);
                        }
                    }
                }

                // If at least one valid disparity value was found in the n-neighborhood,
                // set the current pixel's disparity value to the median of the valid values
                if (!neighbors.empty()) {
                    dispMapFilled[y * width + x] = neighbors[neighbors.size() / 2];
                }
            }
        }
    }
}

void NormalizeToChar(const std::vector<int>& dispMap, const int& width, const int& height, const int& ndisp, std::vector<unsigned char>& normVec)
{
    // Loop over all pixels and normalize the disparity values
    for (int i = 0; i < width * height; i++) {
        normVec[i] = static_cast<unsigned char>(static_cast<float>(dispMap[i]) / ndisp * 255);
    }
}

int main()
{   
    // from calib.txt - downsized
    // each pixel in the downsampled image corresponds to a larger area in the original image
    // reducing the resolution of the images reduces the maximum disparity that can be reliably estimated
    // Disparity value represents the number of pixels that one image point is shifted relative to the other image point,
    // so the maximum disparity value is directly related to the image resolution.
    // To account for this reduction in resolution, we need to adjust the maximum disparity value by the same factor that we used to downsample the image

    int ndisp = 260;
    unsigned int resize_factor = 4;
    int win_size = 11;
    int neighbours = 32;
    int crossDiff = 32;

    // setup inputs and outputs
    const char* leftImgName = "../img/im0.png";
    const char* rightImgName = "../img/im1.png";

    const char* depthmapOut = "../img/depthmap.png";

    // create containers for raw images
    std::vector<unsigned char> leftImage;
    std::vector<unsigned char> rightImage;
    unsigned int width, height;

    // decode images
    unsigned int error = lodepng::decode(leftImage, width, height, leftImgName, LCT_RGBA, 8);
    if (error) std::cout << "decoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

    error = lodepng::decode(rightImage, width, height, rightImgName, LCT_RGBA, 8);
    if (error) std::cout << "decoder error second image: " << error << ": " << lodepng_error_text(error) << std::endl;

    // start timing execution time
    LARGE_INTEGER start, end, frequency;
    double elapsed_time;

    QueryPerformanceFrequency(&frequency);

    QueryPerformanceCounter(&start);

    // convert image to grayscale, ignoring the alpha channel
    std::vector<unsigned char> leftImageGray(width * height);
    std::vector<unsigned char> rightImageGray(width * height);
    GrayScaleImageConversion(leftImage, width, height, leftImageGray);
    GrayScaleImageConversion(rightImage, width, height, rightImageGray);

    // resize image
    std::vector<unsigned char> leftImageResized(width * height);
    std::vector<unsigned char> rightImageResized(width * height);
    ResizeImage(leftImageGray, width, height, resize_factor, leftImageResized);
    ResizeImage(rightImageGray, width, height, resize_factor, rightImageResized);

    // update values depending on resolution
    int oldWidth = width;
    width = width / resize_factor;
    height = height / resize_factor;
    ndisp = ndisp * (static_cast<float>(width) / oldWidth);

    // apply zncc
    std::vector<int> leftImageDisparity(width * height);
    std::vector<int> rightImageDisparity(width * height);
    CalcZNCC(leftImageResized, rightImageResized, width, height, win_size, ndisp, leftImageDisparity);
    CalcZNCC(rightImageResized, leftImageResized, width, height, win_size, ndisp, rightImageDisparity, -1);

    // CrossChecking
    std::vector<int> crossCheckedMap(width * height);
    CrossCheck(leftImageDisparity, rightImageDisparity, width, height, crossDiff, crossCheckedMap);

    // occlusion filling
    std::vector<int> oclussionFilledMap(width * height);
    OcclusionFilling(crossCheckedMap, width, height, neighbours, oclussionFilledMap);

    // normalization to 8 bit
    std::vector<unsigned char> depthmapNormalized(width * height);
    NormalizeToChar(oclussionFilledMap, width, height, ndisp, depthmapNormalized);

    // end execution timing and print
    QueryPerformanceCounter(&end);

    elapsed_time = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    std::cout << "Elapsed time: " << elapsed_time << " seconds\n";

    std::cout << "Elapsed time: " << elapsed_time / 60 << " minutes\n";

    // encode resized and grayscaled images (im*_out)
    error = lodepng::encode(depthmapOut, depthmapNormalized, width, height, LCT_GREY, 8);
    if (error) std::cout << "encoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;
}