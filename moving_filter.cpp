#include <vector>
#include <iostream>

#include <lodepng.h>


// gaussian kernel as seen in https://www.codingame.com/playgrounds/2524/basic-image-manipulation/filtering
const int GAUSSIAN_SIZE = 5;
const float gaussian_filter_matrix[GAUSSIAN_SIZE * GAUSSIAN_SIZE] =
{
	1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f,
	4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,
	6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f,
	4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,
	1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f
};

std::vector<unsigned char> rgb_to_grayscale(std::vector<unsigned char> image, unsigned int width, unsigned int height)
{
	// output image vector will only have one channel for grayscale, so size needs to be equal to resolution
	std::vector<unsigned char> imageGray(width * height);

	// iterate over every four values, as input is 4 channeled RGBA
	char channel = 4;
	for (size_t i = 0; i < image.size(); i += channel)
	{
		// Add up R, G and B values and divide to get grayscale value.
		// We don't care about the A value, so it is not used.
		imageGray[i / channel] = (image[i + 0] + image[i + 1] + image[i + 2]) / 3;
	}
	return imageGray;
}

std::vector<unsigned char> image_resize_16(std::vector<unsigned char> image, unsigned int width, unsigned int height)
{
	// Divide image into 4x4 blocks and then use the average value of said blocks as the value of the new pixel
	std::vector<unsigned char> imageResized((width * height) / 16);
	for (int i = 0; i < height / 4; ++i)
	{
		for (int j = 0; j < width / 4; ++j)
		{
			int sum = 0;
			for (int k = i * 4; k < (i + 1) * 4; k++) {
				for (int l = j * 4; l < (j + 1) * 4; l++) {
					sum += image[k * width + l];
				}
			}
			imageResized[i * (width / 4) + j] = sum / 16;
		}
	}

	return imageResized;
}

std::vector<unsigned char> gaussian_filter(std::vector<unsigned char> image, unsigned int width, unsigned int height)
{
	std::vector<unsigned char> imageGauss(width * height);

	// Gaussian blur with 5x5 moving filter
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
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

					// Add up the values of the pixels around the current one,
					// while applying weights to each pixel based on the the gaussian kernel matrix
					int index = y * width + x;
					sum += gaussian_filter_matrix[k * GAUSSIAN_SIZE + l] * static_cast<float>(image[index]);
				}
			}

			imageGauss[i * width + j] = static_cast<char>(round(sum));
		}
	}

	return imageGauss;
}

int main()
{
	// setup inputs and outputs
	const char* firstImgName = "../img/im0.png";
	const char* secondImgName = "../img/im1.png";

	const char* firstImgNameOut = "../img/im0_out.png";
	const char* secondImgNameOut = "../img/im1_out.png";

	const char* firstImgNameGaussOut = "../img/im0_gauss_out.png";
	const char* secondImgNameGaussOut = "../img/im1_gauss_out.png";

	// create containers for raw images
	std::vector<unsigned char> firstImage;
	std::vector<unsigned char> secondImage;
	unsigned int width, height;

	// decode images
	unsigned int error = lodepng::decode(firstImage, width, height, firstImgName, LCT_RGBA, 8);
	if (error) std::cout << "decoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

	error = lodepng::decode(secondImage, width, height, secondImgName, LCT_RGBA, 8);
	if (error) std::cout << "decoder error second image: " << error << ": " << lodepng_error_text(error) << std::endl;

	//convert image to grayscale, ignoring the alpha channel
	std::vector<unsigned char> firstImageGray = rgb_to_grayscale(firstImage, width, height);
	std::vector<unsigned char> secondImageGray = rgb_to_grayscale(secondImage, width, height);

	//resize image 1/16
	std::vector<unsigned char> firstImageGrayResized = image_resize_16(firstImageGray, width, height);
	std::vector<unsigned char> secondImageGrayResized = image_resize_16(secondImageGray, width, height);

	// update stored resolution
	width = width / 4;
	height = height / 4;

	// encode resized and grayscaled images (im*_out)
	error = lodepng::encode(firstImgNameOut, firstImageGrayResized, width, height, LCT_GREY, 8);
	if (error) std::cout << "encoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

	error = lodepng::encode(secondImgNameOut, secondImageGrayResized, width, height, LCT_GREY, 8);
	if (error) std::cout << "encoder error second image: " << error << ": " << lodepng_error_text(error) << std::endl;

	// apply 5x5 moving filter (gaussian blur) to processed images
	std::vector<unsigned char> firstImageGaussian = gaussian_filter(firstImageGrayResized, width, height);
	std::vector<unsigned char> secondImageGaussian = gaussian_filter(secondImageGrayResized, width, height);

	// encode blurred images (im*_gauss_out)
	error = lodepng::encode(firstImgNameGaussOut, firstImageGaussian, width, height, LCT_GREY, 8);
	if (error) std::cout << "encoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

	error = lodepng::encode(secondImgNameGaussOut, secondImageGaussian, width, height, LCT_GREY, 8);
	if (error) std::cout << "encoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;
}