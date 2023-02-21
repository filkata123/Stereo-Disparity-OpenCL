#include <vector>
#include <iostream>

#include <lodepng.h>

std::vector<unsigned char> rgb_to_grayscale(std::vector<unsigned char> image, unsigned int width, unsigned int height)
{
	std::vector<unsigned char> imageGray(width * height);
	size_t count = 0;
	for (size_t i = 0; i < image.size(); i += 4)
	{
		imageGray[count++] = (image[i + 0] + image[i + 1] + image[i + 2]) / 3;
	}

	return imageGray;
}

std::vector<unsigned char> image_resize_16(std::vector<unsigned char> image, unsigned int width, unsigned int height)
{
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

int decode()
{
	const char* firstImgName = "img/im0.png";
	const char* secondImgName = "img/im1.png";

	const char* firstImgNameOut = "img/im0_out.png";
	const char* secondImgNameOut = "img/im1_out.png";

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


	// encode images
	error = lodepng::encode(firstImgNameOut, firstImageGrayResized, width / 4, height / 4, LCT_GREY, 8);
	if (error) std::cout << "encoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

	error = lodepng::encode(secondImgNameOut, secondImageGrayResized, width / 4, height / 4, LCT_GREY, 8);
	if (error) std::cout << "encoder error second image: " << error << ": " << lodepng_error_text(error) << std::endl;

	return 1;
}