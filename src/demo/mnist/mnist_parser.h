/*
 * Eona Studio (c) 2015
 */

#ifndef DEMO_MNIST_MNIST_PARSER_H_
#define DEMO_MNIST_MNIST_PARSER_H_

#include "../../utils/global_utils.h"
#include "../../utils/laminar_utils.h"

inline int reverse_int(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

/**
 * @param filePath
 * @param numberOfImages set to zero to read the entire database
 * @param normalize true to divide everything by 255
 */
template<typename FloatT = float>
inline vector<vector<FloatT>> read_mnist_image(
		string filePath, int numberOfImages, bool normalize = true)
{
    std::ifstream file(filePath, std::ios::binary);

    LMN_ASSERT_THROW(file.is_open(),
			DataException("MNIST image file not found"));

	int magicNumber=0;
	int totalNumberOfImages=0;
	int rowdim=0;
	int coldim=0;

	auto read_reverse_int = [&](int& number) {
		file.read((char*) &number, sizeof(int));
		number = reverse_int(number);
	};

	read_reverse_int(magicNumber);
	read_reverse_int(totalNumberOfImages);
	read_reverse_int(rowdim);
	read_reverse_int(coldim);

	LMN_ASSERT_THROW(numberOfImages <= totalNumberOfImages,
			DataException("MNIST image read exceeds total number of images"));

	if (numberOfImages <= 0)
		numberOfImages = totalNumberOfImages;

    vector<vector<float>> images(numberOfImages,vector<FloatT>(rowdim * coldim));

    FloatT divisor = normalize ? 255.0 : 1.0;

	for(int i=0; i<numberOfImages; ++i)
		for(int r=0;r<rowdim;++r)
			for(int c=0;c<coldim;++c)
			{
				unsigned char temp=0;
				file.read((char*)&temp, sizeof(temp));
				images[i][rowdim * r + c]= (FloatT) temp / divisor;
			}

	return images;
}

/**
 * @param filePath
 * @param numberOfLabels set to zero to read the entire database
 */
inline vector<int> read_mnist_label(string filePath, int numberOfLabels)
{
    std::ifstream file(filePath, std::ios::binary);

    LMN_ASSERT_THROW(file.is_open(),
			DataException("MNIST label file not found"));

	int magicNumber=0;
	int totalNumberOfLabels=0;

	auto read_reverse_int = [&](int& number) {
		file.read((char*) &number, sizeof(int));
		number = reverse_int(number);
	};

	read_reverse_int(magicNumber);
	read_reverse_int(totalNumberOfLabels);

	LMN_ASSERT_THROW(numberOfLabels <= totalNumberOfLabels,
			DataException("MNIST label read exceeds total number of images"));

	if (numberOfLabels <= 0)
		numberOfLabels = totalNumberOfLabels;

    vector<int> labels(numberOfLabels);


	for(int i=0; i < numberOfLabels; ++i)
	{
		unsigned char temp=0;
		file.read((char*)&temp,sizeof(temp));
		labels[i]= (int) temp;
	}

	return labels;
}

#endif /* DEMO_MNIST_MNIST_PARSER_H_ */
