/*
 * Eona Studio (c) 2015
 */

#ifndef DEMO_MNIST_MNIST_PARSER_H_
#define DEMO_MNIST_MNIST_PARSER_H_

#include "../../utils/global_utils.h"
#include "../../utils/laminar_utils.h"

static constexpr const char* MNIST_TRAIN_IMAGE_FILE = "train-images-idx3-ubyte";
static constexpr const char* MNIST_TRAIN_LABEL_FILE = "train-labels-idx1-ubyte";
static constexpr const char* MNIST_TEST_IMAGE_FILE = "t10k-images-idx3-ubyte";
static constexpr const char* MNIST_TEST_LABEL_FILE = "t10k-labels-idx1-ubyte";

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
 * @param batches set to zero to read the entire database.
 * each vector<FloatT> will be length imagePerBatch * (28*28)
 * @param normalize true to divide everything by 255
 */
template<typename FloatT = float>
inline vector<vector<FloatT>> read_mnist_image(
		string filePath, int batches, int imagePerBatch, bool normalize = true)
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

	LMN_ASSERT_THROW(batches * imagePerBatch <= totalNumberOfImages,
			DataException("MNIST image read exceeds total number of images"));

	if (batches <= 0)
		batches = totalNumberOfImages / imagePerBatch;

    vector<vector<FloatT>> images(batches,
    		vector<FloatT>(imagePerBatch * rowdim * coldim));

    FloatT divisor = normalize ? 255.0 : 1.0;

	for(int b=0; b<batches; ++b)
		for (int i = 0; i < imagePerBatch; ++i)
			for(int r=0; r<rowdim; ++r)
				for(int c=0; c<coldim; ++c)
				{
					unsigned char temp=0;
					file.read((char*)&temp, sizeof(temp));
					images[b][i * (rowdim*coldim) + rowdim * r + c]= (FloatT) temp / divisor;
				}

	return images;
}

/**
 * @param filePath
 * @param batch set to zero to read the entire database
 * each vector<float> is a batch of int labels
 */
template<typename FloatT = float>
inline vector<vector<FloatT>> read_mnist_label(string filePath, int batch, int labelPerBatch)
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

	LMN_ASSERT_THROW(batch * labelPerBatch <= totalNumberOfLabels,
			DataException("MNIST label read exceeds total number of images"));

	if (batch <= 0)
		batch = totalNumberOfLabels / labelPerBatch;

    vector<vector<FloatT>> labels(batch, vector<FloatT>(labelPerBatch));

	for(int b=0; b < batch; ++b)
		for (int i = 0; i < labelPerBatch; ++i)
		{
			unsigned char temp=0;
			file.read((char*)&temp,sizeof(temp));
			labels[b][i]= (float) temp;
		}

	return labels;
}

#endif /* DEMO_MNIST_MNIST_PARSER_H_ */
