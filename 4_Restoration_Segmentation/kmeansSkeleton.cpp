#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif


// Note that this code is for the case when an input data is a color value.
int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;
	int clusterCount = 10;
	int attempts = 5;

	cvtColor(input, input_gray, CV_RGB2GRAY);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original_Gray", WINDOW_AUTOSIZE);
	imshow("Original_Gray", input_gray);

	namedWindow("Original_RGB", WINDOW_AUTOSIZE);
	imshow("Original_RGB", input);

	Mat labels;
	Mat centers;

	//------------------------------------ G R A Y S C A L E -------------------------------------

	Mat samples(input_gray.rows * input_gray.cols, 3, CV_32F);
	for (int y = 0; y < input_gray.rows; y++)
		for (int x = 0; x < input_gray.cols; x++) {
			samples.at<float>(y + x * input_gray.rows, 0) = (double)input_gray.at<G>(y, x) / (double)255;
			samples.at<float>(y + x * input_gray.rows, 1) = (double)y / (double)input_gray.rows;
			samples.at<float>(y + x * input_gray.rows, 2) = (double)x / (double)input_gray.cols;
		}

	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	
	Mat new_image_gray(input_gray.size(), input_gray.type());

	for (int y = 0; y < input_gray.rows; y++)
		for (int x = 0; x < input_gray.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * input_gray.rows, 0);
			new_image_gray.at<G>(y, x) = 255 * centers.at<float>(cluster_idx, 0);
		}
	imshow("clustered image", new_image_gray);


	//----------------------------------------- R G B ------------------------------------------------

	Mat samples_r(input.rows * input.cols, 3, CV_32F);
	Mat samples_g(input.rows * input.cols, 3, CV_32F);
	Mat samples_b(input.rows * input.cols, 3, CV_32F);

	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++) {
			samples_r.at<float>(y + x * input.rows, 0) = (double)input.at<Vec3b>(y, x)[0]/(double)255;
			samples_r.at<float>(y + x * input.rows, 1) = (double)y/(double)input.rows;
			samples_r.at<float>(y + x * input.rows, 2) = (double)x/(double)input.cols;
			
			samples_g.at<float>(y + x * input.rows, 0) = (double)input.at<Vec3b>(y, x)[1]/ (double)255;
			samples_g.at<float>(y + x * input.rows, 1) = (double)y/ (double)input.rows;
			samples_g.at<float>(y + x * input.rows, 2) = (double)x/ (double)input.cols;

			samples_b.at<float>(y + x * input.rows, 0) = (double)input.at<Vec3b>(y, x)[2]/ (double)255;
			samples_b.at<float>(y + x * input.rows, 1) = (double)y/ (double)input.rows;
			samples_b.at<float>(y + x * input.rows, 2) = (double)x/ (double)input.cols;
		}
	
	// Clustering is performed for each channel (RGB)
	// Note that the intensity value is not normalized here (0~1). You should normalize both intensity and position when using them simultaneously.
	
	Mat labels_r;
	Mat labels_g;
	Mat labels_b;

	Mat centers_r;
	Mat centers_g;
	Mat centers_b;
	
	kmeans(samples_r, clusterCount, labels_r, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers_r);
	kmeans(samples_g, clusterCount, labels_g, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers_g);
	kmeans(samples_b, clusterCount, labels_b, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers_b);

	Mat new_image(input.size(), input.type());

	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			int cluster_idx_R = labels_r.at<int>(y + x * input.rows, 0);
			int cluster_idx_G = labels_g.at<int>(y + x * input.rows, 0);
			int cluster_idx_B = labels_b.at<int>(y + x * input.rows, 0);

			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
			new_image.at<C>(y, x)[0] = 255*centers_r.at<float>(cluster_idx_R, 0);
			new_image.at<C>(y, x)[1] = 255*centers_g.at<float>(cluster_idx_G, 0);
			new_image.at<C>(y, x)[2] = 255*centers_b.at<float>(cluster_idx_B, 0);
		}
	imshow("clustered image RGB", new_image);

	waitKey(0);

	return 0;
}

