#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
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

Mat laplacian(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);
	output = laplacian(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", output);


	waitKey(0);

	return 0;
}


Mat laplacian(const Mat input) {

	Mat kernel = (Mat_<int>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N
	int tempa = 0;
	int tempb = 0;
	int kernelvalue = 0;


	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
			float sum2 = 0.0;
			float sum3 = 0.0;
			float val1 = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					kernelvalue = kernel.at<int>(a + n, b + n);
					if (i + a > row - 1)
						tempa = i - a;
					else if (i + a < 0)
						tempa = -(i + a);
					else
						tempa = i + a;
					if (j + b > col - 1)
						tempb = j - b;
					else if (j + b < 0)
						tempb = -(j + b);
					else
						tempb = j + b;
					sum1 += kernelvalue * (float)(input.at<C>(tempa, tempb)[0]);
					sum2 += kernelvalue * (float)(input.at<C>(tempa, tempb)[1]);
					sum3 += kernelvalue * (float)(input.at<C>(tempa, tempb)[2]);
				}
			}
			sum1 = abs(sum1)*5;
			sum2 = abs(sum2)*5;
			sum3 = abs(sum3)*5;
	
			if (sum1 > 255)
				sum1 = 255;
			if (sum2 > 255)
				sum2 = 255;
			if (sum3 > 255)
				sum3 = 255;
			output.at<C>(i, j)[0] = (G)sum1;
			output.at<C>(i, j)[1] = (G)sum2;
			output.at<C>(i, j)[2] = (G)sum3;
		}
	}
	return output;
}