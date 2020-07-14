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

Mat sobelfilter(const Mat input);

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
	output = sobelfilter(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);


	waitKey(0);

	return 0;
}


Mat sobelfilter(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N
	int tempa = 0;
	int tempb = 0;

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
	Mat Sx = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat Sy = Sx.t();



	Mat output = Mat::zeros(row, col, input.type());

	float sum1 = 0.0;
	float sum2 = 0.0;
	float sum3 = 0.0;
	float val = 0.0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float ix_0 = 0.0;
			float ix_1 = 0.0;
			float ix_2 = 0.0;
			float iy_0 = 0.0;
			float iy_1 = 0.0;
			float iy_2 = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
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
					ix_0 += Sx.at<int>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[0]);
					ix_1 += Sx.at<int>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[1]);
					ix_2 += Sx.at<int>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[2]);
					iy_0 += Sy.at<int>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[0]);
					iy_1 += Sy.at<int>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[1]);
					iy_2 += Sy.at<int>(a + n, b + n) * (float)(input.at<C>(tempa, tempb)[2]);
				}
			}
			sum1 = sqrt(ix_0 * ix_0 + iy_0 * iy_0);
			sum2 = sqrt(ix_1 * ix_1 + iy_1 * iy_1);
			sum3 = sqrt(ix_2 * ix_2 + iy_2 * iy_2);

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