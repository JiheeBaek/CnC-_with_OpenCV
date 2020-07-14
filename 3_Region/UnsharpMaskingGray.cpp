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

Mat UnsharpMask(const Mat input, int N, float sigmaT, float sigmaS, const char* boundary_proc, float k);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;


	cvtColor(input, input_gray, CV_RGB2GRAY);


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	output = UnsharpMask(input_gray, 1, 1, 1, "zero-paddle", 0.5); //Boundary process: zero-paddle, mirroring, adjustkernel
	//output = UnsharpMask(input_gray, 1, 1, 1, "mirroring", 0.5);
	//output = UnsharpMask(input_gray, 1, 1, 1, "adjustkernel", 0.5);

	namedWindow("Unsharp Masking", WINDOW_AUTOSIZE);
	imshow("Unsharp Masking", output);

	waitKey(0);

	return 0;
}


Mat UnsharpMask(const Mat input, int N, float sigmaT, float sigmaS, const char* boundary_proc, float k) {

	//############################## IMPLEMENT LOW PASS FILTERING_GAUSSIAN FILTER ##############################

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * N + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	denom = 0.0;
	for (int a = -N; a <= N; a++) {  // Denominator in m(s,t)
		for (int b = -N; b <= N; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
			kernel.at<float>(a + N, b + N) = value1;
			denom += value1;
		}
	}

	for (int a = -N; a <= N; a++) {  // Denominator in m(s,t)
		for (int b = -N; b <= N; b++) 
			kernel.at<float>(a + N, b + N) /= denom;
	}


	Mat L = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
			float sum2 = 0.0;
			if (!strcmp(boundary_proc, "zero-paddle")) {
				for (int a = -N; a <= N; a++) {
					for (int b = -N; b <= N; b++) {
						// Gaussian filter with Zero-paddle boundary process:
						kernelvalue = kernel.at<float>(a + N, b + N);
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
						}
					}
				}
				L.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(boundary_proc, "mirroring")) {
				for (int a = -N; a <= N; a++) {
					for (int b = -N; b <= N; b++) {
						// Gaussian filter with "mirroring" process:
						kernelvalue = kernel.at<float>(a + N, b + N);
						if (i + a > row - 1) {  //mirroring for the border pixels
							tempa = i - a;
						}
						else if (i + a < 0) {
							tempa = -(i + a);
						}
						else {
							tempa = i + a;
						}
						if (j + b > col - 1) {
							tempb = j - b;
						}
						else if (j + b < 0) {
							tempb = -(j + b);
						}
						else {
							tempb = j + b;
						}
						sum1 += kernelvalue * (float)(input.at<G>(tempa, tempb));
					}
				}
				L.at<G>(i, j) = (G)sum1;
			}


			else if (!strcmp(boundary_proc, "adjustkernel")) {
				for (int a = -N; a <= N; a++) {
					for (int b = -N; b <= N; b++) {
						kernelvalue = kernel.at<float>(a + N, b + N);
						// Gaussian filter with "adjustkernel" process:
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
							sum2 += kernelvalue;
						}
					}
				}
				L.at<G>(i, j) = (G)(sum1 / sum2);
			}
		}
	}
	//############################## L IS LOW PASS FILTERED OUTPUT ##############################

	Mat kL = k * L;		// L is scaled with k<1
	Mat I_kL = input - kL;		//Subtract output
	Mat output = I_kL / (1 - k);

	return output;
}