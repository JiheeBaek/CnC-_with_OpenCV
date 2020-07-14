#include "hist_func.h"

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_matching(Mat& input, Mat& matched, G* trans_func_eq, G* trans_func_ref, G* trans_func_matching);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ref = imread("ref.jpg", CV_LOAD_IMAGE_COLOR);
	Mat equalized_YUV;
	Mat equalized_ref;

	cvtColor(input, equalized_YUV, CV_RGB2YUV);	// RGB -> YUV
	cvtColor(ref, equalized_ref, CV_RGB2YUV);	// RGB -> YUV

	// split each channel(Y, U, V)
	Mat channels[3];
	split(equalized_YUV, channels);
	Mat Y = channels[0];						// U = channels[1], V = channels[2]

	// split each channel(Y, U, V)
	Mat channels_ref[3];
	split(equalized_ref, channels_ref);
	Mat Y_ref = channels_ref[0];						// U = channels_ref[1], V = channels_ref[2]

	// PDF or transfer function txt files1
	FILE* f_hist_matching_PDF_YUV, * f_PDF_RGB, *f_PDF_ref_RGB;
	FILE* f_trans_func_matching_YUV;

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_PDF_ref_RGB, "PDF_ref_RGB.txt", "w+");
	fopen_s(&f_hist_matching_PDF_YUV, "matched_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_matching_YUV, "trans_func_matching_YUV.txt", "w+");
	
	float** PDF_RGB = cal_PDF_RGB(input);		// PDF of Input image(RGB) : [L][3]
	float* CDF_YUV = cal_CDF(Y);				// CDF of Y channel image
	G trans_func_eq_YUV[L] = { 0 };			// transfer function
	hist_eq(Y, Y, trans_func_eq_YUV, CDF_YUV);		// histogram equalization on Y channel

	float** PDF_ref = cal_PDF_RGB(ref);		// PDF of ref image(RGB) : [L][3]
	float* CDF_ref = cal_CDF(Y_ref);				// CDF of Y_ref channel image
	G trans_func_eq_ref[L] = { 0 };
	hist_eq(Y_ref, Y_ref, trans_func_eq_ref, CDF_ref);	// histogram equalization on Y_ref channel

	Mat matched_YUV=Y.clone();
	G trans_func_matching[L] = { 0 };
	hist_matching(Y, Y, trans_func_eq_YUV, trans_func_eq_ref, trans_func_matching);

	// merge Y, U, V channels
	merge(channels, 3, matched_YUV);

	// YUV -> RGB (use "CV_YUV2RGB" flag)
	cvtColor(matched_YUV, matched_YUV, CV_YUV2RGB);

	// matched PDF (YUV)
	float** matched_PDF_YUV = cal_PDF_RGB(matched_YUV);

	for (int i = 0; i < L; i++) {
		for (int j = 0; j < 3; j++) {
			// write PDF
			fprintf(f_PDF_RGB, "%d\t%d\t%f\n", i, j, PDF_RGB[i][j]);
			fprintf(f_PDF_ref_RGB, "%d\t%d\t%f\n", i, j, PDF_ref[i][j]);
			fprintf(f_hist_matching_PDF_YUV, "%d\t%d\t%f\n", i, j, matched_PDF_YUV[i][j]);
		}

		// write transfer functions
		fprintf(f_trans_func_matching_YUV, "%d\t%d\n", i, trans_func_matching[i]);
	}

	// memory release
	free(PDF_RGB);
	free(CDF_YUV);
	free(PDF_ref);
	free(CDF_ref);
	fclose(f_PDF_RGB);
	fclose(f_hist_matching_PDF_YUV);
	fclose(f_trans_func_matching_YUV);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Matched_YUV", WINDOW_AUTOSIZE);
	imshow("Matched_YUV", matched_YUV);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization
void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}

// histogram matching
void hist_matching(Mat& input, Mat& matched, G* trans_func_eq, G* trans_func_ref, G* trans_func_matching) {
	G temp[L] = { 0 };

	//compute transfer function
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < L; j++) {
			if (i == trans_func_ref[j]) {
				temp[i] = j;
				break;
			}
			if (j == L - 1)
				temp[i] = temp[i - 1] + 1;
		}
	}

	for (int i = 0; i < L; i++)
		trans_func_matching[i] = temp[trans_func_eq[i]];
	
	//perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			matched.at<G>(i, j) = trans_func_matching[input.at<G>(i, j)];
}