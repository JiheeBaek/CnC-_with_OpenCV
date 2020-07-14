#include "hist_func.h"

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_matching(Mat& input, Mat& matched, G* trans_func_1, G* trans_func_2, G* trans_func_matching);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ref = imread("ref.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat ref_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
	cvtColor(ref, ref_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

	Mat equalized = input_gray.clone();
	Mat matched = input_gray.clone();

	// PDF or transfer function txt files
	FILE* f_PDF;
	FILE* f_PDF_ref;
	FILE* f_hist_matching_PDF;
	FILE* f_trans_func_matching;

	fopen_s(&f_PDF, "PDF.txt", "w+");
	fopen_s(&f_PDF_ref, "PDF_ref.txt", "w+");
	fopen_s(&f_hist_matching_PDF, "hist_matching_PDF.txt", "w+");
	fopen_s(&f_trans_func_matching, "trans_func_matching.txt", "w+");
	
	float* PDF = cal_PDF(input_gray);	// PDF of Input image(Grayscale) : [L]
	float* CDF = cal_CDF(input_gray);	// CDF of Input image(Grayscale) : [L]

	G trans_func_eq[L] = { 0 };			// transfer function
	G trans_func_eq_ref[L] = { 0 };
	G trans_func_matching[L] = { 0 };

	hist_eq(input_gray, equalized, trans_func_eq, CDF);					// histogram equalization on grayscale image
	
	Mat equalized_ref = ref_gray.clone();

	float* PDF_ref = cal_PDF(ref_gray);
	float* CDF_ref = cal_CDF(ref_gray);
	
	hist_eq(ref_gray, equalized_ref, trans_func_eq_ref, CDF_ref);			//histogram equalization on reference image
	
	hist_matching(input_gray, matched, trans_func_eq, trans_func_eq_ref, trans_func_matching);		//histogram matching

	float* hist_matching_PDF = cal_PDF(matched);									// matched PDF (grayscale)
	
	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_PDF_ref, "%d\t%f\n", i, PDF_ref[i]);
		fprintf(f_hist_matching_PDF, "%d\t%f\n", i, hist_matching_PDF[i]);

		// write transfer functions
		fprintf(f_trans_func_matching, "%d\t%d\n", i, trans_func_matching[i]);
	}

	// memory release
	free(PDF);
	free(CDF);
	free(PDF_ref);
	free(CDF_ref);

	fclose(f_PDF);
	fclose(f_hist_matching_PDF);
	fclose(f_trans_func_matching);

	////////////////////// Show each image ///////////////////////

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("Matched", WINDOW_AUTOSIZE);
	imshow("Matched", matched);

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
void hist_matching(Mat& input, Mat& matched, G* trans_func_eq, G* trans_func_ref, G*trans_func_matching) {
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