#include "Header.h"

//8 micron values
const int MASKX = 554;
const int MASKY = 79;
const int TEMPMASKX = 543;
const int TEMPMASKY = 74;

const float Euler = 2.71828;

//15 micron values
//const int MASKX = 299;
//const int MASKY = 47;
//const int TEMPMASKX = 290;
//const int TEMPMASKY = 40;

void multiplyOwn(Mat& a, Mat& b, Mat& dst) {

	int event = 0;
	Mat aPrime = a.clone();
	Mat bPrime = b.clone();

	int height = a.rows;
	int width = a.cols;

	if (a.type() == 0 && b.type() == 0) {
		dst.create(height, width, CV_8UC1);
		event = 1;
	}
	else if (a.type() == 5 && b.type() == 0) {
		bPrime.convertTo(bPrime, CV_32FC1);
		dst.create(height, width, CV_32FC1);
		event = 2;
	}
	else if (a.type() == 0 && b.type() == 5) {
		aPrime.convertTo(aPrime, CV_32FC1);
		dst.create(height, width, CV_32FC1);
		event = 2;
	}
	else {
		dst.create(height, width, CV_32FC1);
		event = 2;
	}

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (event == 1) {
				dst.at<uchar>(i, j) = bPrime.at<uchar>(i, j) * aPrime.at<uchar>(i, j);
			}
			else if (event == 2) {
				dst.at<float>(i, j) = bPrime.at<float>(i, j) * aPrime.at<float>(i, j);
			}
			else {
				cout << "No Event Specified" << endl;
			}
		}
	}
}

void subtractOwn(Mat& a, Mat& b, Mat& dst) {

	int event = 0;
	Mat aPrime = a.clone();
	Mat bPrime = b.clone();

	int height = a.rows;
	int width = a.cols;

	if (a.type() == 0 && b.type() == 0) {
		dst.create(height, width, CV_8UC1);
		event = 1;
	}
	else if (a.type() == 5 && b.type() == 0) {
		bPrime.convertTo(bPrime, CV_32FC1);
		dst.create(height, width, CV_32FC1);
		event = 2;
	}
	else if (a.type() == 0 && b.type() == 5) {
		aPrime.convertTo(aPrime, CV_32FC1);
		dst.create(height, width, CV_32FC1);
		event = 2;
	}
	else {
		dst.create(height, width, CV_32FC1);
		event = 2;
	}

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (event == 1) {
				dst.at<uchar>(i, j) = aPrime.at<uchar>(i, j) - bPrime.at<uchar>(i, j);
			}
			else if (event == 2) {
				dst.at<float>(i, j) = aPrime.at<float>(i, j) - bPrime.at<float>(i, j);
			}
			else {
				cout << "No Event Specified" << endl;
			}
		}
	}
}

void nullifyNegatives(Mat& src, Mat& dst)
{
	int height = src.rows;
	int width = src.cols;
	int event = 0;

	if (src.type() == 5) {
		dst.create(height, width, CV_32FC1);
		event = 2;
	}
	else if (src.type() == 0) {
		dst.create(height, width, CV_8UC1);
		event = 1;
	}
	else {
		return;
	}

	if (event == 2) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				float temp = src.at<float>(i, j);
				if (temp < 0) {
					dst.at<float>(i, j) = 0;
				}
				else {
					dst.at<float>(i, j) = temp;
				}
			}
		}
	}
	else {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar temp = src.at<uchar>(i, j);
				if (temp < 0) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = temp;
				}
			}
		}
	}
	return;
	cv::threshold(-src, dst, 0, 0, THRESH_TRUNC);
	dst = -dst;
	return;
}

void controlledNormalize(Mat& src, Mat& dst, float k, int midpoint) {
	int event = 0;
	Mat srcPrime = src.clone();

	int height = src.rows;
	int width = src.cols;

	if (src.type() == 0) {
		srcPrime.convertTo(srcPrime, CV_32FC1);
	}

	dst.create(height, width, CV_32FC1);

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			float x = srcPrime.at<float>(i, j);
			if (x > 3.0) {
				dst.at<float>(i, j) = 255 / (1 + pow(Euler, -k * (x - midpoint)));
			}
			else {
				dst.at<float>(i, j) = 0.0;
			}
		}
	}
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

//all inputs must be 32F
void pickingPoints(Mat& q, Mat& guidedR1, Mat& dst, Mat EdgeImg2pt) {
	int i, j;
	int count = 0;
	int height = q.rows;
	int width = q.cols;
	dst.create(height, width, CV_32FC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			float temp = EdgeImg2pt.at<float>(i, j);
			if (temp >= 5) {
				float guidedPercentage = temp / 255;
				float guidedR1_percentage = 1.0 - guidedPercentage;

				dst.at<float>(i, j) = q.at<float>(i, j) * guidedPercentage +
					255 * guidedR1_percentage;
			}
			else {
				if (guidedR1.at<float>(i, j) > 128) {
					dst.at<float>(i, j) = 255.0;
				}
				else {
					dst.at<float>(i, j) = 0;
				}
			}
		}
	}
}

void correlationPadding(Mat& src, Mat& templ, Mat& dst) {
	//catching if image cannot be read
	if (src.empty())
	{
		cout << "Can't read src image" << endl;
		return;
	}

	if (templ.empty())
	{
		cout << "Can't read template image" << endl;
		return;
	}

	Mat result;

	//make input image copy
	int result_cols = src.cols - templ.cols + 1;
	int result_rows = src.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);

	//Template Matching and Normalizing
	matchTemplate(src, templ, result, TM_CCORR);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	//Finding global max and min
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	//get match location to be matchLoc
	matchLoc = maxLoc;
	cout << "Location for pic is: " << matchLoc << endl;

	//padding the obtained mask with zeroes
	Mat padded;
	copyMakeBorder(templ, padded, matchLoc.y, MASKY - matchLoc.y - TEMPMASKY, matchLoc.x, MASKX - TEMPMASKX - matchLoc.x, BORDER_CONSTANT, Scalar(0));
	padded.convertTo(padded, CV_8UC1);
	dst = padded;
}

void whitify(Mat& q) {
	int event;
	if (q.type() == 0) {
		event = 1;
	}
	else {
		event = 2;
	}

	int i, j;
	int height = q.rows;
	int width = q.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (event == 1) {
				if (q.at<uchar>(i, j) > 1) {
					q.at<uchar>(i, j) = 255;
				}
			}
			else {
				if (q.at<float>(i, j) > 1) {
					q.at<float>(i, j) = 255;
				}
			}
		}
	}

}

//void flattenNoise(Mat& diffMap, unordered_set <Point> pointSet) {
//
//	int i, j;
//	int height = diffMap.rows;
//	int width = diffMap.cols;
//
//	for (const auto& point : pointSet) {
//		diffMap.at<uchar>(point.x, point.y) = 0;
//	}
//}
void flattenNoise(Mat& diffMap, Point* pointArr) {

	int height = diffMap.rows;
	int width = diffMap.cols;
	int arrSize = sizeof(pointArr) / sizeof(pointArr[0]);

	for (int i = 0; i < arrSize; i++) {
		diffMap.at<uchar>(pointArr[i].x, pointArr[i].y) = 0;
	}
}

//cross checking the edge of smudged Guided Filter
Mat crossAndDiagCheck(Mat& smudgedGF, Mat& difference, Mat edge) {

	Mat dst = smudgedGF.clone();

	int thresCross = 80;
	int range = 35;
	int thresDiag = 100;
	int rangeDiag = 50;

	const int height = dst.rows;
	const int width = dst.cols;

	unordered_set <Point> pointSet;
	/*iterate through pointSet
	unordered_set<string> ::iterator itr;
	for (itr = CBA.begin(); itr != CBA.end(); itr++)
		cout << (*itr) << endl;
	}*/
	//vector<Point> arr;
	//arr.resize(1000);

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {

			uchar curr = difference.at<uchar>(i, j);

			//CROSS CHECK
			if (curr <= thresCross) {
				continue;
			}
			else if (edge.at<uchar>(i, j) == 255) {
				//cross check occurs
				//only if it is brighter
				//if eiter pixel in the cross out of range, remove from smudged in dst
				if (curr - difference.at<uchar>(i + 1, j) >= range && curr - difference.at<uchar>(i - 1, j) >= range &&
					curr - difference.at<uchar>(i, j + 1) >= range && curr - difference.at<uchar>(i, j - 1) >= range) {
					dst.at<uchar>(i, j) = 0;
					pointSet.insert(Point(i, j));
				}
			}

			//DIAGONAL CHECK
			if (curr <= thresDiag) {
				continue;
			}
			else if (edge.at<uchar>(i, j) == 255) {
				//diagonal check occurs
				//only if it is brighter
				//if eiter pixel in the cross out of range, remove from smudged in dst
				if (curr - difference.at<uchar>(i + 1, j + 1) >= rangeDiag && curr - difference.at<uchar>(i + 1, j - 1) >= rangeDiag &&
					curr - difference.at<uchar>(i - 1, j + 1) >= rangeDiag && curr - difference.at<uchar>(i - 1, j - 1) >= rangeDiag) {
					dst.at<uchar>(i, j) = 0;
					pointSet.insert(Point(i, j));
				}
			}
		}
	}
	return dst;
}

//unordered_set <Point> crossDiagCheck(Mat& smudgedGF, Mat& difference, Mat edge) {
//
//	Mat dst = smudgedGF.clone();
//
//	int thresCross = 80;
//	int range = 35;
//	int thresDiag = 100;
//	int rangeDiag = 50;
//
//	const int height = dst.rows;
//	const int width = dst.cols;
//
//	unordered_set <Point> pointSet;
//	/*iterate through pointSet
//	unordered_set<string> ::iterator itr;
//	for (itr = CBA.begin(); itr != CBA.end(); itr++)
//		cout << (*itr) << endl;
//	}*/
//	//vector<Point> arr;
//	//arr.resize(1000);
//
//	int i, j;
//
//	for (i = 0; i < height; i++) {
//		for (j = 0; j < width; j++) {
//
//			uchar curr = difference.at<uchar>(i, j);
//
//			//CROSS CHECK
//			if (curr <= thresCross) {
//				continue;
//			}
//			else if (edge.at<uchar>(i, j) == 255) {
//				//cross check occurs
//				//only if it is brighter
//				//if eiter pixel in the cross out of range, remove from smudged in dst
//				if (curr - difference.at<uchar>(i + 1, j) >= range && curr - difference.at<uchar>(i - 1, j) >= range &&
//					curr - difference.at<uchar>(i, j + 1) >= range && curr - difference.at<uchar>(i, j - 1) >= range) {
//					dst.at<uchar>(i, j) = 0;
//					pointSet.insert(Point(i, j));
//				}
//			}
//
//			//DIAGONAL CHECK
//			if (curr <= thresDiag) {
//				continue;
//			}
//			else if (edge.at<uchar>(i, j) == 255) {
//				//diagonal check occurs
//				//only if it is brighter
//				//if eiter pixel in the cross out of range, remove from smudged in dst
//				if (curr - difference.at<uchar>(i + 1, j + 1) >= rangeDiag && curr - difference.at<uchar>(i + 1, j - 1) >= rangeDiag &&
//					curr - difference.at<uchar>(i - 1, j + 1) >= rangeDiag && curr - difference.at<uchar>(i - 1, j - 1) >= rangeDiag) {
//					dst.at<uchar>(i, j) = 0;
//					pointSet.insert(Point(i, j));
//				}
//			}
//		}
//	}
//	return pointSet;
//}
void crossDiagCheck(Mat& smudgedGF, Mat& difference, Mat edge, Point* arr) {

	Mat dst = smudgedGF.clone();

	int thresCross = 80;
	int range = 35;
	int thresDiag = 100;
	int rangeDiag = 50;

	const int height = dst.rows;
	const int width = dst.cols;

	int count = 0;

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {

			uchar curr = difference.at<uchar>(i, j);

			//CROSS CHECK
			if (curr <= thresCross) {
				continue;
			}
			else if (edge.at<uchar>(i, j) == 255) {
				//cross check occurs
				//only if it is brighter
				//if eiter pixel in the cross out of range, remove from smudged in dst
				if (curr - difference.at<uchar>(i + 1, j) >= range && curr - difference.at<uchar>(i - 1, j) >= range &&
					curr - difference.at<uchar>(i, j + 1) >= range && curr - difference.at<uchar>(i, j - 1) >= range) {
					dst.at<uchar>(i, j) = 0;
					arr[count] = Point(i, j);
					count++;
				}
			}

			//DIAGONAL CHECK
			if (curr <= thresDiag) {
				continue;
			}
			else if (edge.at<uchar>(i, j) == 255) {
				//diagonal check occurs
				//only if it is brighter
				//if eiter pixel in the cross out of range, remove from smudged in dst
				if (curr - difference.at<uchar>(i + 1, j + 1) >= rangeDiag && curr - difference.at<uchar>(i + 1, j - 1) >= rangeDiag &&
					curr - difference.at<uchar>(i - 1, j + 1) >= rangeDiag && curr - difference.at<uchar>(i - 1, j - 1) >= rangeDiag) {
					dst.at<uchar>(i, j) = 0;
					arr[count] = Point(i, j);
					count++;
				}
			}
		}
	}
	//return arr;
}

Mat crossCheck(Mat& smudgedGF, Mat& difference, Mat edge, int threshold, int range) {

	Mat dst = smudgedGF.clone();

	int height = dst.rows;
	int width = dst.cols;

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {

			uchar curr = difference.at<uchar>(i, j);
			if (difference.at<uchar>(i, j) <= threshold) {
				continue;
			}
			else if (edge.at<uchar>(i, j) == 255) {
				//cross check occurs
				//only if it is brighter
				//if eiter pixel in the cross out of range, remove from smudged in dst
				if (curr - difference.at<uchar>(i + 1, j) >= range && curr - difference.at<uchar>(i - 1, j) >= range &&
				    curr - difference.at<uchar>(i, j + 1) >= range && curr - difference.at<uchar>(i, j - 1) >= range) {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return dst;
}

Mat diagCheck(Mat& smudgedGF, Mat& difference, Mat edge, int threshold, int range) {

	Mat dst = smudgedGF.clone();

	int height = dst.rows;
	int width = dst.cols;

	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {

			uchar curr = difference.at<uchar>(i, j);
			if (difference.at<uchar>(i, j) <= threshold) {
				continue;
			}
			else if (edge.at<uchar>(i, j) == 255) {
				//diagonal check occurs
				//only if it is brighter
				//if eiter pixel in the cross out of range, remove from smudged in dst
				if (curr - difference.at<uchar>(i + 1, j + 1) >= range && curr - difference.at<uchar>(i + 1, j - 1) >= range &&
				    curr - difference.at<uchar>(i - 1, j + 1) >= range && curr - difference.at<uchar>(i - 1, j - 1) >= range) {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return dst;
}

Mat deepGFSelection(Mat& smudgedGF, Mat& deepgf) {

	int height = deepgf.rows;
	int width = deepgf.cols;

	Mat dst;
	dst.create(height, width, CV_8UC1);
	int i, j;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (smudgedGF.at<uchar>(i, j) < 128) {
				dst.at<uchar>(i, j) = 0;
			}
			else {
				dst.at<uchar>(i, j) = deepgf.at<uchar>(i, j);
			}
		}
	}
	return dst;
}

//original edge guided filter
Mat guidedFilter(Mat& I, Mat& p, int r, float eps)
{
	Size ksize(2 * r + 1, 2 * r + 1);
	Mat Guide = I.clone();
	Mat Input = p.clone();

	//Step 1:
	//mean_I, mean_p, corr_I, corr_Ip
	Mat mean_p;
	Mat mean_I;
	Mat corr_Ip;
	Mat corr_I;

	boxFilter(Guide, mean_I, CV_32F, ksize);
	boxFilter(Input, mean_p, CV_32F, ksize);
	I.convertTo(Guide, CV_32FC1);
	p.convertTo(Input, CV_32FC1);

	Mat tmpIp;
	multiplyOwn(Guide, Input, tmpIp);
	boxFilter(tmpIp, corr_Ip, CV_32F, ksize);

	Mat_<double> mean_II, tmpII;
	tmpII = Guide.mul(Guide);
	boxFilter(tmpII, corr_I, CV_32F, ksize);


	//Step 2:
	//var_I  - variance of I in each local patch: the matrix Sigma in Eqn (14)
	//cov_Ip - covariance of (I, p) in each local patch 
	Mat var_I;
	Mat tmp_II;
	multiplyOwn(mean_I, mean_I, mean_II);
	subtractOwn(corr_I, mean_II, var_I);

	Mat cov_Ip;
	Mat mean_Ip;
	multiplyOwn(mean_I, mean_p, mean_Ip);
	cov_Ip = corr_Ip - mean_Ip;


	//Step 3:
	//compute a and b
	Mat a(Input.rows, Input.cols, CV_MAKETYPE(CV_32F, 1));
	Mat b(Input.rows, Input.cols, CV_32F);
	divide(cov_Ip, var_I + eps, a);

	Mat aMulmean_I;
	multiplyOwn(a, mean_I, aMulmean_I);
	b = mean_p - aMulmean_I;


	//Step 5:
	//find mean_a and mean_b
	Mat mean_a;
	Mat mean_b;
	boxFilter(a, mean_a, CV_32F, ksize);
	boxFilter(b, mean_b, CV_32F, ksize);
	Mat aI;
	multiplyOwn(mean_a, Guide, aI);

	Mat q, dst;
	q = aI + mean_b;

	//only taking q points of the edges
	q.convertTo(q, CV_8UC1);
	return q;
}

Mat fastGuidedFilter(Mat& I, Mat& p, int r, float eps, int scale)
{

	



	Size ksize(2 * r + 1, 2 * r + 1);
	Mat Guide = I.clone();
	Mat Input = p.clone();

	//Step 1:
	//mean_I, mean_p, corr_I, corr_Ip
	Mat mean_p;
	Mat mean_I;
	Mat corr_Ip;
	Mat corr_I;

	boxFilter(Guide, mean_I, CV_32F, ksize);
	boxFilter(Input, mean_p, CV_32F, ksize);
	I.convertTo(Guide, CV_32FC1);
	p.convertTo(Input, CV_32FC1);

	Mat tmpIp;
	multiplyOwn(Guide, Input, tmpIp);
	boxFilter(tmpIp, corr_Ip, CV_32F, ksize);

	Mat_<double> mean_II, tmpII;
	tmpII = Guide.mul(Guide);
	boxFilter(tmpII, corr_I, CV_32F, ksize);


	//Step 2:
	//var_I  - variance of I in each local patch: the matrix Sigma in Eqn (14)
	//cov_Ip - covariance of (I, p) in each local patch 
	Mat var_I;
	Mat tmp_II;
	multiplyOwn(mean_I, mean_I, mean_II);
	subtractOwn(corr_I, mean_II, var_I);

	Mat cov_Ip;
	Mat mean_Ip;
	multiplyOwn(mean_I, mean_p, mean_Ip);
	cov_Ip = corr_Ip - mean_Ip;


	//Step 3:
	//compute a and b
	Mat a(Input.rows, Input.cols, CV_MAKETYPE(CV_32F, 1));
	Mat b(Input.rows, Input.cols, CV_32F);
	divide(cov_Ip, var_I + eps, a);

	Mat aMulmean_I;
	multiplyOwn(a, mean_I, aMulmean_I);
	b = mean_p - aMulmean_I;


	//Step 5:
	//find mean_a and mean_b
	Mat mean_a;
	Mat mean_b;
	boxFilter(a, mean_a, CV_32F, ksize);
	boxFilter(b, mean_b, CV_32F, ksize);
	Mat aI;
	multiplyOwn(mean_a, Guide, aI);

	Mat q, dst;
	q = aI + mean_b;

	//only taking q points of the edges
	q.convertTo(q, CV_8UC1);
	return q;
}

Mat noMeanGuidedFilter(Mat& I, Mat& p, int r, float eps)
{
	Size ksize(2 * r + 1, 2 * r + 1);
	Mat Guide = I.clone();
	Mat Input = p.clone();

	//Step 1:
	//mean_I, mean_p, corr_I, corr_Ip
	Mat mean_p;
	Mat mean_I;
	Mat corr_Ip;
	Mat corr_I;

	boxFilter(Guide, mean_I, CV_32F, ksize);
	boxFilter(Input, mean_p, CV_32F, ksize);
	I.convertTo(Guide, CV_32FC1);
	p.convertTo(Input, CV_32FC1);

	Mat tmpIp;
	multiplyOwn(Guide, Input, tmpIp);
	boxFilter(tmpIp, corr_Ip, CV_32F, ksize);

	Mat_<double> mean_II, tmpII;
	tmpII = Guide.mul(Guide);
	boxFilter(tmpII, corr_I, CV_32F, ksize);


	//Step 2:
	//var_I  - variance of I in each local patch: the matrix Sigma in Eqn (14)
	//cov_Ip - covariance of (I, p) in each local patch 

	Mat var_I;
	Mat tmp_II;
	multiplyOwn(mean_I, mean_I, mean_II);
	subtractOwn(corr_I, mean_II, var_I);

	Mat cov_Ip;
	Mat mean_Ip;
	multiplyOwn(mean_I, mean_p, mean_Ip);
	cov_Ip = corr_Ip - mean_Ip;


	//Step 3:
	//compute a and b

	Mat a(Input.rows, Input.cols, CV_MAKETYPE(CV_32F, 1));
	Mat b(Input.rows, Input.cols, CV_32F);
	divide(cov_Ip, var_I + eps, a);

	Mat aMulmean_I;
	multiplyOwn(a, mean_I, aMulmean_I);
	a = a * 1.0;
	b = mean_p - aMulmean_I;


	//getting output without mean
	Mat aI;
	multiplyOwn(a, Guide, aI);

	Mat q, dst;
	q = aI + b;
	q.convertTo(q, CV_8UC1);
	return q;
}

Mat edgeGuidedFilter(Mat& I, Mat& p, int r, float eps)
{
	Mat imgDia, results_32, resultsR2_32, imgDia_32, dst;
	Mat guidedR1 = guidedFilter(I, p, 3, 0.1);

	//thickening canny edge by 1
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(guidedR1, imgDia, kernel);


	//Getting guided 2pt edge
	guidedR1.convertTo(results_32, CV_32FC1);
	imgDia.convertTo(imgDia_32, CV_32FC1);
	whitify(imgDia_32);

	Mat guidedEdge2pt = imgDia_32 - results_32;

	//Nullifying edges
	Mat nullifiedGuidedEdge2pt;
	nullifyNegatives(guidedEdge2pt, nullifiedGuidedEdge2pt);

	//only taking q points of the edges
	Mat guidedR2 = guidedFilter(I, p, r, eps);
	guidedR2.convertTo(resultsR2_32, CV_32FC1);
	pickingPoints(resultsR2_32, results_32, dst, nullifiedGuidedEdge2pt);
	dst.convertTo(dst, CV_8UC1);
	return dst;
}

Mat normedEdgeGuidedFilter(Mat& I, Mat& p, int r, float eps, int k, int midpoint) {

	Mat temp, temp2;
	Mat erodedbetterEdgeMask, normedErodedbetterEdgeMask;

	Mat resultsSource = edgeGuidedFilter(I, p, r, eps);
	//imshow("edge guided", resultsSource);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(resultsSource, temp, kernel);
	//imshow("eroded", temp);
	boxFilter(temp, temp, CV_8UC1, Size(7, 7));
	temp2 = (resultsSource) / 255;
	multiplyOwn(temp, temp2, erodedbetterEdgeMask);
	erodedbetterEdgeMask.convertTo(erodedbetterEdgeMask, CV_8UC1);
	//imshow("weighted", erodedbetterEdgeMask);

	controlledNormalize(erodedbetterEdgeMask, normedErodedbetterEdgeMask, k * 0.01, midpoint);
	normedErodedbetterEdgeMask.convertTo(normedErodedbetterEdgeMask, CV_8UC1);

	return normedErodedbetterEdgeMask;
}

Mat smudgedEdgeGuidedFilter(Mat& I, Mat& p, int r, float eps) {

	Mat temp, temp2;
	Mat betterEdgeMask, normedbetterEdgeMask;

	Mat resultsSource = edgeGuidedFilter(I, p, r, eps);
	//imshow("edge guided", resultsSource);

	temp2 = (resultsSource) / 255;
	boxFilter(resultsSource, resultsSource, CV_8UC1, Size(8, 8));

	multiplyOwn(resultsSource, temp2, betterEdgeMask);
	betterEdgeMask.convertTo(betterEdgeMask, CV_8UC1);

	return betterEdgeMask;
}

//Blurring edged guided filtering
int main() {
	//Reading images
	//Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
	//Mat input   = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/42/FlexiGoldFinger_CurMask.jpg", IMREAD_GRAYSCALE);
	vector<int> arr{ 1,2,21,22,41,42,61,62 };

	//for (int i = 0; i < 8; i++) {
	Mat input = imread("Resources/8 Micron Images/Median Stack Images/maskTemplate.bmp", IMREAD_GRAYSCALE);

	//Mat guide = imread("Resources/8 Micron Images/Angle 3 Reduce Border 0/" + to_string(arr[i]) + "/FlexiGoldFinger_RegisteredInspectionImage.bmp", IMREAD_GRAYSCALE);
	Mat guide = imread("Resources/8 Micron Images/Angle 3 Reduce Border 0/21/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);

	//catching if image cannot be read
	if (input.empty())
	{
		cout << "Can't read input image" << endl;
		return EXIT_FAILURE;
	}
	if (guide.empty())
	{
		cout << "Can't read guide image" << endl;
		return EXIT_FAILURE;
	}

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	correlationPadding(guide, input, input);
	//erode(input, input, kernel);

	//////Trackbar
	int r1 = 5;
	int x1 = 1;
	int r2 = 5;
	int x2 = 1;
	int k = 4;
	int midpoint = 102;
	int threshold_U = 80;
	int threshold_L = 25;

	namedWindow("Trackbar", (1500, 1000));
	createTrackbar("r1", "Trackbar", &r1, 20);
	createTrackbar("x1", "Trackbar", &x1, 20);
	createTrackbar("r2", "Trackbar", &r2, 20);
	createTrackbar("x2", "Trackbar", &x2, 20);
	createTrackbar("Image Upper Threshold", "Trackbar", &threshold_U, 255);
	createTrackbar("Image Lower Threshold", "Trackbar", &threshold_L, 150);

	//while (true) {
		Mat resultsOpenCV;
		Mat resultsOpenCV_32;
		Mat results_32;
		Mat resultsSource;
		Mat imgCanny;
		Mat imgDia;
		Mat imgDia_32;
		Mat guidedEdge2pt;
		Mat guidedWeights;

		//running opencv guided filter first
		Mat smudgedGuided = smudgedEdgeGuidedFilter(guide, input, r2, pow(10, -x2));
		Mat deepGuided = guidedFilter(guide, input, r1, pow(10, -x1));


		Mat invertedGuide = 255 - deepGuided;
		invertedGuide.convertTo(invertedGuide, CV_32FC1);
		Mat invertedWeights = invertedGuide/255;
		Mat results;
		multiplyOwn(invertedWeights, smudgedGuided, results);


		//controlledNormalize(difference)
		Mat edge, erodedMask;
		whitify(smudgedGuided);
		Mat difference = smudgedGuided - guide;
		nullifyNegatives(difference, difference);

		////not necessary
		deepGuided = deepGFSelection(smudgedGuided, deepGuided);
		//imwrite("Resources/8 Micron Images/presentation results/selectedDGF.bmp", deepGuided);
		erode(smudgedGuided, erodedMask, kernel);
		Mat difference2 = smudgedGuided - deepGuided;
		edge = smudgedGuided - erodedMask;

		//weghted test image see if better comparison than image
		//Mat normed;
		//controlledNormalize(difference2, normed, 0.01 * threshold_U, threshold_L);

		results.convertTo(results, CV_8UC1);
		


		//cross and diagonal checking
		Mat crossChecked = crossCheck(smudgedGuided, difference2, edge, threshold_U, threshold_L);
		Mat diagChecked = diagCheck(smudgedGuided, difference2, edge, threshold_U, threshold_L);
		Mat crossDiagChecked = crossAndDiagCheck(smudgedGuided, difference2, edge);
		Mat newDiff3 = crossDiagChecked - deepGuided;
		Mat newDiff2 = diagChecked - deepGuided;
		Mat newDiff = crossChecked - deepGuided;
		nullifyNegatives(newDiff, newDiff);
		nullifyNegatives(newDiff2, newDiff2);

		Point arr[1000];
		crossDiagCheck(smudgedGuided, difference2, edge, arr);
		int height = difference.rows;
		int width = difference.cols;
		int arrSize = sizeof(arr) / sizeof(arr[0]);

		for (int i = 0; i < arrSize; i++) {
			difference.at<uchar>(arr[i].x, arr[i].y) = 0;
		}

		//concantenate filtered results vertically
		Mat a, b, c, d, e, f, g, h, i, j, guidedEdge2pt_8, nullifiedGuidedEdge2pt_8;

		resize(smudgedGuided, a, Size(), 2.0, 2.0);
		resize(deepGuided, b, Size(), 2.0, 2.0);
		resize(invertedGuide, c, Size(), 2.0, 2.0);
		resize(results, d, Size(), 2.0, 2.0);
		resize(crossChecked, e, Size(), 4.0, 4.0);
		resize(newDiff2, f, Size(), 4.0, 4.0);
		//resize(resultsSource, e, Size(), 1.0, 1.0);
		//erode(results2, g, kernel);
		//resize(resultsSource, e, Size(), 1.0, 1.0);

		//h = resultsSource - guide;
		//nullifyNegatives(h, h);
		//resize(h, h, Size(), 1.0, 1.0);


		//Mat imgArray[] = {e, f};
		//Mat dst;
		//vconcat(imgArray, 2, dst);
		//imshow("Results", dst);
		//imshow("Results", e);
		//imwrite("Resources/8 Micron Images/presentation results/results" + to_string(arr[i]) + ".bmp", dst);
		//imwrite("Resources/8 Micron Images/presentation results/smudgedGuided.bmp", smudgedGuided);
		//imwrite("Resources/8 Micron Images/presentation results/deepGuided.bmp", deepGuided);
		imwrite("Resources/8 Micron Images/presentation results/inspectionEGMask.bmp", difference);
	//	waitKey(40);
	//}
	//}
}

////Blurring edged guided filtering
//int main() {
//	//Reading images
//	//Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
//	//Mat input   = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/42/FlexiGoldFinger_CurMask.jpg", IMREAD_GRAYSCALE);
//	vector<int> arr{ 1,2,21,22,41,42,61,62 };
//
//	//for (int i = 0; i < 8; i++) {
//		Mat input = imread("Resources/8 Micron Images/Median Stack Images/maskTemplate.bmp", IMREAD_GRAYSCALE);
//
//		//Mat guide = imread("Resources/8 Micron Images/Angle 3 Reduce Border 0/" + to_string(arr[i]) + "/FlexiGoldFinger_RegisteredInspectionImage.bmp", IMREAD_GRAYSCALE);
//		Mat guide = imread("Resources/8 Micron Images/Angle 3 Reduce Border 0/21/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
//		//int rows = input.rows;
//		//int cols = input.cols;
//		//cout << "dimensions x" << rows << endl;
//		//cout << "dimensions y" << cols << endl;
//
//		//catching if image cannot be read
//		if (input.empty())
//		{
//			cout << "Can't read input image" << endl;
//			return EXIT_FAILURE;
//		}
//		if (guide.empty())
//		{
//			cout << "Can't read guide image" << endl;
//			return EXIT_FAILURE;
//		}
//
//		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//		correlationPadding(guide, input, input);
//		//erode(input, input, kernel);
//
//		//////Trackbar
//		int r1 = 1;
//		int x1 = 1;
//		int r2 = 4;
//		int x2 = 1;
//		int threshold_L = 10;
//		int threshold_U = 8;
//
//		namedWindow("Trackbar", (1500, 1000));
//		createTrackbar("r1", "Trackbar", &r1, 50);
//		createTrackbar("x1", "Trackbar", &x1, 20);
//		createTrackbar("r2", "Trackbar", &r2, 50);
//		createTrackbar("x2", "Trackbar", &x2, 20);
//		createTrackbar("Image Upper Threshold", "Trackbar", &threshold_U, 20);
//		createTrackbar("Image Lower Threshold", "Trackbar", &threshold_L, 50);
//
//		while (true) {
//			Mat resultsOpenCV;
//			Mat results;
//			Mat resultsOpenCV_32;
//			Mat results_32;
//			Mat resultsSource;
//			Mat imgCanny;
//			Mat imgDia;
//			Mat imgDia_32;
//			Mat guidedEdge2pt;
//			Mat guidedWeights;
//
//			//running opencv guided filter first
//			//cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, r1, pow(10, -x1));
//			results = guidedFilter(guide, input, r1, pow(10, -x1));
//			Mat results2 = guidedFilter(guide, input, r1 + 1, pow(10, -x1));
//			//Canny(results, imgCanny, threshold_L, threshold_U);
//
//
//			resultsSource = edgeGuidedFilter(guide, input, r2, pow(10, -x2));
//
//			//concantenate filtered results vertically
//			Mat a, b, c, d, e, f, g, h, i, j, guidedEdge2pt_8, nullifiedGuidedEdge2pt_8;
//
//
//
//			//blurring eroded source image
//			kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//			erode(resultsSource, f, kernel);
//			boxFilter(f, i, CV_8UC1, Size(threshold_U, threshold_U));
//			Mat temp = (resultsSource) / 255;
//			Mat erodedbetterEdgeMask;
//			multiplyOwn(i, temp, erodedbetterEdgeMask);
//			erodedbetterEdgeMask.convertTo(erodedbetterEdgeMask, CV_8UC1);
//
//
//			h = resultsSource - guide;
//			nullifyNegatives(h, h);
//			resize(h, h, Size(), 1.0, 1.0);
//
//			//blurring source image itself
//			Mat blurred;
//			boxFilter(resultsSource, blurred, CV_8UC1, Size(threshold_L, threshold_L));
//			temp = (resultsSource - 20) / 255;
//			Mat betterEdgeMask;
//			multiplyOwn(blurred, temp, betterEdgeMask);
//			betterEdgeMask.convertTo(betterEdgeMask, CV_8UC1);
//
//			resize(input, a, Size(), 1.0, 1.0);
//			resize(guide, b, Size(), 1.0, 1.0);
//			//resize(resultsOpenCV, c, Size(), 2, 2);
//			resize(results, c, Size(), 1.0, 1.0);
//			resize(results2, d, Size(), 1.0, 1.0);
//			erode(results2, g, kernel);
//			resize(resultsSource, e, Size(), 1.0, 1.0);
//
//
//			//Mat betterEdgeMask = blurred * (resultsSource / 255);
//
//			//i = resultsSource - g - 50;
//			//i = 255 - resultsSource;
//			//dilate(i, i, kernel);
//
//			//j = guidedFilter(i, resultsSource, 5, 0.1);
//			//j = input - guide;
//			//nullifyNegatives(j, j);
//			//normalize(j, j, 0, 255, NORM_MINMAX, -1, Mat());
//			//resize(j, j, Size(), 1.0, 1.0);
//
//
//			Mat imgArray[] = { b, a, c, d, g, e ,f, erodedbetterEdgeMask, h ,betterEdgeMask};
//			Mat dst;
//			vconcat(imgArray, 10, dst);
//			imshow("Results", dst);
//			//imwrite("Resources/8 Micron Images/presentation results/results" + to_string(arr[i]) + ".bmp", dst);
//			//imwrite("Resources/8 Micron Images/presentation results/blurringEdgeFilter2.bmp", dst);
//			waitKey(40);
//		}
//	//}
//}
//
//#include "Header.h"
//
//void multiplyOwn(Mat& a, Mat& b, Mat& dst) {
//
//	int event = 0;
//	Mat aPrime = a.clone();
//	Mat bPrime = b.clone();
//
//	int height = a.rows;
//	int width = a.cols;
//
//	if (a.type() == 0 && b.type() == 0) {
//		dst.create(height, width, CV_8UC1);
//		event = 1;
//	}
//	else if (a.type() == 5 && b.type() == 0) {
//		bPrime.convertTo(bPrime, CV_32FC1);
//		dst.create(height, width, CV_32FC1);
//		event = 2;
//	}
//	else if (a.type() == 0 && b.type() == 5) {
//		aPrime.convertTo(aPrime, CV_32FC1);
//		dst.create(height, width, CV_32FC1);
//		event = 2;
//	}
//	else {
//		dst.create(height, width, CV_32FC1);
//		event = 2;
//	}
//
//	int i, j;
//
//	for (i = 0; i < height; i++) {
//		for (j = 0; j < width; j++) {
//			if (event == 1) {
//				dst.at<uchar>(i, j) = bPrime.at<uchar>(i, j) * aPrime.at<uchar>(i, j);
//			}
//			else if (event == 2) {
//				dst.at<float>(i, j) = bPrime.at<float>(i, j) * aPrime.at<float>(i, j);
//			}
//			else {
//				cout << "No Event Specified" << endl;
//			}
//		}
//	}
//}
//
//void subtractOwn(Mat& a, Mat& b, Mat& dst) {
//
//	int event = 0;
//	Mat aPrime = a.clone();
//	Mat bPrime = b.clone();
//
//	int height = a.rows;
//	int width = a.cols;
//
//	if (a.type() == 0 && b.type() == 0) {
//		dst.create(height, width, CV_8UC1);
//		event = 1;
//	}
//	else if (a.type() == 5 && b.type() == 0) {
//		bPrime.convertTo(bPrime, CV_32FC1);
//		dst.create(height, width, CV_32FC1);
//		event = 2;
//	}
//	else if (a.type() == 0 && b.type() == 5) {
//		aPrime.convertTo(aPrime, CV_32FC1);
//		dst.create(height, width, CV_32FC1);
//		event = 2;
//	}
//	else {
//		dst.create(height, width, CV_32FC1);
//		event = 2;
//	}
//
//	int i, j;
//
//	for (i = 0; i < height; i++) {
//		for (j = 0; j < width; j++) {
//			if (event == 1) {
//				dst.at<uchar>(i, j) = aPrime.at<uchar>(i, j) - bPrime.at<uchar>(i, j);
//			}
//			else if (event == 2) {
//				dst.at<float>(i, j) = aPrime.at<float>(i, j) - bPrime.at<float>(i, j);
//			}
//			else {
//				cout << "No Event Specified" << endl;
//			}
//		}
//	}
//}
//
//void edgesBoxFilter(Mat& a, Mat& dst, int ksize) {
//	
//}
//
//void nullifyNegatives(Mat& src, Mat& dst)
//{
//	cv::threshold(-src, dst, 0, 0, THRESH_TRUNC);
//	dst = -dst;
//	return;
//}
//
//string type2str(int type) {
//	string r;
//
//	uchar depth = type & CV_MAT_DEPTH_MASK;
//	uchar chans = 1 + (type >> CV_CN_SHIFT);
//
//	switch (depth) {
//	case CV_8U:  r = "8U"; break;
//	case CV_8S:  r = "8S"; break;
//	case CV_16U: r = "16U"; break;
//	case CV_16S: r = "16S"; break;
//	case CV_32S: r = "32S"; break;
//	case CV_32F: r = "32F"; break;
//	case CV_64F: r = "64F"; break;
//	default:     r = "User"; break;
//	}
//
//	r += "C";
//	r += (chans + '0');
//
//	return r;
//}
//
//vector<Point> edgePointLocator(Mat& edgeImage, int* sizeReturn ) {
//	int height = edgeImage.rows;
//	int width = edgeImage.cols; 
//	int size = height * width;
//
//	vector<Point> arr;
//	arr.resize(size);
//
//	int index = 0;
//
//	for (int i = 0; i < height; i++) {
//		for (int j = 0; j < width; j++) {
//			if (edgeImage.at<float>(i, j) > 1) {
//				arr[index] = Point(i, j);
//				index++;
//			}
//		}
//	}
//	*sizeReturn = index + 1;
//	return arr;
//}
//
////all inputs must be 32F
//void pickingPoints(Mat& q, Mat& Input, Mat& dst, Mat guidedWeights) {
//	int i, j;
//	int count = 0;
//	dst.create(q.rows, q.cols, CV_32FC1);
//
//	for (int i = 0; i < q.rows; i++) {
//		for (int j = 0; j < q.cols; j++) {
//			if (guidedWeights.at<float>(i, j) > 0.0) {
//				float filterOutput_percentage = guidedWeights.at<float>(i, j);
//				float inputImportance_percentage = 1.0 - filterOutput_percentage;
//
//				dst.at<float>(i, j) = q.at<float>(i, j)    * filterOutput_percentage - 
//									  Input.at<float>(i,j) * inputImportance_percentage;
//				count++;
//			}
//			else {
//				dst.at<float>(i, j) = Input.at<float>(i, j);
//			}
//		} 
//	}
//}
//
//Mat edgeGuidedFilter(Mat& I, Mat& p, int r, float eps, Mat guidedWeights )
//{
//	Size ksize(2 * r + 1, 2 * r + 1);
//	Mat Guide = I.clone();
//	Mat Input = p.clone();
//
//	//Step 1:
//	//mean_I, mean_p, corr_I, corr_Ip
//	Mat mean_p;
//	Mat mean_I;
//	Mat corr_Ip;
//	Mat corr_I;
//
//	boxFilter(Guide, mean_I, CV_32F, ksize);
//	boxFilter(Input, mean_p, CV_32F, ksize);
//	I.convertTo(Guide, CV_32FC1);
//	p.convertTo(Input, CV_32FC1);
//
//	Mat tmpIp;
//	multiplyOwn(Guide, Input, tmpIp);
//	boxFilter(tmpIp, corr_Ip, CV_32F, ksize);
//
//	Mat_<double> mean_II, tmpII;
//	tmpII = Guide.mul(Guide);
//	boxFilter(tmpII, corr_I, CV_32F, ksize);
//
//
//	//Step 2:
//	//var_I  - variance of I in each local patch: the matrix Sigma in Eqn (14)
//	//cov_Ip - covariance of (I, p) in each local patch 
//
//	Mat var_I;
//	Mat tmp_II;
//	multiplyOwn(mean_I, mean_I, mean_II);
//	subtractOwn(corr_I, mean_II, var_I);
//
//	Mat cov_Ip;
//	Mat mean_Ip;
//	multiplyOwn(mean_I, mean_p, mean_Ip);
//	cov_Ip = corr_Ip - mean_Ip;
//
//
//	//Step 3:
//	//compute a and b
//
//	Mat a(Input.rows, Input.cols, CV_MAKETYPE(CV_32F, 1));
//	Mat b(Input.rows, Input.cols, CV_32F);
//	divide(cov_Ip, var_I + eps, a);
//
//	Mat aMulmean_I;
//	multiplyOwn(a, mean_I, aMulmean_I);
//	b = mean_p - aMulmean_I;
//
//
//	//Step 5:
//	//find mean_a and mean_b
//
//	Mat mean_a;
//	Mat mean_b;
//	boxFilter(a, mean_a, CV_32F, ksize);
//	boxFilter(b, mean_b, CV_32F, ksize);
//	Mat aI;
//	multiplyOwn(mean_a, Guide, aI);
//
//	Mat q, dst;
//	q = aI + mean_b;
//
//	//only taking q points of the edges
//	pickingPoints(q, Input, dst, guidedWeights);
//	dst.convertTo(dst, CV_8UC1);
//	return dst;
//}
//
////Mat GuidedFilter(Mat& I, Mat& p, int r, float eps) {
////
////}
//
////Clean Guided Canny Code
//int main() {
//	//Reading images
//	Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
//	Mat guide = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/1/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
//	Mat edgeImg = imread("Resources/15 Micron Images/ASE15um/EdgeMasks/2ptEdgeMask.bmp", IMREAD_GRAYSCALE);
//	int numEdges;
//
//	//catching if image cannot be read
//	if (input.empty())
//	{
//		cout << "Can't read input image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (guide.empty())
//	{
//		cout << "Can't read guide image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (edgeImg.empty())
//	{
//		cout << "Can't read edge image" << endl;
//		return EXIT_FAILURE;
//	}
//
//	////////Trackbar
//	int r1 = 1;
//	int x1 = 1;
//	int threshold_L = 78;
//	int threshold_U = 213;
//
//	namedWindow("Trackbar", (1500, 1000));
//	createTrackbar("r1", "Trackbar", &r1, 50);
//	createTrackbar("x1", "Trackbar", &x1, 20);
//	createTrackbar("Image Upper Threshold", "Trackbar", &threshold_U, 255);
//	createTrackbar("Image Lower Threshold", "Trackbar", &threshold_L, 255);
//
//	while (true) {
//		Mat resultsOpenCV;
//		Mat resultsOpenCV_32;
//		Mat resultsSource;
//		Mat imgCanny;
//		Mat imgDia;
//		Mat imgDia_32;
//		Mat guidedEdge2pt;
//		Mat guidedWeights;
//
//		//running opencv guided filter first
//		cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, 1, pow(10, -1));
//
//		Canny(resultsOpenCV, imgCanny, threshold_L, threshold_U);
//
//		//thickening canny edge by 1
//		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//		dilate(imgCanny, imgDia, kernel);
//
//		//Getting guided 2pt edge
//		resultsOpenCV.convertTo(resultsOpenCV_32, CV_32FC1);
//		imgDia.convertTo(imgDia_32, CV_32FC1);
//		guidedEdge2pt = imgDia_32 - resultsOpenCV_32; 
//
//		//Nullifying edges
//		Mat nullifiedGuidedEdge2pt;
//		nullifyNegatives(guidedEdge2pt, nullifiedGuidedEdge2pt);
//		//nullifiedGuidedEdge2pt = nullifiedGuidedEdge2pt / 255;
//		//imwrite("Resources/guidedEdge2pt.bmp", nullifiedGuidedEdge2pt * 255);
//		//imshow("nullifiedGuidedEdge2pt", nullifiedGuidedEdge2pt);
//		//waitKey(0);
//
//		//getting weights
//		normalize(nullifiedGuidedEdge2pt, guidedWeights, 0, 1, NORM_MINMAX, -1, Mat());
//		resultsSource = edgeGuidedFilter(guide, input, r1, pow(10, -x1), guidedWeights);
//
//		//concantenate filtered results vertically
//		Mat a, b, c, d, e, f, g;
//		resize(resultsOpenCV, a, Size(), 2.5, 2.5);
//		resize(imgCanny, b, Size(), 2.5, 2.5);
//		resize(imgDia, c, Size(), 2.5, 2.5);
//		resize(nullifiedGuidedEdge2pt/255, e, Size(), 2.5, 2.5);
//		resize(resultsSource, f, Size(), 2.5, 2.5);
//		g = resultsSource - guide;
//		nullifyNegatives(g, g);
//		resize(g, g, Size(), 2.5, 2.5);
//
//		//Mat imgArray[] = { a, b, c, d, e, f, g };
//		//Mat dst;
//		//vconcat(imgArray, 7, dst);
//		//imshow("Results", dst);
//		imshow("1px guided results", a);
//		imshow("Canny", b);
//		imshow("Dia", c);
//		imshow("Nullified 2pt edge", e);
//		imshow("result source", f);
//		imshow("Results Source - guide", g);
//
//		waitKey(30);
//	}
//}
//
//Guided Canny Attempt#1
//int main() {
//	//Reading images
//	Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
//	Mat guide = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/1/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
//	Mat edgeImg = imread("Resources/15 Micron Images/ASE15um/EdgeMasks/2ptEdgeMask.bmp", IMREAD_GRAYSCALE);
//	int numEdges;
//
//	//catching if image cannot be read
//	if (input.empty())
//	{
//		cout << "Can't read input image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (guide.empty())
//	{
//		cout << "Can't read guide image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (edgeImg.empty())
//	{
//		cout << "Can't read edge image" << endl;
//		return EXIT_FAILURE;
//	}
//
//	////////Trackbar
//	int r1 = 1;
//	int x1 = 1;
//	int threshold_L = 78;
//	int threshold_U = 213;
//
//	namedWindow("Trackbar", (1500, 1000));
//	createTrackbar("r1", "Trackbar", &r1, 50);
//	createTrackbar("x1", "Trackbar", &x1, 20);
//	createTrackbar("Image Upper Threshold", "Trackbar", &threshold_U, 255);
//	createTrackbar("Image Lower Threshold", "Trackbar", &threshold_L, 255);
//
//	while (true) {
//		Mat resultsOpenCV;
//		Mat resultsOpenCV_32;
//		Mat resultsSource;
//		Mat imgCanny;
//		Mat imgDia;
//		Mat imgDia_32;
//		Mat guidedEdge2pt;
//		Mat guidedWeights;
//
//		//comparing with opencv guided filter results
//		cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, 1, pow(10, -1));
//		
//		Canny(resultsOpenCV, imgCanny, threshold_L, threshold_U);
//
//			//thickening canny edge by 1
//		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//		dilate(imgCanny, imgDia, kernel);
//		
//			//Getting guided 2pt edge
//		resultsOpenCV.convertTo(resultsOpenCV_32, CV_32FC1);
//		imgDia.convertTo(imgDia_32, CV_32FC1);
//		//cout << "Results OpenCV: " << resultsOpenCV_32.type() << endl;
//		//cout << "Img Dialate: " << imgDia_32.type() << endl;
//		guidedEdge2pt = imgDia_32 - resultsOpenCV_32; //guidedEdge pt is having -ve values
//		//cout << "Guided edge 2pt: " << guidedEdge2pt.type() << endl;
//		//cout << "guided2pt" << endl << guidedEdge2pt << endl << endl;
//		Mat nullifiedGuidedEdge2pt;
//
//
//		nullifyNegatives(guidedEdge2pt, nullifiedGuidedEdge2pt);
//		//cout << "guided2ptNullified" << endl << nullifiedGuidedEdge2pt << endl << endl;
//		
//			//locating edges into an array
//		vector<Point> edgesArr = edgePointLocator(nullifiedGuidedEdge2pt, &numEdges);
//
//			//getting weights
//		normalize(nullifiedGuidedEdge2pt, guidedWeights, 0, 1, NORM_MINMAX, -1, Mat());
//		//cout << "Guided Weight Weights: " << guidedWeights.type() << endl;
//		resultsSource = edgeGuidedFilter(guide, input, r1, pow(10, -x1), edgesArr, numEdges, guidedWeights);
//		//cout << "results Source: " << endl << resultsSource << endl << endl;
//
//		//concantenate filtered results vertically
//		Mat a, b, c, d, e, f, g;
//		resize(resultsOpenCV, f, Size(), 2.5, 2.5);
//		resize(imgCanny, a, Size(), 2.5, 2.5);
//		resize(imgDia, b, Size(), 2.5, 2.5);
//		resize(guidedEdge2pt, c, Size(), 1, 1);
//		resize(nullifiedGuidedEdge2pt, g, Size(), 1, 1);
//		resize(resultsSource, d, Size(), 2.5, 2.5);
//		resize(resultsSource - guide, e, Size(), 2.5, 2.5);
//
//		//imwrite("Resources/guidedEdge2pt.bmp", guidedEdge2pt);
//		//imwrite("Resources/dialated.bmp", imgDia);
//		//imwrite("Resources/guidedMask1.bmp", resultsOpenCV);
//
//
//		//Mat imgArray[] = { a, b, c, d, e };
//		//Mat dst;
//		//vconcat(imgArray, 5, dst);
//		//imshow("Results", dst);
//		imshow("1px guided results", f);
//		imshow("Canny", a);
//		imshow("Dia", b);
//		imshow("2pt edge", c);
//		imshow("Nullified 2pt edge", g);
//		imshow("result source", d);
//		imshow("Results Source - guide", e);
//
//		waitKey(100000);
//	}
//}
//
//guided Canny Guided
//int main() {
//	//Reading images
//	Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
//	Mat guide = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/1/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
//	Mat edgeImg = imread("Resources/15 Micron Images/ASE15um/EdgeMasks/2ptEdgeMask.bmp", IMREAD_GRAYSCALE);
//
//	int numEdges;
//
//	//catching if image cannot be read
//	if (input.empty())
//	{
//		cout << "Can't read input image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (guide.empty())
//	{
//		cout << "Can't read guide image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (edgeImg.empty())
//	{
//		cout << "Can't read edge image" << endl;
//		return EXIT_FAILURE;
//	}
//
//	////////Trackbar
//	int r1 = 1;
//	int x1 = 1;
//	int r2 = 1;
//	int x2 = 1;
//
//	namedWindow("Trackbar", (1500, 1000));
//	createTrackbar("Image Upper Threshold", "Trackbar", &r1, 50);
//	createTrackbar("Image Lower Threshold", "Trackbar", &x1, 20);
//
//	//locating edges into an array
//	vector<Point> edgesArr = edgePointLocator(edgeImg, &numEdges);
//
//	while (true) {
//		Mat resultsOpenCV;
//		Mat resultsSource;
//
//		//comparing with opencv guided filter results
//		cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, r1, pow(10, -x1));
//		resultsSource = edgeGuidedFilter(guide, input, r1, pow(10, -x1), edgesArr, numEdges);
//		//cout << "reference guided filter = " << endl << " " << resultsOpenCV << endl << endl;
//
//		//concantenate filtered results vertically
//		Mat a, b, c, d;
//		resize(resultsOpenCV, a, Size(), 3, 3);
//		resize(resultsSource, b, Size(), 3, 3);
//		resize(resultsOpenCV - guide, c, Size(), 3, 3);
//		resize(resultsSource - guide, d, Size(), 3, 3);
//
//		Mat imgArray[] = { a, b, c, d };
//		Mat dst;
//		vconcat(imgArray, 4, dst);
//		imshow("Results", dst);
//		//imshow("Results Source", a);
//		//imshow("Results Opencv", b);
//
//		waitKey(30);
//	}
//}
//
//int main() {
//	//Reading images
//	Mat input = imread("Resources/15 Micron Images/ASE15um/MedianMask/PaddedMask.bmp", IMREAD_GRAYSCALE);
//	Mat guide = imread("Resources/15 Micron Images/ASE15um/Angle 3 Reduce Border 0/1/FlexiGoldFinger_RegisteredInspectionImage.jpg", IMREAD_GRAYSCALE);
//	Mat edgeImg = imread("Resources/15 Micron Images/ASE15um/EdgeMasks/2ptEdgeMask.bmp", IMREAD_GRAYSCALE);
//	//Mat input = imread("Resources/InputImage.bmp", IMREAD_GRAYSCALE);
//	//Mat guide = imread("Resources/GuideImage.bmp", IMREAD_GRAYSCALE);
//	//Mat smootest = imread("Resources/GuideImage.bmp", IMREAD_GRAYSCALE);
//	//Mat edgeImg3 = imread("Resources/edge3px.bmp", IMREAD_GRAYSCALE);
//	//Mat edgeImg2 = imread("Resources/edge2px.bmp", IMREAD_GRAYSCALE);
//	//Mat edgeImg1 = imread("Resources/edge1px.bmp", IMREAD_GRAYSCALE);
//	int numEdges;
//
//	//catching if image cannot be read
//	if (input.empty())
//	{
//		cout << "Can't read input image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (guide.empty())
//	{
//		cout << "Can't read guide image" << endl;
//		return EXIT_FAILURE;
//	}
//	if (edgeImg.empty())
//	{
//		cout << "Can't read edge image" << endl;
//		return EXIT_FAILURE;
//	}
//
//	////////Trackbar
//	int r1 = 1;
//	int x1 = 1;
//	int r2 = 1;
//	int x2 = 1;
//
//	namedWindow("Trackbar", (1500, 1000));
//	createTrackbar("Image Upper Threshold", "Trackbar", &r1, 50);
//	createTrackbar("Image Lower Threshold", "Trackbar", &x1, 20);
//
//	//locating edges into an array
//	vector<Point> edgesArr = edgePointLocator(edgeImg, &numEdges);
//
//	while (true) {
//		Mat resultsOpenCV;
//		Mat resultsSource;
//
//		//comparing with opencv guided filter results
//		cv::ximgproc::guidedFilter(guide, input, resultsOpenCV, r1, pow(10, -x1));
//		resultsSource = edgeGuidedFilter(guide, input, r1, pow(10, -x1), edgesArr, numEdges);
//		//cout << "reference guided filter = " << endl << " " << resultsOpenCV << endl << endl;
//
//		//concantenate filtered results vertically
//		Mat a, b, c, d;
//		resize(resultsOpenCV, a, Size(), 3, 3);
//		resize(resultsSource, b, Size(), 3, 3);
//		resize(resultsOpenCV - guide, c, Size(), 3, 3);
//		resize(resultsSource - guide, d, Size(), 3, 3);
//
//		Mat imgArray[] = { a, b, c, d };
//		Mat dst;
//		vconcat(imgArray, 4, dst);
//		imshow("Results", dst);
//		//imshow("Results Source", a);
//		//imshow("Results Opencv", b);
//
//		waitKey(30);
//	}
////}