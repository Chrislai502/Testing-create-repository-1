#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2\cvconfig.h>
#include <opencv2\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\ximgproc.hpp>
#include <opencv2\imgproc.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

#include <stdio.h>

void GuidedFilter(Mat guide, Mat src, Mat dst, int radius, double eps);

#endif