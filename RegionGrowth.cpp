// RegionGrowth.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

#define MAX_N	90000

cv::Mat globalImage;
static int target_side = 300;
static int stack[MAX_N];
static char used[MAX_N];
static int thresh = 27;
static string windowName = "hand";
cv::Rect handRect;
float scale;

int calcDistance(int v1, int v2);
int calcDistanceRGB(cv::Vec3i& c1, cv::Vec3i& c2, cv::Vec3f& weights);
void regionGrow(cv::Mat& img, int seedX, int seedY);
void regionGrowRGB(cv::Mat& img, int seedX, int seedY);

void mouseHandler(int event, int x, int y, int flag, void* param){
	if (event == cv::EVENT_LBUTTONUP){
		cv::Mat gray;
		cv::cvtColor(globalImage, gray, cv::COLOR_BGR2GRAY);
		float ratio = 1.0*target_side / gray.cols;
		if (gray.cols < gray.rows)
			ratio = 1.0*target_side / gray.rows;
		scale = ratio;
		cv::resize(globalImage, gray, cv::Size(), ratio, ratio, INTER_CUBIC);
		cv::imshow("gray", gray);
		cv::Vec3i seed = globalImage.at<Vec3i>(y, x);
		regionGrowRGB(gray, (int)(x*ratio), (int)(y*ratio));
	}
}

int calcDistance(int v1, int v2){
	return abs(v1 - v2);
}
int calcDistanceRGB(cv::Vec3i& c1, cv::Vec3i& c2, cv::Vec3f& weights){
	float sum = 0.0;
	for (auto i = 0; i < 3; ++i){
		sum += 1.0*abs(c1[i] - c2[i])*weights[i];
	}
	return (int)sum;
}
void regionGrow(cv::Mat& img, int seedX, int seedY){
	uchar* data = img.data;
	int w = img.cols;
	int h = img.rows;
	vector<int> neighbors = { -w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1 };

	int top = -1;
	
	memset(used, 0, sizeof(used));
	stack[++top] = seedY*w + seedX;
	int seed_value = data[seedY*w + seedX];
	while (top >= 0 && top < MAX_N){
		int top_index = stack[top];
		char push = 0;
		for (auto i = 0; i < neighbors.size(); ++i){
			int cur_index = top_index + neighbors[i];
			int cur_value = (int)data[cur_index]; 
			int r = cur_index / w;
			int c = cur_index - r*w;
			if (!used[cur_index] && r >= 0 && r < h && c>=0 && c < w && calcDistance(seed_value, cur_value) < thresh){
				used[cur_index] = 1;
				stack[++top] = cur_index;
				push = 1;
			}
		}
		if (!push)
			--top;
	}
	cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC1);
	uchar* ptr_canvas = canvas.data;
	for (auto i = 0; i < MAX_N; ++i){
		if (used[i])
			ptr_canvas[i] = 255;
	}
	cv::imshow("mask", canvas);
}

void regionGrowRGB(cv::Mat& img, int seedX, int seedY){
	uchar* data = img.data;
	int w = img.cols;
	int h = img.rows;
	int c = img.channels();
	vector<int> neighbors = { -w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1 };

	int top = -1;
	int seedIndex = seedY*w + seedX;
	memset(used, 0, sizeof(used));
	stack[++top] = seedIndex;
	cv::Vec3i color_seed((int)data[seedIndex * 3], (int)data[seedIndex * 3 + 1], (int)data[seedIndex * 3 + 2]);
	float ratio_b = color_seed[2] * 1.0 / (color_seed[0] + color_seed[1] + color_seed[2]);
	float ratio_g = color_seed[1] * 1.0 / color_seed[2] * ratio_b;
	float ratio_r = color_seed[0] * 1.0 / color_seed[2] * ratio_b;
	cv::Vec3f seed_weights(ratio_b, ratio_g, ratio_r);
	int ltx = 1e4;
	int lty = 1e4;
	int rbx = 0;
	int rby = 0;
	while (top >= 0 && top < MAX_N){
		int top_index = stack[top];
		char push = 0;
		for (auto i = 0; i < neighbors.size(); ++i){
			int cur_index = top_index + neighbors[i];
			cv::Vec3i color_cur((int)data[cur_index*3],(int)data[cur_index*3+1],(int)data[cur_index*3+2]);
			int r = cur_index / w;
			int c = cur_index - r*w;
			if (!used[cur_index] && r >= 0 && r < h && c >= 0 && c < w && calcDistanceRGB(color_seed, color_cur, seed_weights) < thresh){
				used[cur_index] = 1;
				stack[++top] = cur_index;
				push = 1;
				if (c < ltx)
					ltx = c;
				if (c > rbx)
					rbx = c;
				if (r < lty)
					lty = r;
				if (r > rby)
					rby = r;
			}
		}
		if (!push)
			--top;
	}
	handRect.x = (int)(ltx*1.0/scale);
	handRect.y = (int)(lty*1.0 / scale);
	handRect.width = (int)(1.0*(rbx - ltx)/scale);
	handRect.height = (int)(1.0*(rby - lty) / scale);

	cv::Mat handImg = globalImage.clone();
	cv::rectangle(handImg, handRect, cv::Scalar(0, 0, 255), 2);
	cv::Mat canvas = cv::Mat::zeros(h, w, CV_8UC1);
	uchar* ptr_canvas = canvas.data;
	for (auto i = 0; i < MAX_N; ++i){
		if (used[i])
			ptr_canvas[i] = 255;
	}
	cv::imshow("rect", handImg);
	cv::imshow("mask", canvas);
}

int _tmain(int argc, _TCHAR* argv[])
{
	globalImage = cv::imread("E:/yfji/Pictures/HandPose/hand4.jpg");
	namedWindow(windowName);
	cv::setMouseCallback(windowName, mouseHandler);
	cv::imshow(windowName, globalImage);
	cv::waitKey();
	return 0;
}

