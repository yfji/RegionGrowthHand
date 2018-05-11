#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

cv::Mat bilinear(cv::Mat& im, const cv::Size& dsize, float fx, float fy);

cv::Mat deconvolution(cv::Mat& im, int scale);

cv::Mat roiAlignForward(cv::Mat& im, const std::vector<cv::Rect>& rois, const cv::Size& roiSize, int stride);

cv::Mat roiAlignBackward(cv::Mat& im, cv::Mat& diff, const std::vector<cv::Rect>& rois, const cv::Size& roiSize, int stride);