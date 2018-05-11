#include "stdafx.h"
#include "bilinear.h"

#define fmin(x,y)	(x)<(y)?(x):(y)
#define fmax(x,y)	(x)>(y)?(x):(y)

cv::Mat bilinear(cv::Mat& im, const cv::Size& dsize, float fx, float fy){
	assert(im.channels() == 1); //only test single channel
	
	cv::Mat tmp_im = im;
	if (im.type() == CV_8UC1){
		im.convertTo(tmp_im, CV_32FC1, 1, 0); 
	}

	int h = tmp_im.rows;
	int w = tmp_im.cols;
	int nh = std::floor(1.0*h*fy);
	int nw = std::floor(1.0*w*fx);
	if (dsize.width != 0 || dsize.height != 0){
		nh = dsize.height;
		nw = dsize.width;
		fx = 1.0*nw / w;
		fy = 1.0*nh / h;
	}
	cv::Mat scaleMat(nh, nw, tmp_im.type());

	float* im_data = (float*)tmp_im.data; 
	float* n_data = (float*)scaleMat.data;
	int count = nh*nw;
	for (int i = 0; i < count; ++i){
		int y = i / nw;
		int x = i - y*nw;
		float tx = (1.0*x + 0.5) / fx - 0.5;
		float ty = (1.0*y + 0.5) / fy - 0.5;

		tx = fmin(w-1, fmax(0, tx));
		ty = fmin(h-1, fmax(0, ty));

		int ylow = std::floor(ty);
		int xlow = std::floor(tx);
		int yhigh = fmin(h-1,ylow + 1);
		int xhigh = fmin(w-1,xlow + 1);

		if (ylow == yhigh)
			ylow -= 1;
		if (xlow == xhigh)
			xlow -= 1;
		int indices[4] = { ylow*w + xlow, yhigh*w + xlow, ylow*w + xhigh, yhigh*w + xhigh };
		float weights[4] = { (xhigh - tx)*(yhigh - ty), (ty - ylow)*(xhigh - tx), \
			(yhigh - ty)*(tx - xlow), (ty - ylow)*(tx - xlow) };

		n_data[i] = 0;
		for (int k = 0; k < 4; ++k)
			n_data[i] += 1.0*im_data[indices[k]] * weights[k];
	}
	return scaleMat;
}

cv::Mat deconvolution(cv::Mat& im, int scale){
	assert(im.channels() == 1); //only test single channel
	
	cv::Mat tmp_im = im;
	if (im.type() == CV_8UC1){
		im.convertTo(tmp_im, CV_32FC1, 1, 0);
	}

	int dh = tmp_im.rows;
	int dw = tmp_im.cols;
	int ksize = scale * 2 - scale % 2;
	int factor = (ksize + 1) >> 1;	//ceil(1.0*ksize/2)
	float center = factor - 1;
	if (ksize % 2 == 0)
		center = factor - 0.5;
	int pad = (ksize - scale) / 2;
	int tp = ksize - 1 - pad;

	cv::Mat kernel(ksize, ksize, CV_32FC1);
	float* k_data = (float*)kernel.data;
	int count = ksize*ksize;
	for (int i = 0; i < count; ++i){
		int y = i / ksize;
		int x = i - y*ksize;

		float wy = 1 - 1.0*abs(y - center) / factor;
		float wx = 1 - 1.0*abs(x - center) / factor;
		k_data[i] = wy*wx;
	}
	//cv::Mat im_float = cv::Mat::zeros(scale*(dh - 1) + ksize - 2 * pad, scale*(dw - 1) + ksize - 2 * pad, CV_32FC1);
	cv::Mat im_float = cv::Mat::zeros(scale*(dh - 1) + 2 * ksize - 2 * pad - 1, scale*(dw - 1) + 2 * ksize - 2 * pad - 1, CV_32FC1);
	int nh = im_float.rows;
	int nw = im_float.cols;
	int offset_y = std::floor((nh - scale*dh)*0.5);
	int offset_x = std::floor((nw - scale*dw)*0.5);
	cv::Rect roi(offset_x, offset_y, dw*scale, dh*scale);

	count = nh*nw;
	float* n_data = (float*)im_float.data;
	for (int i = 0; i < count; ++i){
		int y = i / nw;
		int x = i - y*nw;
		if (y < tp || x < tp)
			continue;
		if ((y - tp) % scale == 0 && (x - tp) % scale == 0){
			int dy = (y - tp) / scale;
			int dx = (x - tp) / scale;
			n_data[i] = 1.0*tmp_im.at<float>(dy, dx);
		}
	}

	cv::Mat out_float;
	cout << im_float.rows << "," << im_float.cols << endl;

	//cv::filter2D卷积后自动填充pad，保持尺寸不变
	cv::filter2D(im_float, out_float, im_float.depth(), kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
	out_float = out_float(roi);
	cout << out_float.rows << "," << out_float.cols << endl;
	//out_float.convertTo(out_int, CV_8UC1, 1);
	cout << "scale: " << out_float.rows / dh << endl;
	cout << "========" << endl;
	return out_float;
}

cv::Mat roiAlign(cv::Mat& im, const std::vector<cv::Rect>& rois, const cv::Size& roiSize, int stride){
	return cv::Mat();
}

cv::Mat roiAlignBackward(cv::Mat& im, cv::Mat& diff, const std::vector<cv::Rect>& rois, const cv::Size& roiSize, int stride){
	return cv::Mat();
}