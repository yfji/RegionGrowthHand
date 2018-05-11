// ROIAlign.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "bilinear.h"

void testImage(){
	cv::Mat image = cv::imread("hand.jpg", 0);
	cv::Mat m_bilinear = bilinear(image, cv::Size(0, 0), 0.5, 0.5);
	cv::Mat m_deconv = deconvolution(image, 2);

	m_bilinear.convertTo(m_bilinear, CV_8UC1, 1);
	m_deconv.convertTo(m_deconv, CV_8UC1, 1);
	cv::imshow("raw", image);
	cv::imshow("bi", m_bilinear);
	cv::imshow("dec", m_deconv);
	cv::waitKey();
}
void testBilinearDeconv(){
	float data[9] = {
		1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5
	};
	cv::Mat matrix = cv::Mat(3, 3, CV_32FC1, data);
	int scale = 2;
	cv::Mat m_bilinear = bilinear(matrix,cv::Size(0, 0), scale, scale);
	cv::Mat m_deconv = deconvolution(matrix, scale);
	float* mb_data = (float*)m_bilinear.data;
	float* md_data = (float*)m_deconv.data;

	int nsize = 3 * scale;
	int count = nsize*nsize;
	for (int i = 0; i < count; ++i){
		cout << mb_data[i];
		if ((i + 1) % nsize == 0)
			cout << endl;
		else
			cout << '\t';
	}
	cout << "========" << endl;
	for (int i = 0; i < count; ++i){
		cout << md_data[i];
		if ((i + 1) % nsize == 0)
			cout << endl;
		else
			cout << '\t';
	}

}
int _tmain(int argc, _TCHAR* argv[])
{
	testImage();
	testBilinearDeconv();
	return 0;
}

