#pragma once

//#define _CRT_SECURE_NO_WARNINGS 1

#include <opencv2\opencv.hpp>
#include <opencv\cv.h>

#include <easyx.h>
#include <graphics.h>

#include "tensor.h"

namespace image {

	using namespace std;
	using namespace tensor;
	using namespace cv;

	template<class T>
	Tensor<T> im2tensor(Mat im) {
		Tensor<T> tensor(1, im.rows, im.cols, im.channels());
		tensor.foreach_assign([&](int i, int j, int k, int l) {
			return (double)(im.at<uchar>(j, k, l));
		});
		return tensor;
	}

	template<class T>
	Mat im2tensor(Tensor<T> tensor) {
		Shape shape = tensor.getShape();
		Mat mat(shape[1], shape[2], CV_8UC3);// COLOR3
		Vec3b pixel;
		Tensor<T> tensor(1, im.rows, im.cols, im.channels);
		tensor.foreach([&](int i, int j, int k, int l) {
			uchar value = (uchar)(tensor.at(0, j, k, l));
			pixel[l] = value;// BGR
			if (l == shape[3] - 1) {
				mat.at<Vec3b>(j, k) = pixel;
			}
		});
		return mat;
	}
}