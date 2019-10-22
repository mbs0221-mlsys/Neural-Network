#pragma once

#ifndef _OPS_
#define _OPS_

#include "matrix.h"

namespace ops {

	template<class T>
	Matrix<T> dropout(Matrix<T> &x, double rate) {
		Matrix<T> w = Matrix<T>::mask(x.shape(), 0.1, 1);
		return w * x;
	}

	template<class T>
	Matrix<T> matmul(Matrix<T> &a, Matrix<T> &b) {
		return a.matmul(b);
	}

	template<class T>
	Matrix<T> softmax(Matrix<T> &m) {
		Matrix<T> m_sum = m.exp().reduce_sum(1);
		return m / m_sum;
	}

	template<class T>
	Matrix<T> sigmoid(Matrix<T> &y) {
		return y.sigmoid();
	}

	template<class T>
	Matrix<T> grad_sigmoid(Matrix<T> &y) {
		int size[] = { y.row, y.col };
		Matrix<T> ones = Matrix<T>::ones(Shape(size));
		return y * (ones - y);
	}

	template<class T>
	Matrix<T> grad_relu(Matrix<T> &x) {
		Matrix<T> grad(x.row, x.col);
		for (int i = 0; i < x.row; i++) {
			for (int j = 0; j < x.col; j++) {
				grad.data[i][j] = x.data[i][j] > 0 ? 1.0 : 0;
			}
		}
		return grad;
	}

	template<class T>
	Matrix<T> mse(Matrix<T> &y_, Matrix<T> &y) {
		Matrix<T> mse = (0.5 * (y_ - y)*(y_ - y)).reduce_mean(0).reduce_mean(1);
		return mse;
	}

}

#endif // !_OPS_