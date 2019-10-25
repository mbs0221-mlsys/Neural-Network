#pragma once

#ifndef _OPS_
#define _OPS_

#include "tensor.h"

namespace ops {

	using namespace tensor;

	namespace conv {

		template<class T>
		Tensor<T>& conv3d(Tensor<T> input, Tensor<T> kernel, int padding = 0) {
			Shape spi = input.getShape();
			Shape spk = kernel.getShape();
			int sz[] = { spi[0], spi[1] - spk[1] + 1, spi[2] - spk[2] + 1, spi[3] };
			Shape shape(sz);
			Tensor<T> output(shape);
			for (int i = 0; i < spi[0]; i++) {
				__conv3d_(output[i], input, kernel);
			}
			return output;
		}
	}

	using namespace tensor;

	template<class T>
	Tensor<T> dropout(Tensor<T> &x, double rate) {
		Tensor<T> w = Tensor<T>::mask(x.shape(), 0.1, 1);
		return w * x;
	}

	template<class T>
	Tensor<T> matmul(Tensor<T> &a, Tensor<T> &b) {
		return a.matmul(b);
	}

	template<class T>
	Tensor<T> softmax(Tensor<T> &m) {
		Tensor<T> m_sum = m.exp();
		return m_sum / m_sum.reduce_sum(1);
	}

	template<class T>
	Tensor<T> sigmoid(Tensor<T> &y) {
		return y.sigmoid();
	}

	template<class T>
	Tensor<T> grad_sigmoid(Tensor<T> &y) {
		Tensor<T> ones = Tensor<T>::ones(y.getShape());
		return y * (ones - y);
	}

	template<class T>
	Tensor<T> grad_relu(Tensor<T> &x) {
		Tensor<T> grad(x.getShape());
		grad.foreach_assign([&](int i, int j, int k, int l) {
			return __relu_grad_(x.getValue(i, j, k, l));
		});
		return grad;
	}

	template<class T>
	Tensor<T> grad_relu(Tensor<T> &x, double max_value, double threshold, double negative_slope) {
		Tensor<T> grad(x.getShape());
		grad.foreach_assign([&](int i, int j, int k, int l) {
			return __relu_grad_(x.data[i][j], max_value, threshold, negative_slope);
		});
		return grad;
	}

	// loss function
	template<class T>
	Tensor<T> mse(Tensor<T> &y_, Tensor<T> &y) {
		Tensor<T> mse = (0.5 * (y_ - y)*(y_ - y));
		return mse.reduce_mean(2).reduce_mean(3);
	}

	template<class T>
	Tensor<T> cross_entropy_loss(Tensor<T> &y_, Tensor<T> &y) {
		Tensor<T> error = ((T)0.0f-(y*y_.log() + ((T)1.0f - y)*((T)1.0f - y_).log()));
		return error.reduce_mean(2).reduce_mean(3);
	}
}

#endif // !_OPS_