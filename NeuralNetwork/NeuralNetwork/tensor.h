#pragma once

#include "matrix.h"

#ifndef _TENSOR_H_

inline int sub2ind(const Shape &s, int i, int j, int k, int l) {
	return ((((i*s[1]) + j)*s[2] + k)*s[3] + l);
}

namespace tensor {

	template<class T>
	class Tensor {
	private:
		Shape shape;
		T *data;
	public:
		Tensor() { data = nullptr; }
		Tensor(Shape &shape) :shape(shape) {
			data = new T[shape.size()];
		}
		Tensor(Tensor<T> &tensor) {
			data = new T[tensor.shape.size()];
		}
		~Tensor() { if (data != nullptr) delete data; }
		inline T& operator()(int i, int j, int k, int l) {
			int idx = sub2ind(shape, i, j, k, l);
			return data[idx];
		}
		inline T& operator()(int j, int k, int l) {
			int idx = sub2ind(shape, 0, j, k, l);
			return data[idx];
		}
		inline T& operator()(int k, int l) {
			int idx = sub2ind(shape, 0, 0, k, l);
			return data[idx];
		}
		inline T& operator()(int l) {
			int idx = sub2ind(shape, 0, 0, 0, l);
			return data[idx];
		}
		Shape& getShape() {
			return shape;
		}
	};

	template<class T>
	void __reduce_sum_(Tensor<T> &output, Tensor<T> &input, int axis = 0) {
		Shape shape = input.getShape();
		// n_samples
		for (int i = 0; i < shape[0]; i++) {
			T sum = 0;

		}
	}

	template<class T>
	void __conv1d_(Tensor<T> &output, Tensor<T> &input, Tensor<T> &kernel) {

	}

	template<class T>
	void __conv2d_(Tensor<T> &output, Tensor<T> &input, Tensor<T> &kernel) {
		// width
		for (int ii = 0; ii < sp_i[1]; ii++) {
			// height
			for (int ij = 0; ij < sp_i[2]; ij++) {
				output(0, ii, ij, 0) = input(0, ii, ij, 0)*kernel(0, ii, ij, 0);
			}
		}
	}

	template<class T>
	void __conv3d_(Tensor<T> &output, Tensor<T> &input, Tensor<T> &kernel) {
		Shape sp_i = input.getShape();
		Shape sp_k = kernel.getShape();
		// channel
		for (int ik = 0; ik < sp_i[3]; ik++) {
			// width
			for (int ii = 0; ii < sp_i[1]; ii++) {
				// height
				for (int ij = 0; ij < sp_i[2]; ij++) {
					output(0, ii, ij, ik) = input(0, ii, ij, ik)*kernel(0, ii, ij, ik);
				}
			}
		}
	}

	template<class T>
	void __grad_conv3d_(Tensor<T> &output, Tensor<T> &input, Tensor<T> &kernel) {

	}

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
};
#endif // !_TENSOR_H_
