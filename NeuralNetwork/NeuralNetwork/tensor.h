#pragma once

#ifndef _TENSOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <string>
#include <functional>
#include <vector>
#include <numeric>
#include <set>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <map>

#include "shape.h"

#ifndef RANDOM
#define RANDOM ((rand() % 100) / 100.0)
#endif // !RANDOM

// scalar function
template<class T>
inline T __exp_(T x) { return exp(x); }

template<class T>
inline T __log_(T x) { return log(x); }

template<class T>
inline T __sigmoid_(T x) { return 1 / (1 + exp(x)); }

template<class T>
inline T __sigmoid_grad_(T y) { return y * (1 - y); }

template<class T>
inline T __pow_(T x, int y) { return pow(x, y); }

template<class T>
inline T __relu_(T x) { return ((x > 0) ? x : 0); }

template<class T>
inline T __relu_grad_(T x) { return ((x > 0) ? 1.0f : 0.0f); }

template<class T>
inline T __relu_(T x, double max_value, double threshold, double negative_slop) {
	if (x >= max_value) {
		return max_value;
	}
	else if (x >= threshold) {
		return x;
	}
	else {
		return negative_slop * (x - threshold);
	}
}

template<class T>
inline T __relu_grad_(T x, double max_value, double threshold, double negative_slop) {
	if (x >= max_value) {
		return 0;
	}
	else if (x >= threshold) {
		return 1.0;
	}
	else {
		return negative_slop;
	}
}

namespace tensor {

	using namespace std;
	using namespace shape::oldshape;
	
	// Tensor definition
	template<class T>
	class Tensor {
	private:
		// attributes
		Shape shape;
		T *data;

		// __allocate_
		inline void __free_() {
			if (data != nullptr) {
				delete[] data;
			}
		}
		inline void __allocate_() {
			try {
				data = new T[length()];
			} catch (const bad_alloc & e){
				cerr << e.what() << endl;
			}
		}

	protected:

		Tensor<T> __foreach_assign_(Tensor<T> &tensor, function<T(T, T)> func) {
			Shape m_shape = tensor.getShape();
			Tensor<T> out(m_shape);
			int axis = -1;
			for (int i = 0; i < 5; i++) {
				if (shape[i] != m_shape[i]) {
					axis = i;
					break;
				}
			}
			switch (axis) {
			case 0: // foreach sample
				out.foreach_assign([&](int s, int f, int i, int j, int c) {
					return func(this->at(s, f, i, j, c), tensor.at(0, f, i, j, c));
				});
				break;
			case 1: // foreach frame
				out.foreach_assign([&](int s, int f, int i, int j, int c) {
					return func(this->at(s, f, i, j, c), tensor.at(s, 0, i, j, c));
				});
				break;
			case 2: // foreach column
				out.foreach_assign([&](int s, int f, int i, int j, int c) {
					return func(this->at(s, f, i, j, c), tensor.at(s, f, 0, j, c));
				});
				break;
			case 3: // foreach row
				out.foreach_assign([&](int s, int f, int i, int j, int c) {
					return func(this->at(s, f, i, j, c), tensor.at(s, f, i, 0, c));
				});
				break;
			case 4: // foreach channel
				out.foreach_assign([&](int s, int f, int i, int j, int c) {
					return func(this->at(s, f, i, j, c), tensor.at(s, f, i, j, 0));
				});
				break;
			default: // foreach element
				out.foreach_assign([&](int s, int f, int i, int j, int c) {
					return func(this->at(s, f, i, j, c), tensor.at(s, f, i, j, c));
				});
				break;
			}
			return out;
		}
		Tensor<T> __foreach_elem_assign_(function<T(T)> func) {
			Tensor<T> out(shape);
			out.foreach_elem_assign([&](int i) {
				return func(data[i]);
			});
			return out;
		}

		// pooling/upsampling
		inline Tensor<T> __pooling_(int width, T (*func)(T, T)) {
			// 2d pooling (MAX, MIN, AVG)
			int size[] = { shape[0], shape[1],	shape[2] / width, shape[3] / width,	shape[4] };
			Tensor<T> out = Tensor<T>::zeros(Shape(size));
			out.foreach_assign([&](int oi, int oj, int ok, int ol, int om) {
				// find (MAX, MIN) in input region (oj, ok)
				T value = this->at(oi, oj, ok*width, ol*width, om);
				for (int pk = 0; pk < width; pk++) {
					for (int pl = 0; pl < width; pl++) {
						T curr = this->at(oi, oj, ok*width + pk, ol*width + pl, om);
						value = func(value, curr);
					}
				}
				return value;
			});
			return out;
		}
		
	public: // constructor & destructor
		Tensor() {
			data = nullptr;
		}
		Tensor(int f, int w, int h, int c, int m) {
			int size[] = { f, w, h, c, m };// 1 samples by default
			shape = Shape(size);
			__allocate_();
		}
		Tensor(int size[]) {
			shape = Shape(size);
			__allocate_();
		}
		Tensor(Shape &shape) : shape(shape) {
			__allocate_();
		}
		Tensor(const Tensor<T> &tensor) {
			shape = tensor.getShape();
			__allocate_();
			tensor.foreach([&](int i, int j, int k, int l, int m) {
				T value = tensor.at(i, j, k, l, m);
				this->set(value, i, j, k, l, m);
			});
		}
		~Tensor() {
			__free_();
		}

		
	public: // get & set methods
		Shape getShape() const { return shape; }
		int length() { return shape.size(); }
		int size() { return (sizeof(T)*shape.size()); }

		// non-parallel foreach
		void foreach(function<void(int,int,int,int,int)> func) const {
			for (int i = 0; i < shape[0]; i++) {// sample
				for (int j = 0; j < shape[1]; j++) {// frame
					for (int k = 0; k < shape[2]; k++) {// column(width)
						for (int l = 0; l < shape[3]; l++) {// row(height)
							for (int m = 0; m < shape[4]; m++) {// depth(channel)
								func(i, j, k, l, m);
							}
						}
					}
				}
			}
		}
		void foreach_assign(function<T(int, int, int, int, int)> func) const {
			foreach([&](int i, int j, int k, int l, int m) {
				int idx = shape.sub2ind(i, j, k, l, m);
				data[idx] = func(i, j, k, l, m);
			});
		}
		
		// parallel foreach elem
		template<class Type>
		void foreach_elem(Type func) {
			int len = length();
			for (int i = 0; i < len; i++) {
				func(i);
			}
		}
		template<class Type>
		void foreach_elem_assign(Type func) {
			int len = length();
			for (int i = 0; i < len; i++) {
				data[i] = func(i);
			}
		}

	public:
		// operators
		Tensor<T> one_hot(int num) {
			Shape output_shape(shape[0], shape[1], shape[2], shape[3], num);
			// one-hot编码
			map<int, int> codes;
			Tensor<T> out = Tensor<T>::zeros(output_shape);
			out.foreach([&](int i, int j, int k, int l, int m) {
				int value = (int)(this->at(i, j, k, l, m));
				// 维护一个one-hot映射表
				if (codes.find(value) == codes.end()) {
					codes[value] = codes.size();
				}
				// 查找对应编码
				out.set(1.0f, i, j, k, l, codes[value]);
			});
			return out;
		}
		Tensor<T> add(Tensor<T> &m) {
			return (*this) + m;
		}
		Tensor<T> sub(Tensor<T> &m) {
			return (*this) - m;
		}
		Tensor<T> matmul(Tensor<T> &tensor) {
			Shape shape_b = tensor.getShape();
			// calculate this(:,:,:,ok,col)*(1,1,1,col,ol)
			int n_cols = shape[3];
			Tensor<T> out(shape[0], shape[1], shape[2], shape[3], shape_b[4]);
			out.foreach_assign([&](int oi, int oj, int ok, int ol, int om) {
				T value = 0;
				for (int k = 0; k < n_cols; k++) {
					value += this->at(oi, oj, ok, ol, k)*tensor.at(0, 0, 0, k, om);// broadcast
				}
				return value;
			});
			return out;
		}
		Tensor<T> Transpose() {
			Shape output_shape(shape[0], shape[1], shape[2], shape[4], shape[3]);
			output_shape.print();
			Tensor<T> out(output_shape);
			out.foreach_assign([&](int i, int j, int k, int l, int m) {
				return this->at(i, j, k, m, l);
			});
			return out;
		}
		Tensor<T> permute(int order[]) {
			// get new shape
			int size[] = { 0, 0, 0, 0, 0 };
			for (int i = 0; i < 5; i++) {
				size[i] = shape[order[i]];
			}
			Shape shape_out(size);
			Tensor<T> out = Tensor<T>::zeros(shape_out);
			// permute
			out.foreach_assign([&](int i, int j, int k, int l, int m) {
				int subs[] = { 0, 0, 0, 0, 0 };
				subs[order[0]] = i;
				subs[order[1]] = j;
				subs[order[2]] = k;
				subs[order[3]] = l;
				subs[order[4]] = m;
				return this->at(subs[0], subs[1], subs[2], subs[3], subs[4]);
			});
			return out;
		}
		Tensor<T> reshape(int size[]) {
			Shape shape_out(size);
			return reshape(shape_out);
		}
		Tensor<T> reshape(Shape &shape_out) {
			Tensor<T> out = Tensor<T>::zeros(shape_out);
			out.foreach_assign([&](int i, int j, int k, int l, int m) {
				int idx = shape_out.sub2ind(i, j, k, l, m);
				return data[idx];
			});
			return out;
		}
		Tensor<T> flatten(int dim = 2) {
			// merge last two dimensions by default
			Shape shape_out = shape;
			return reshape(shape_out.flatten(2));
		}
		Tensor<T> reduce_sum(int dim) {
			// sample, frame, width(column), height(row), channel. 
			Shape shape_out = shape;
			shape_out.set(1, dim);
			Tensor<T> out = Tensor<T>::zeros(shape_out);
			switch (dim) {
			case 0:// samples
				foreach([&](int i, int j, int k, int l, int m) {
					T value = out.at(0, j, k, l, m) + this->at(i, j, k, l, m);
					out.set(value, 0, j, k, l, m);
				});
			case 1:// frame
				foreach([&](int i, int j, int k, int l, int m) {
					T value = out.at(i, 0, k, l, m) + this->at(i, j, k, l, m);
					out.set(value, i, 0, k, l, m);
				});
				break;
			case 2:// width
				foreach([&](int i, int j, int k, int l, int m) {
					T value = out.at(i, j, 0, l, m) + this->at(i, j, k, l, m);
					out.set(value, i, j, 0, l, m);
				});
				break;
			case 3:// height
				foreach([&](int i, int j, int k, int l, int m) {
					T value = out.at(i, j, k, 0, m) + this->at(i, j, k, l, m);
					out.set(value, i, j, k, 0, m);
				});
				break;
			case 4:// channel
				foreach([&](int i, int j, int k, int l, int m) {
					T value = out.at(i, j, k, l, 0) + this->at(i, j, k, l, m);
					out.set(value, i, j, k, l, 0);
				});
				break;
			default:
				cout << "error in reduce sum" << endl;
				break;
			}
			return out;
		}
		Tensor<T> reduce_mean(int dim) {
			Tensor<T> out = reduce_sum(dim);
			int N = shape[dim];
			return out / N;
		}
		Tensor<T> reduce_mean() {
			Shape shape_out(1, 1, 1, 1, 1);
			Tensor<T> out = Tensor<T>::zeros(shape_out);
			T value = 0;
			foreach_elem([&](int i) {
				value += data[i];
			});
			out.set(value, 0);
			return out;
		}
		
	public: // scalar operator
		Tensor<T> operator +(T b) { return __foreach_elem_assign_([&](T x) { return x + b; }); }
		Tensor<T> operator -(T b) { return __foreach_elem_assign_([&](T x) { return x - b; }); }
		Tensor<T> operator *(T b) { return __foreach_elem_assign_([&](T x) { return x * b; }); }
		Tensor<T> operator /(T b) { return __foreach_elem_assign_([&](T x) { return x / b; }); }
		
	public: // matrix operator
		Tensor<T> operator +(Tensor<T> &x) {
			return __foreach_assign_(x, [](T a, T b) { return a + b; });
		}
		Tensor<T> operator -(Tensor<T> &x) {
			return __foreach_assign_(x, [](T a, T b) { return a - b; });
		}
		Tensor<T> operator *(Tensor<T> &x) {
			return __foreach_assign_(x, [](T a, T b) { return a * b; });
		}
		Tensor<T> operator /(Tensor<T> &x) {
			return __foreach_assign_(x, [](T a, T b) { return a / b; });
		}

		// find min/max value
		T find_min() {
			T value = data[0];
			foreach_elem([&](int i) {
				if (data[i] < value) {
					value = data[i];
				}
			});
			return value;
		}
		T find_max() {
			T value = data[0];
			foreach_elem([&](int i) {
				if (data[i] > value) {
					value = data[i];
				}
			});
			return value;
		}

		// operator ()
		inline void set(T value, int i) {
			data[i] = value;
		}
		inline void set(T value, int i, int j, int k, int l, int m) {
			int idx = shape.sub2ind(i, j, k, l, m);
			data[idx] = value;
		}
		inline T at(int i, int j, int k, int l, int m) const {
			int idx = shape.sub2ind(i, j, k, l, m);
			return data[idx];
		}
		inline T at(int l) {
			int idx = shape.sub2ind(0, 0, 0, 0, l);
			return data[idx];
		}

		inline T get(int idx) const { return data[idx]; }
		
	public:
		// rotate operation
		Tensor<T> rotate180() {
			Tensor<T> out = Tensor<T>::zeros(shape);
			// rotate by (:,:,w,h,:)
			out.foreach([&](int ii, int ij, int ik, int il, int im) {
				T value = at(ii, ij, shape[2] - ik - 1, shape[3] - il - 1, im);
				out.set(value, ii, ij, ik, il, im);
			});
			return out;
		}
		
		// padding/clipping operation
		Tensor<T> padding(int width) {
			int size[] = {
				shape[0], shape[1],
				shape[2] + width * 2, shape[3] + width * 2, 
				shape[4]
			};
			Tensor<T> out = Tensor<T>::zeros(Shape(size));
			// calculate 2d padding
			foreach([&](int ii, int ij, int ik, int il, int im) {
				T value = this->at(ii, ij, ik, il, im);
				out.set(value, ii, ij, ik + width, il + width, im);
			});
			return out;
		}
		Tensor<T> clipping(int margin) {
			int size[] = {
				shape[0], shape[1],
				shape[2] - margin * 2, shape[3] - margin * 2,
				shape[4]
			};
			Tensor<T> out = Tensor<T>::zeros(Shape(size));
			// calculate 2d clipping
			out.foreach_assign([&](int ii, int ij, int ik, int il, int im) {
				return this->at(ii, ij, ik + margin, il + margin, im);
			});
			return out;
		}

		// convolution operation
		Tensor<T> conv2d(Tensor<T> &filter, Tensor<T> &bias, int stride) {

			// output shape (n_samples, 1, width, height, channel)
			Shape filter_shape = filter.getShape();
			int n_samples = shape[0];
			int n_frames = shape[1];
			int width = (shape[2] - filter_shape[2]) / stride + 1;
			int height = (shape[3] - filter_shape[3]) / stride + 1;
			int n_channels = filter_shape[0];// number of channels
			Shape output_shape(n_samples, n_frames, width, height, n_channels);
			
			// for unit test
			int before[] = { 0, 1, 4, 2, 3 };
			int count = filter_shape[1] * filter_shape[2] * filter_shape[3] * filter_shape[4];

			// calculate 2d concolution
#ifdef DEBUG
			cout << "<---------------------------------" << endl;
#endif // DEBUG
			std::vector<T> list;
			Tensor<T> out = Tensor<T>::zeros(output_shape);
			out.foreach([&](int oi, int oj, int ok, int ol, int om) {
				if (om == 0) {// channel					
					// (n_samples,1,:,:,n_channels)*(n_filters,1,:,:,n_channels)
					filter.foreach([&](int ki, int kj, int kk, int kl, int km) {
						// (1, frame, width, height, channel)		
						T a = this->at(oi, oj, ok*stride + kk, ol*stride + kl, km);
						T b = filter.at(ki, kj, kk, kl, km);
						list.push_back(a * b);
						if (list.size() == count) {
							T value = accumulate(list.begin(), list.end(), 0);
							T z = bias.at(ki);
							out.set(value + z, oi, oj, ok, ol, ki);
#ifdef DEBUG
							cout << "---------------------------------" << endl;
							out.permute(before).print();
#endif // DEBUG
							list.clear();
						}
					});
				}
			});
#ifdef DEBUG
			cout << "--------------------------------->" << endl;
#endif // DEBUG
			return out;
		}
		Tensor<T> conv2d(Tensor<T> &filter, int stride) {

			// output shape (n_samples, 1, width, height, channel)
			Shape filter_shape = filter.getShape();
			int n_samples = shape[0];
			int n_frames = shape[1];
			int width = (shape[2] - filter_shape[2]) / stride + 1;
			int height = (shape[3] - filter_shape[3]) / stride + 1;
			int n_channels = filter_shape[0];// number of channels
			Shape output_shape(n_samples, n_frames, width, height, n_channels);

			// for unit test
			int before[] = { 0, 1, 4, 2, 3 };
			int count = filter_shape[1] * filter_shape[2] * filter_shape[3] * filter_shape[4];

			// calculate 2d concolution
			std::vector<T> list;
			Tensor<T> out = Tensor<T>::zeros(output_shape);
			out.foreach([&](int oi, int oj, int ok, int ol, int om) {
				if (om == 0) {
					// (1, frame, width, height, channel)		
					filter.foreach([&](int ki, int kj, int kk, int kl, int km) {
						T a = this->at(oi, oj, ok*stride + kk, ol*stride + kl, km);
						T b = filter.at(ki, kj, kk, kl, km);
						list.push_back(a * b);
						if (list.size() == count) {
							T value = accumulate(list.begin(), list.end(), 0);
							out.set(value, oi, oj, ok, ol, ki);
							list.clear();
						}
					});
				}
			});
			return out;
		}
		Tensor<T> conv3d(Tensor<T> &filter, Tensor<T> &bias, int stride) {
			Shape filter_shape = filter.getShape();

			// output shape (n_samples, n_frames, n_columns, n_rows, n_filters)
			int n_samples = shape[0];
			int n_frames = (shape[1] - filter_shape[1]) / stride + 1;
			int width = (shape[2] - filter_shape[2]) / stride + 1;
			int height = (shape[3] - filter_shape[3]) / stride + 1;
			int n_filters = filter_shape[0];
			Shape output_shape(n_samples, n_frames, width, height, n_filters);

			// for unit test
			int before[] = { 0, 1, 4, 2, 3 };
			int count = filter_shape[1] * filter_shape[2] * filter_shape[3] * filter_shape[4];

#ifdef DEBUG
			cout << "<---------------------------------" << endl;
#endif // DEBUG
			// calculate 3d concolution	
			std::vector<T> list;
			Tensor<T> out = Tensor<T>::zeros(output_shape);
			out.foreach([&](int oi, int oj, int ok, int ol, int om) {
				if (om == 0) {// channel
					int index = 1; T value = 0;
					// (n_samples,:,:,:,n_channels)*(n_filters,:,:,:,n_channels)
					filter.foreach([&](int ki, int kj, int kk, int kl, int km) {
						// (1, frame, width, height, channel)
						T a = this->at(oi, oj*stride + kj, ok*stride + kk, ok*stride + kl, km);
						T b = filter.at(ki, kj, kk, kl, km);
						list.push_back(a * b);
						if (list.size() == count) {
							T value = accumulate(list.begin(), list.end(), 0);
							T z = bias.at(ki);
							out.set(value + z, oi, oj, ok, ol, ki);
#ifdef DEBUG
							cout << "---------------------------------" << endl;
							out.permute(before).print();
#endif // DEBUG
							list.clear();
						}
					});
				}
			});
#ifdef DEBUG
			cout << "--------------------------------->" << endl;
#endif // DEBUG
			return out;
		}
		
		// pooling operation
		Tensor<T> max_pooling(int width) {
			return __pooling_(width, [](T a, T b)->T { return ((a > b) ? (a) : (b)); });
		}
		Tensor<T> min_pooling(int width) {
			return __pooling_(width, [](T a, T b)->T { return ((a < b) ? (a) : (b)); });
		}
		Tensor<T> avg_pooling(int width) {
			Tensor<T> out = __pooling_(width, [](T a, T b)->T { return a + b; });
			double area = (width*width);
			return out / area;
		}

		// kronecker
		Tensor<T> kronecker(Tensor<T> &tensor) {
			Shape tensor_shape = tensor.getShape();
			int size[] = {
				shape[0] * tensor_shape[0],	shape[1] * tensor_shape[1],
				shape[2] * tensor_shape[2],	shape[3] * tensor_shape[3],
				shape[4] * tensor_shape[4]
			};
			Shape output_shape(size);
			Tensor<T> out = Tensor<T>::zeros(output_shape);
			foreach([&](int ii, int ij, int ik, int il, int im) {
				T a = this->at(ii, ij, ik, il, im);
				tensor.foreach([&](int ki, int kj, int kk, int kl, int km) {
					T value = a * tensor.at(ki, kj, kk, kl, km);					
					out.set(value, ii*tensor_shape[0] + ki, ij*tensor_shape[1] + kj,
						ik*tensor_shape[2] + kk, il*tensor_shape[3] + kl,
						im*tensor_shape[4] + km);
				});
			});
			return out;
		}

		// upsampling
		Tensor<T> upsampling(Tensor<T> &input, int width) {
			// 2d up_sampling (MAX, MIN)  https://www.cnblogs.com/pinard/p/6494810.html#!comments
			Tensor<T> out = Tensor<T>::zeros(input.getShape());
			foreach([&](int oi, int oj, int ok, int ol, int om) {
				T value = this->at(oi, oj, ok, ol, om);
				// find the position of value (MAX, MIN) in input
				int pk, pl;
				for (int k = 0; k < width; k++) {
					for (int l = 0; l < width; l++) {
						T curr = input.at(oi, oj, ok*width + k, ol*width + l, om);
						if (value == curr) {
							pk = k, pl = l;
							break;
						}
					}
				}
				// assign corresponding value
				out.set(value, oi, oj, ok*width + pk, ol*width + pl, om);
			});
			return out;
		}
		Tensor<T> avg_upsampling(int width) {
			// 2d up_sampling (AVG)  https://www.cnblogs.com/pinard/p/6494810.html#!comments
			Shape shape(1, 1, width, width, 1);
			return kronecker(Tensor<T>::numbers(shape, 1.0f / (width*width)));
		}
		
	public:
		// math function
		Tensor<T> softmax() { 
			Tensor<T> sum_e = exp().reduce_sum(4); 
			return (*this) / sum_e; 
		}
		Tensor<T> sigmoid() {
			return __foreach_elem_assign_([=](T x) {
				return __sigmoid_(x); 
			});
		}
		Tensor<T> exp() { 
			return __foreach_elem_assign_([=](T x) {
				return __exp_(x); 
			}); 
		}
		Tensor<T> log() {
			return __foreach_elem_assign_([=](T x) {
				return __log_(x); 
			});
		}
		Tensor<T> pow(int k) { 
			return __foreach_elem_assign_([=](T x) { return __pow_(x, k);
			}); 
		}
		Tensor<T> relu() { 
			return __foreach_elem_assign_([=](T x) {
				return __relu_(x); 
			});
		}
		Tensor<T> relu(double max_value, double threshold = 0.0f, double negative_slop = 0.1f) {
			return __foreach_elem_assign_([=](T x) {
				return __relu_(x, max_value, threshold, negative_slop);
			});
		}
		Tensor<T> hinge(T t) {
			return __foreach_elem_assign_([=](T x) {
				T y = 1 - t * x;
				return ((y > 0) ? y : 0);
			});
		}
		Tensor<T> tanh() {
			return __foreach_elem_assign_([=](T x) {
				T exp_p = __exp_(x);
				T exp_n = __exp_(-x);
				return ((exp_p - exp_n) / (exp_p + exp_n));
			});
		}
		Tensor<T> neg() {
			return __foreach_elem_assign_([=](T x) {
				return -x;
			});
		}
		Tensor<T> randomize() {
			foreach_elem_assign([](int i) {
				return RANDOM;
			});
			return (*this);
		}

		// slice
		Tensor<T> slice(int start, int end, int dim) {
			// frame, width(column), height(row), channel. 
			Shape shape_out = shape;
			shape_out.set(end - start, dim);
			Tensor<T> out(shape_out);
			// slice
			switch (dim) {
			case 0:// sample
				out.foreach_assign([&](int i, int j, int k, int l, int m) {
					return this->at(i + start, j, k, l, m);
				});
				break;
			case 1:// frame
				out.foreach_assign([&](int i, int j, int k, int l, int m) {
					return this->at(i, j + start, k, l, m);
				});
				break;
			case 2:// width
				out.foreach_assign([&](int i, int j, int k, int l, int m) {
					return this->at(i, j, k + start, l, m);
				});
				break;
			case 3:// height
				out.foreach_assign([&](int i, int j, int k, int l, int m) {
					return this->at(i, j, k, l + start, m);
				});
				break;
			case 4:// channel
				out.foreach_assign([&](int i, int j, int k, int l, int m) {
					return this->at(i, j, k, l, m + start);
				});
				break;
			default:
				break;
			}
			return out;
		}
		
		// matrix operation
		Tensor<T>& operator=(Tensor<T> &tensor) {			
			if (length() != tensor.length()) {
				shape = tensor.getShape();
				__free_();
				__allocate_();
			}
			this->foreach_assign([&](int ii, int ij, int ik, int il, int im) {
				return tensor.at(ii, ij, ik, il, im);
			});
			return (*this);
		}
		bool operator ==(Tensor<T> &a) {
			bool result = true;
			foreach_elem([&](int i) {
				if (fabs(this->get(i) - a.get(i)) > 10e-6) {
					result = false;
				}
			});
			return result;
		}
		
		void print() {
			shape.print();
			foreach([&](int i, int j, int k, int l, int m)->void {
				if (l == 0 && m == 0) {
					printf("Tensor (%d, %d, %d :, :)\n", i, j, k);
				}
				printf("%5.2f\t", this->at(i, j, k, l, m));
				if (m == shape[4] - 1) {
					printf("\n");
				}
			});
		}

		// static method
		static Tensor<T> random(Shape &shape) {
			Tensor<T> out(shape);
			out.foreach_elem_assign([&](int i) {
				return RANDOM;
			});
			return out;
		}
		static Tensor<T> numbers(Shape &shape, T value) {
			Tensor<T> out(shape);
			out.foreach_elem_assign([&](int i) {
				return value;
			});
			return out;
		}
		static Tensor<T> ones(Shape &shape) {
			Tensor<T> out(shape);
			out.foreach_elem_assign([&](int i) {
				return 1;
			});
			return out;
		}
		static Tensor<T> zeros(Shape &shape) {
			Tensor<T> out(shape);
			out.foreach_elem_assign([](int i) {
				return 0;
			});
			return out;
		}
		static Tensor<T> eye(int n) {
			int size[] = { 1, 1, 1, n, n };
			Shape shape(size);
			Tensor<T> out(shape);
			out.foreach_assign([](int oi, int oj, int ok, int ol, int om) {
				return ((ol == om) ? 1 : 0);
			});
			return out;
		}
		static Tensor<T> mask(Shape &shape, double rate) {
			Tensor<T> out = Tensor<T>::ones(shape);
			out.foreach_elem_assign([=](int i) {
				return (RANDOM < rate) ? 0 : 1;
			});
			return out;
		}

		// file input/output
		void load(string path) {
			ifstream inf;
			inf.open(path, ios::in);
			if (inf.is_open()) {
				inf >> (*this);
				inf.close();
			}
		}
		void save(string path) {
			ofstream outf;
			outf.open(path, ios::out);
			if (outf.is_open()) {
				outf << (*this);
				outf.close();
			}
		}

		// serialize & deserialize
		friend istream& operator >> (istream &in, Tensor<T> &tensor) {
			in >> tensor.shape;
			// re-allocate
			tensor.__allocate_();
			tensor.foreach_assign([&](int i, int j, int k, int l, int m) {
				T value;
				in >> setiosflags(ios::basefield) >> setprecision(18) >> value;
				return value;
			});
			return in;
		}
		friend ostream& operator << (ostream &out, Tensor<T> &tensor) {
			out << tensor.shape << endl;
			tensor.foreach([&](int i, int j, int k, int l, int m) {
				T value = tensor.at(i, j, k, l, m);
				out << setiosflags(ios::basefield) << setprecision(18) << value << " ";
			});
			return out;
		}
	};

	// scalar matrix functions
	template<class T, class Type>
	Tensor<T> foreach_elem(T x, Tensor<T> &y, Type func) {
		Tensor<T> out = Tensor<T>::zeros(y.getShape());
		out.foreach_elem_assign([&](int i) { 
			return func(x, y.get(i));
		});
		return out;
	}

	template<class T>
	Tensor<T> operator +(T x, Tensor<T> &y) {
		return foreach_elem(x, y, [](T a, T b) { return a + b; });
	}

	template<class T>
	Tensor<T> operator -(T x, Tensor<T> &y) {
		return foreach_elem(x, y, [](T a, T b) { return a - b; });
	}

	template<class T>
	Tensor<T> operator *(T x, Tensor<T> &y) {
		return foreach_elem(x, y, [](T a, T b) { return a * b; });
	}

	template<class T>
	Tensor<T> operator /(T x, Tensor<T> &y) {
		return foreach_elem(x, y, [](T a, T b) { return a / b; });
	}
	
	template<class T>
	void test_basic();

	template<class T>
	void test_conv();

	template<class T>
	void test_pooling();
}

#endif // !_TENSOR_H_
