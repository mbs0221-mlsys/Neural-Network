#pragma once

#ifndef _TENSOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <easyx.h>
#include <graphics.h>

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
inline T __pow_(T x, int y) { return pow(x, y); }

template<class T>
inline T __relu_(T x) { return max(x, 0); }

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
	using namespace shape;

	// Tensor definition
	template<class T>
	class Tensor {
	private:
		// attributes
		string name;
		Shape shape;
		T *data;

		// __allocate_
		void __free_() {
			if (data != nullptr) {
				delete[] data;
			}
		}
		void __allocate_() {
			try {
				data = new T[length()];
			} catch (const bad_alloc & e){
				cerr << e.what() << endl;
			}
		}
	protected:

		//template<class K>
		Tensor<T> element_wise_ops(Tensor<T> &tensor, T(*func)(T, T)) {
			Shape m_shape = tensor.getShape();
			Tensor<T> out(m_shape);
			int axis = -1;
			for (int i = 0; i < 4; i++) {
				if (shape[i] != m_shape[i]) {
					axis = i;
					break;
				}
			}
			switch (axis) {
			case 0: // foreach frame
				out.foreach_assign([&](int f, int i, int j, int c) {
					return func(this->getValue(f, i, j, c), tensor.getValue(0, i, j, c));
				});
				break;
			case 1: // foreach column
				out.foreach_assign([&](int f, int i, int j, int c) {
					return func(this->getValue(f, i, j, c), tensor.getValue(f, 0, j, c));
				});
				break;
			case 2: // foreach row
				out.foreach_assign([&](int f, int i, int j, int c) {
					return func(this->getValue(f, i, j, c), tensor.getValue(f, i, 0, c));
				});
				break;
			case 3: // foreach channel
				out.foreach_assign([&](int f, int i, int j, int c) {
					return func(this->getValue(f, i, j, c), tensor.getValue(f, i, j, 0));
				});
				break;
			default: // foreach element
				out.foreach_assign([&](int f, int i, int j, int c) {
					return func(this->getValue(f, i, j, c), tensor.getValue(f, i, j, c));
				});
				break;
			}
			return out;
		}
		template<class K>
		Tensor<T> element_wise_ops(K func) {
			Tensor<T> out(shape);
			out.foreach_assign([&](int f, int i, int j, int c) {
				return func(this->getValue(f, i, j, c));
			});
			return out;
		}

	public:

		// constructor
		Tensor() : name("temp") {
			data = nullptr;
		}
		Tensor(int f, int w, int h, int c) : name("temp") {
			int size[] = { f, w, h, c };
			shape = Shape(size);
			__allocate_();
		}
		Tensor(int size[]) : name("temp") {
			shape = Shape(size);
			__allocate_();
		}
		Tensor(Shape shape) : name("temp"), shape(shape) {
			__allocate_();
		}
		Tensor(Tensor<T> &tensor) : name("temp") {
			shape = tensor.getShape();
			__allocate_();
			foreach_assign([&](int i, int j, int k, int l) {
				return tensor.getValue(i, j, k, l);
			});
		}
		~Tensor() {
			name.clear();
			__free_();
		}

		// get & set methods
		Shape getShape() { return shape; }
		string getName() { return name; }
		int size() { return (sizeof(T)*shape.size()); }
		int length() { return shape.size(); }

		// 对所有元素执行相同的操作
		template<class K>
		void foreach(K func) {
			//for (int i = 0; i < shape.size(); i++) {
			//	int *sub = shape.ind2sub(i);				
			//	func(sub[0], sub[1], sub[2], sub[3]);
			//}
			 
			for (int i = 0; i < shape[0]; i++) {// frame
				for (int j = 0; j < shape[1]; j++) {// column(width)
					for (int k = 0; k < shape[2]; k++) {// row(height)
						for (int l = 0; l < shape[3]; l++) {// depth(channel)
							func(i, j, k, l);
						}
					}
				}
			}
		}
		template<class K>
		void foreach_assign(K func) {
			foreach([&](int i, int j, int k, int l) {
				int idx = shape.sub2ind(i, j, k, l);
				data[idx] = func(i, j, k, l);
			});
		}

		// operators
		Tensor<T> one_hot(int num) {
			int size[] = { shape[0], shape[1], shape[2], num };// (:, :, row, col).
			Shape shape(size);
			// one-hot编码
			map<int, int> codes;
			Tensor<T> out = Tensor<T>::zeros(shape);
			out.foreach([&](int i, int j, int k, int l) {
				int value = (int)(this->getValue(i, j, k, l));
				// 维护一个one-hot映射表
				if (codes.find(value) == codes.end()) {
					codes[value] = codes.size();
				}
				// 查找对应编码
				out.setValue(1.0f, i, j, k, codes[value]);
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
			// calculate this(:,:,ok,col)*(0,0,col,ol)
			int n_cols = shape[3];// n_cols = shape[3] = shape_b[2];
			Tensor<T> out(shape[0], shape[1], shape[2], shape_b[3]);
			out.foreach_assign([&](int oi, int oj, int ok, int ol) {
				T value = 0;
				for (int col = 0; col < n_cols; col++) {
					value += this->getValue(oi, oj, ok, col)*tensor.getValue(0, 0, col, ol);// broadcast
				}
				return value;
			});
			return out;
		}
		Tensor<T> Transpose() {
			int size[] = { shape[0], shape[1], shape[3], shape[2] };
			Shape shape(size);
			Tensor<T> out(shape);
			out.foreach_assign([&](int i, int j, int k, int l) {
				return this->getValue(i, j, l, k);
			});
			return out;
		}
		Tensor<T> permute(int order[]) {
			// 计算维度
			int size[] = { 0, 0, 0, 0 };
			for (int i = 0; i < 4; i++) {
				size[i] = shape[order[i]];
			}
			Shape shape_out(size);
			Tensor<T> out = Tensor<T>::zeros(shape_out);
			// TODO: 置换操作
			out.foreach_assign([&](int i, int j, int k, int l) {
				return (*this)->getValue(i, j, k, l);
			});
		}
		Tensor<T> reduce_sum(int axis) {
			// frame, width(column), height(row), channel. 
			Shape shape_out = shape;
			shape_out.setDims(1, axis);
			Tensor<T> out = Tensor<T>::zeros(shape_out);
			switch (axis) {
			case 0:// frame
				out.foreach([&](int i, int j, int k, int l) {
					T value = out.getValue(0, j, k, l) + this->getValue(i, j, k, l);
					out.setValue(value, 0, j, k, l);
				});
				break;
			case 1:// width
				out.foreach([&](int i, int j, int k, int l) {
					T value = out.getValue(i, 0, k, l) + this->getValue(i, j, k, l);
					out.setValue(value, i, 0, k, l);
				});
				break;
			case 2:// height
				out.foreach([&](int i, int j, int k, int l) {
					T value = out.getValue(i, j, 0, l) + this->getValue(i, j, k, l);
					out.setValue(value, i, j, 0, l);
				});
				break;
			case 3:// channel
				out.foreach([&](int i, int j, int k, int l) {
					T value = out.getValue(i, j, k, 0) + this->getValue(i, j, k, l);
					out.setValue(value, i, j, k, 0);
				});
				break;
			default:
				cout << "error in reduce sum" << endl;
				break;
			}
			return out;
		}
		Tensor<T> reduce_mean(int axis) {
			Tensor<T> out = reduce_sum(axis);
			int N = shape[axis];
			return out / N;
		}

		// scalar operator
		Tensor<T> operator +(T b) { return element_wise_ops([=](T x) { return x + b; }); }
		Tensor<T> operator -(T b) { return element_wise_ops([=](T x) { return x - b; }); }
		Tensor<T> operator *(T b) { return element_wise_ops([=](T x) { return x * b; }); }
		Tensor<T> operator /(T b) { return element_wise_ops([=](T x) { return x / b; }); }

		// matrix operator
		Tensor<T> operator +(Tensor<T> &x) {
			return element_wise_ops(x, [](T a, T b) { return a + b; });
		}
		Tensor<T> operator -(Tensor<T> &x) {
			return element_wise_ops(x, [](T a, T b) { return a - b; });
		}
		Tensor<T> operator *(Tensor<T> &x) {
			return element_wise_ops(x, [](T a, T b) { return a * b; });
		}
		Tensor<T> operator /(Tensor<T> &x) {
			return element_wise_ops(x, [](T a, T b) { return a / b; });
		}

		// operator ()
		inline void setValue(T t, int i) {
			data[i] = t;
		}
		inline void setValue(T t, int i, int j, int k, int l) {
			int idx = shape.sub2ind(i, j, k, l);
			data[idx] = t;
		}
		
		inline T getValue(int i, int j, int k, int l) {
			int idx = shape.sub2ind(i, j, k, l);
			return data[idx];
		}
		inline T getValue(int j, int k, int l) {
			int idx = shape.sub2ind(0, j, k, l);
			return data[idx];
		}
		inline T getValue(int k, int l) {
			int idx = shape.sub2ind(0, 0, k, l);
			return data[idx];
		}
		inline T getValue(int l) {
			int idx = shape.sub2ind(0, 0, 0, l);
			return data[idx];
		}

		inline T operator()(int i, int j, int k, int l) {
			int idx = shape.sub2ind(i, j, k, l);
			return data[idx];
		}
		inline T operator()(int j, int k, int l) {
			int idx = shape.sub2ind(0, j, k, l);
			return data[idx];
		}
		inline T operator()(int k, int l) {
			int idx = shape.sub2ind(0, 0, k, l);
			return data[idx];
		}
		inline T operator()(int l) {
			int idx = shape.sub2ind(0, 0, 0, l);
			return data[idx];
		}
		
		// tensor operation
		Tensor<T> padding(int pad) {
			int size[] = { shape[0], shape[1] + pad*2, shape[2] + pad*2, shape[3] };
			Tensor<T> out = Tensor<T>::zeros(Shape(size));
			// calculate 2d padding
			foreach([&](int ii, int ij, int ik, int il) {
				T value = this->getValue(ii, ij, ik, il);
				out.setValue(value, ii, ij + pad, ik + pad, il);
			});
			return out;
		}
		Tensor<T> conv2d(Tensor<T> filter, int stride) {
			Shape filter_shape = filter.getShape();// (1, width, height, channel)
			int width = (shape[1] - filter_shape[1]) / stride + 1;
			int height = (shape[2] - filter_shape[2]) / stride + 1;
			int size[] = { shape[0], width, height, shape[3] };
			// calculate 2d concolution
			T *value = new T[size[3]];
			Tensor<T> out = Tensor<T>::zeros(Shape(size));
			out.foreach_assign([&](int oi, int oj, int ok, int ol) {
				value[ol] = 0.0f;// for each channel
				filter.foreach([&](int ki, int kj, int kk, int kl) {
					value[kl] += this->getValue(oi, oj*stride + kj, ok*stride + kk, ol + kl)*filter.getValue(0, kj, kk, kl);
				});
				return value[ol];
			});
			return out;
		}

		// math function
		Tensor<T> softmax() { 
			Tensor<T> sum_e = exp().reduce_sum(1); 
			return (*this) / sum_e; 
		}
		Tensor<T> sigmoid() {
			return element_wise_ops([=](T x) {
				return __sigmoid_(x); 
			});
		}
		Tensor<T> exp() { 
			return element_wise_ops([=](T x) {
				return __exp_(x); 
			}); 
		}
		Tensor<T> log() {
			return element_wise_ops([=](T x) { 
				return __log_(x); 
			});
		}
		Tensor<T> pow(int k) { 
			return element_wise_ops([=](T x) { return __pow_(x, k);
			}); 
		}
		Tensor<T> relu() { 
			return element_wise_ops([=](T x) {
				return __relu_(x); 
			});
		}
		Tensor<T> relu(double max_value, double threshold = 0.0f, double negative_slop = 0.1f) {
			return element_wise_ops([=](T x) {
				return relu(x, max_value, threshold, negative_slop);
			});
		}
		Tensor<T> hinge(T t) {
			return element_wise_ops(x, [=](T x) {
				int y = 1 - t * x;
				return max(0, y);
			});
		}
		Tensor<T> tanh() {
			return element_wise_ops([=](T x) {
				return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
			});
		}
		Tensor<T> neg() {
			return element_wise_ops([=](T x) {
				return -x;
			})
		}

		// slice
		Tensor<T> slice(int start, int end, int axis) {
			// frame, width(column), height(row), channel. 
			Shape sp = shape;
			sp[axis] = end - start;
			Tensor<T> out(sp);
			// slice
			switch (axis) {
			case 0:
				out.foreach_assign([&](int i, int j, int k, int l) {
					return this->getValue(i + start, j, k, l);
				});
				break;
			case 1:
				out.foreach_assign([&](int i, int j, int k, int l) {
					return this->getValue(i, j + start, k, l);
				});
				break;
			case 2:
				out.foreach_assign([&](int i, int j, int k, int l) {
					return this->getValue(i, j, k + start, l);
				});
				break;
			case 3:
				out.foreach_assign([&](int i, int j, int k, int l) {
					return this->getValue(i, j, k, l + start);
				});
				break;
			default:
				break;
			}
		}

		// matrix operation	
		bool operator ==(Tensor<T> &a) {
			bool result = true;
			foreach([&](int i, int j, int k, int l) {
				if (fabs(this->getValue(i, j, k, l) - a.getValue(i, j, k, l)) > 10e-6) {
					result = false;
				}
			});
			return result;
		}
		void randomize() {
			foreach_assign([](int i, int j, int k, int l) {
				return RANDOM;
			});
		}
		void print() {
			shape.print();
			foreach([&](int i, int j, int k, int l) {
				T value = this->getValue(i, j, k, l);
				if (k == 0 && l == 0) {
					printf("Tensor (%d, %d, :, :)\n", i, j);
				}
				printf("%5.2f\t", value);
				if (l == shape[3] - 1) {
					printf("\n");
				}
			});
		}

		// static method
		static Tensor<T> ones(Shape &shape) {
			Tensor<T> out(shape);
			out.foreach_assign([](int i, int j, int k, int l) {
				return 1;
			});
			return out;
		}
		static Tensor<T> zeros(Shape &shape) {
			Tensor<T> out(shape);
			out.foreach_assign([](int i, int j, int k, int l) {
				return 0;
			});
			return out;
		}
		static Tensor<T> eye(int n) {
			int size[] = { 0, 0, n, n };
			Shape shape(size);
			Tensor<T> out(shape);
			out.foreach_assign([](int i, int j, int k, int l) {
				return ((k == l) ? 1 : 0);
			});
			return out;
		}
		static Tensor<T> mask(Shape &shape, double rate) {
			Tensor<T> out = Tensor<T>::ones(shape);
			out.foreach_assign([=](int i, int j, int k, int l) {
				return (RANDOM < rate) ? 0 : 1;
			});
			return out;
		}

		// serialize & deserialize
		friend istream& operator >> (istream &in, Tensor<T> &tensor) {
			in >> tensor.shape >> tensor.name;
			// re-allocate
			tensor.__allocate_();
			for (int i = 0; i < tensor.length(); i++) {
				in >> setiosflags(ios::basefield) >> setprecision(18) >> tensor.data[i];
			}
			return in;
		}
		friend ostream& operator << (ostream &out, Tensor<T> &tensor) {
			out << tensor.name << " " << tensor.shape << endl;
			for (int i = 0; i < tensor.length(); i++) {
				out << setiosflags(ios::basefield) << setprecision(18) << tensor.data[i] << " ";
			}
		}

	};

	// scalar matrix functions
	template<class T, class K>
	Tensor<T> foreach_elem(T x, Tensor<T> &y, K func) {
		Tensor<T> out = Tensor<T>::zeros(y.getShape());
		out.foreach_assign([&](int i, int j, int k, int l) {
			return func(x, y.getValue(i, j, k, l));
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
	void test() {
		printf("tensor::test()\n");
		int size[]= { 1,1,4,3 };
		Shape shape(size);
		{
			Tensor<T> x = Tensor<T>::ones(shape);
			x.print();
			Tensor<T> b = Tensor<T>::zeros(shape);
			b.print();
			Tensor<T> c = Tensor<T>::mask(shape, 0.2);
			c.print();
			cout << "c.Transpose().sigmoid().print();" << endl;
			c.Transpose().sigmoid().print();
			cout << "x.matmul(c.Transpose()).reduce_mean(2).reduce_mean(3).print();" << endl;
			x.matmul(c.Transpose()).reduce_mean(2).reduce_mean(3).print();
		}
	}
}

#endif // !_TENSOR_H_
