#pragma once

#ifndef _TENSOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <string>

#include <set>
#include <iostream>
#include <iomanip>
#include <fstream>

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
	using namespace shape;

	// Tensor definition
	template<class T>
	class Tensor {
	private:
		// attributes
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
		Tensor<T> __foreach_assign_(Tensor<T> &tensor, T(*func)(T, T)) {
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
					return func(this->at(f, i, j, c), tensor.at(0, i, j, c));
				});
				break;
			case 1: // foreach column
				out.foreach_assign([&](int f, int i, int j, int c) {
					return func(this->at(f, i, j, c), tensor.at(f, 0, j, c));
				});
				break;
			case 2: // foreach row
				out.foreach_assign([&](int f, int i, int j, int c) {
					return func(this->at(f, i, j, c), tensor.at(f, i, 0, c));
				});
				break;
			case 3: // foreach channel
				out.foreach_assign([&](int f, int i, int j, int c) {
					return func(this->at(f, i, j, c), tensor.at(f, i, j, 0));
				});
				break;
			default: // foreach element
				out.foreach_assign([&](int f, int i, int j, int c) {
					return func(this->at(f, i, j, c), tensor.at(f, i, j, c));
				});
				break;
			}
			return out;
		}
		template<class K>
		Tensor<T> __foreach_elem_assign_(K func) {
			Tensor<T> out(shape);
			out.foreach_elem_assign([&](int i) {
				return func(data[i]);
			});
			return out;
		}

	public:

		// constructor
		Tensor() {
			data = nullptr;
		}
		Tensor(int f, int w, int h, int c) {
			int size[] = { f, w, h, c };
			shape = Shape(size);
			__allocate_();
		}
		Tensor(int size[]) {
			shape = Shape(size);
			__allocate_();
		}
		Tensor(Shape shape) : shape(shape) {
			__allocate_();
		}
		Tensor(Tensor<T> &tensor) : shape(tensor.getShape()) {
			__allocate_();
			foreach_assign([&](int i, int j, int k, int l) {
				return tensor.at(i, j, k, l);
			});
		}
		~Tensor() {
			__free_();
		}

		// get & set methods
		Shape getShape() { return shape; }
		int size() { return (sizeof(T)*shape.size()); }
		int length() { return shape.size(); }

		// non-parallel foreach
		template<class K>
		void foreach(K func) {
			// non-parallel	 
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
			// non-parallel
			foreach([&](int i, int j, int k, int l) {
				int idx = shape.sub2ind(i, j, k, l);
				data[idx] = func(i, j, k, l);
			});
		}
		
		// parallel foreach elem
		template<class K>
		void foreach_elem(K func) {
			// parallel
			for (int i = 0; i < length(); i++) {
				func(i);
			}
		}
		template<class K>
		void foreach_elem_assign(K func) {
			// parallel
			for (int i = 0; i < length(); i++) {
				data[i] = func(i);
			}
		}

		// operators
		Tensor<T> one_hot(int num) {
			int size[] = { shape[0], shape[1], shape[2], num };// (:, :, row, col).
			Shape shape(size);
			// one-hot编码
			map<int, int> codes;
			Tensor<T> out = Tensor<T>::zeros(shape);
			out.foreach([&](int i, int j, int k, int l) {
				int value = (int)(this->at(i, j, k, l));
				// 维护一个one-hot映射表
				if (codes.find(value) == codes.end()) {
					codes[value] = codes.size();
				}
				// 查找对应编码
				out.set(1.0f, i, j, k, codes[value]);
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
					value += this->at(oi, oj, ok, col)*tensor.at(0, 0, col, ol);// broadcast
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
				return this->at(i, j, l, k);
			});
			return out;
		}
		Tensor<T> permute(int order[]) {
			// 计算维度(2 1 3 0)
			int size[] = { 0, 0, 0, 0 };
			for (int i = 0; i < 4; i++) {
				size[i] = shape[order[i]];
			}
			Shape shape_out(size);
			Tensor<T> out = Tensor<T>::zeros(shape_out);
			// 置换操作
			out.foreach_assign([&](int i, int j, int k, int l) {
				int subs[] = { 0, 0, 0, 0 };// subscripts
				subs[order[0]] = i; // 2, i
				subs[order[1]] = j; // 1, j
				subs[order[2]] = k; // 3, k
				subs[order[3]] = l; // 0, l
				return this->at(subs[0], subs[1], subs[2], subs[3]);
			});
			return out;
		}
		Tensor<T> reshape(int size[]) {
			Shape shape_out(size);
			return reshape(shape_out);
		}
		Tensor<T> reshape(Shape &shape_out) {
			Tensor<T> out = Tensor<T>::zeros(shape_out);
			out.foreach_assign([&](int i, int j, int k, int l) {
				int idx = shape_out.sub2ind(i, j, k, l);
				return data[idx];
			});
			return out;
		}
		Tensor<T> flatten(int axis = 2) {
			// merge last two dimensions by default
			int after[4];
			if (axis == 0) {
				// merge all dims to one dimension
				after = { 1, 1, 1, shape[0] * shape[1] * shape[2] * shape[3] };
			} else if (axis == 1) {
				// merge last three dimensions
				after = { 1, 1, shape[0], shape[1] * shape[2] * shape[3] };
			} else {
				// merge last two dimensions
				after = { 1, shape[0], shape[1], shape[2] * shape[3] };
			}
			return reshape(after);
		}
		Tensor<T> reduce_sum(int axis) {
			// frame, width(column), height(row), channel. 
			Shape shape_out = shape;
			shape_out.set(1, axis);
			Tensor<T> out = Tensor<T>::zeros(shape_out);
			switch (axis) {
			case 0:// frame
				foreach([&](int i, int j, int k, int l) {
					T value = out.at(0, j, k, l) + this->at(i, j, k, l);
					out.set(value, 0, j, k, l);
				});
				break;
			case 1:// width
				foreach([&](int i, int j, int k, int l) {
					T value = out.at(i, 0, k, l) + this->at(i, j, k, l);
					out.set(value, i, 0, k, l);
				});
				break;
			case 2:// height
				foreach([&](int i, int j, int k, int l) {
					T value = out.at(i, j, 0, l) + this->at(i, j, k, l);
					out.set(value, i, j, 0, l);
				});
				break;
			case 3:// channel
				foreach([&](int i, int j, int k, int l) {
					T value = out.at(i, j, k, 0) + this->at(i, j, k, l);
					out.set(value, i, j, k, 0);
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
		Tensor<T> operator +(T b) { return __foreach_elem_assign_([=](T x) { return x + b; }); }
		Tensor<T> operator -(T b) { return __foreach_elem_assign_([=](T x) { return x - b; }); }
		Tensor<T> operator *(T b) { return __foreach_elem_assign_([=](T x) { return x * b; }); }
		Tensor<T> operator /(T b) { return __foreach_elem_assign_([=](T x) { return x / b; }); }

		// matrix operator
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

		// operator ()
		inline void set(T t, int i) {
			data[i] = t;
		}
		inline void set(T t, int i, int j, int k, int l) {
			int idx = shape.sub2ind(i, j, k, l);
			data[idx] = t;
		}
		
		inline T at(int i, int j, int k, int l) {
			int idx = shape.sub2ind(i, j, k, l);
			return data[idx];
		}
		inline T at(int j, int k, int l) {
			int idx = shape.sub2ind(0, j, k, l);
			return data[idx];
		}
		inline T at(int k, int l) {
			int idx = shape.sub2ind(0, 0, k, l);
			return data[idx];
		}
		inline T at(int l) {
			int idx = shape.sub2ind(0, 0, 0, l);
			return data[idx];
		}

		inline T get(int idx) { return data[idx]; }

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
				T value = this->at(ii, ij, ik, il);
				out.set(value, ii, ij + pad, ik + pad, il);
			});
			return out;
		}
		Tensor<T> conv2d(Tensor<T> filter, int stride) {
			// calculate output shape
			Shape filter_shape = filter.getShape();// (1, width, height, channel)
			int width = (shape[1] - filter_shape[1]) / stride + 1;// new height
			int height = (shape[2] - filter_shape[2]) / stride + 1;// new height
			int channel = filter_shape[0];// number of filters
			// calculate 2d concolution
			T *value = new T[channel];
			Tensor<T> out = Tensor<T>::zeros(Shape(size));
			out.foreach_assign([&](int oi, int oj, int ok, int ol) {
				if (ol == 0) {
					memset(value, 0.0f, sizeof(T)*channel);
					// calculate for all filters (filter, width, height, channel)
					filter.foreach([&](int ki, int kj, int kk, int kl) {
						// for each filter (:, width, height, channel)
						T a = this->at(oi, oj*stride + kj, ok*stride + kk, kl);
						T b = filter.at(ki, kj, kk, kl);
					});
				}
				// save each result from each filter (frame, width, height, channel)
				return value[ol];//	out.set(value[ol], oi, oj, ok, ol);
			});
			delete[] value;
			return out;
		}
		Tensor<T> conv3d(Tensor<T> filter, int stride) {
			// calculate output shape
			Shape filter_shape = filter.getShape();// (1, width, height, channel)
			int width = (shape[1] - filter_shape[1]) / stride + 1;// new height
			int height = (shape[2] - filter_shape[2]) / stride + 1;// new height
			int channel = filter_shape[0];// number of filters
			int size[] = { shape[0], width, height, channel };
			// calculate 3d concolution
			T *value = new T[channel];			
			Tensor<T> out = Tensor<T>::zeros(Shape(size));
			out.foreach_assign([&](int oi, int oj, int ok, int ol) {
				if (ol ==  0) {
					memset(value, 0.0f, sizeof(T)*channel);
					// calculate for all filters (filter, width, height, channel)
					filter.foreach([&](int ki, int kj, int kk, int kl) {
						// for each filter (:, width, height, channel)
						T a = this->at(oi, oj*stride + kj, ok*stride + kk, kl);
						T b = filter.at(ki, kj, kk, kl);						
						value[ki] = value[ki] + a*b;
					});
				}
				// save each result from each filter (frame, width, height, channel)
				return value[ol];//	out.set(value[ol], oi, oj, ok, ol);
			});
			delete[] value;
			return out;
		}

		// math function
		Tensor<T> softmax() { 
			Tensor<T> sum_e = exp().reduce_sum(1); 
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
				return relu(x, max_value, threshold, negative_slop);
			});
		}
		Tensor<T> hinge(T t) {
			return __foreach_elem_assign_([=](T x) {
				int y = 1 - t * x;
				return max(0, y);
			});
		}
		Tensor<T> tanh() {
			return __foreach_elem_assign_([=](T x) {
				return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
			});
		}
		Tensor<T> neg() {
			return __foreach_elem_assign_([=](T x) {
				return -x;
			})
		}
		Tensor<T> randomize() {
			foreach_elem_assign([](int i) {
				return RANDOM;
			});
			return (*this);
		}

		// slice
		Tensor<T> slice(int start, int end, int axis) {
			// frame, width(column), height(row), channel. 
			Shape shape_out = shape;
			shape_out.set(end - start, axis);
			Tensor<T> out(shape_out);
			// slice
			switch (axis) {
			case 0:
				out.foreach_assign([&](int i, int j, int k, int l) {
					return this->at(i + start, j, k, l);
				});
				break;
			case 1:
				out.foreach_assign([&](int i, int j, int k, int l) {
					return this->at(i, j + start, k, l);
				});
				break;
			case 2:
				out.foreach_assign([&](int i, int j, int k, int l) {
					return this->at(i, j, k + start, l);
				});
				break;
			case 3:
				out.foreach_assign([&](int i, int j, int k, int l) {
					return this->at(i, j, k, l + start);
				});
				break;
			default:
				break;
			}
			return out;
		}
		
		// matrix operation	
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
			foreach([&](int i, int j, int k, int l) {
				T value = this->at(i, j, k, l);
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
			out.foreach_elem_assign([](int i) {
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
			out.foreach_elem_assign([=](int i) {
				return (RANDOM < rate) ? 0 : 1;
			});
			return out;
		}

		// serialize & deserialize
		friend istream& operator >> (istream &in, Tensor<T> &tensor) {
			in >> tensor.shape;
			// re-allocate
			tensor.__allocate_();
			for (int i = 0; i < tensor.length(); i++) {
				in >> setiosflags(ios::basefield) >> setprecision(18) >> tensor.data[i];
			}
			return in;
		}
		friend ostream& operator << (ostream &out, Tensor<T> &tensor) {
			out << tensor.shape << endl;
			for (int i = 0; i < tensor.length(); i++) {
				out << setiosflags(ios::basefield) << setprecision(18) << tensor.data[i] << " ";
			}
		}

	};

	// scalar matrix functions
	template<class T, class K>
	Tensor<T> foreach_elem(T x, Tensor<T> &y, K func) {
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
