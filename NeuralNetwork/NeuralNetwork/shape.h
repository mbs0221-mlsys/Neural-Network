#pragma once

#ifndef _SHAPE_H_
#define _SHAPE_H_

#include <iostream>
#include <fstream>

namespace shape {

	using namespace std;

	class Shape {
	private:
		int dims[4] = { 0, 0, 0, 0 };

	public:
		Shape() { 	}
		Shape(Shape &s) {
			dims[0] = s.dims[0];
			dims[1] = s.dims[1];
			dims[2] = s.dims[2];
			dims[3] = s.dims[3];
		}
		Shape(int size[]) {
			dims[0] = size[0];
			dims[1] = size[1];
			dims[2] = size[2];
			dims[3] = size[3];
		}
		void print() {
			printf_s("Shape(%d, %d, %d, %d)\n", dims[0], dims[1], dims[2], dims[3]);
		}
		inline void set(int k, int axis) {
			dims[axis] = k;
		}
		inline void set(int i, int j, int k, int l) {
			dims[0] = i;
			dims[1] = j;
			dims[2] = k;
			dims[3] = l;
		}
		inline int operator[](int k) const  {
			return dims[k];
		}
		Shape& flatten(int axis) {
			if (axis == 0) {
				// merge all dims to one dimension
				set(1, 1, 1, dims[0] * dims[1] * dims[2] * dims[3]);
			}
			else if (axis == 1) {
				// merge last three dimensions
				set(1, 1, dims[0], dims[1] * dims[2] * dims[3]);
			}
			else {
				// merge last two dimensions
				set(1, dims[0], dims[1], dims[2] * dims[3]);
			}
			return (*this);
		}
		int size() {
			return (dims[0] * dims[1] * dims[2] * dims[3]);
		}
		inline int sub2ind(int i, int j, int k, int l) {
			return ((((i*dims[1]) + j)*dims[2] + k)*dims[3] + l);
		}
		inline int* ind2sub(int idx) {
			int *sub = new int[4];
			for (int i = 3; i >= 0; i--) {
				sub[i] = idx % dims[i];
				idx /= dims[i];
			}
			return sub;
		}
		inline int operator() (int i, int j, int k, int l) {
			return sub2ind(i, j, k, l);
		}
		friend istream& operator >> (istream& in, Shape &shape) {
			for (int i = 0; i < 4; i++) {
				in >> shape.dims[i];
			}
			return in;
		}
		friend ostream& operator << (ostream& out, const Shape &shape) {
			for (int i = 0; i < 4; i++) {
				out << shape[i] << " ";
			}
			return out;
		}
	};
}
#endif // !_SHAPE_H