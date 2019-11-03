#pragma once

#ifndef _SHAPE_H_
#define _SHAPE_H_

#include <iostream>
#include <fstream>

namespace shape {

	using namespace std;

	namespace oldshape {

		class Shape {
		private:
			int dims[5] = { 0, 0, 0, 0, 0 };
		public:
			Shape() { 	}
			Shape(const Shape &s) {
				dims[0] = s.dims[0]; dims[1] = s.dims[1];
				dims[2] = s.dims[2]; dims[3] = s.dims[3];
				dims[4] = s.dims[4];
			}
			Shape(int size[]) {
				dims[0] = size[0]; dims[1] = size[1];
				dims[2] = size[2]; dims[3] = size[3];
				dims[4] = size[4];
			}
			Shape(int i, int j, int k, int l, int m) {
				dims[0] = i; dims[1] = j;
				dims[2] = k; dims[3] = l;
				dims[4] = m;
			}
			void print() {
				printf_s("Shape(%d, %d, %d, %d, %d)\n", dims[0], dims[1], dims[2], dims[3], dims[4]);
			}
			inline void set(int k, int axis) {
				dims[axis] = k;
			}
			inline void set(int i, int j, int k, int l, int m) {
				dims[0] = i;
				dims[1] = j;
				dims[2] = k;
				dims[3] = l;
				dims[4] = m;
			}
			Shape& flatten(int axis=2) {
				switch (axis) {
				case 0: // merge all dims to one dimension
					set(1, 1, 1, 1, dims[0] * dims[1] * dims[2] * dims[3] * dims[4]);
					break;
				case 1: // merge last four dimensions
					set(1, 1, 1, dims[0], dims[1] * dims[2] * dims[3] * dims[4]);
					break;
				case 2: // merge last three dimensions
					set(1, 1, dims[0], dims[1], dims[2] * dims[3] * dims[4]);
					break;
				case 3: // merge last two dimensions
					set(1, dims[0], dims[1], dims[2], dims[3] * dims[4]);
					break;
				default:
					break;// do not merge
				}
				return (*this);
			}
			int size() {
				return (dims[0] * dims[1] * dims[2] * dims[3] * dims[4]);
			}
			inline int sub2ind(int i, int j, int k, int l, int m) const {
				return ((((i*dims[1] + j)*dims[2] + k)*dims[3] + l)*dims[4] + m);
			}
			inline int sub2ind(int subs[]) const  {
				return sub2ind(subs[0], subs[1], subs[2], subs[3], subs[4]);
			}
			inline int* ind2sub(int idx) {
				int *sub = new int[5];
				for (int i = 4; i >= 0; i--) {
					sub[i] = idx % dims[i];
					idx /= dims[i];
				}
				return sub;
			}
			inline int operator[](int k) const {
				return dims[k];
			}
			inline int operator() (int i, int j, int k, int l, int m) {
				return sub2ind(i, j, k, l, m);
			}
			bool operator==(Shape &shape) {
				for (int i = 0; i < 5; i++) {
					if (shape.dims[i] != dims[i]) {
						return false;
					}
				}
				return true;
			}
			friend istream& operator >> (istream& in, Shape &shape) {
				for (int i = 0; i < 5; i++) {
					in >> shape.dims[i];
				}
				return in;
			}
			friend ostream& operator << (ostream& out, Shape &shape) {
				for (int i = 0; i < 5; i++) {
					out << shape.dims[i] << " ";
				}
				return out;
			}
		};
	}

	namespace newshape {

		class Shape {
		private:
			int n_samples;
			int n_frames;
			int n_width;
			int n_height;
			int n_channels;
		public:
			Shape() { ; }
			Shape(Shape &s) {
				n_samples = s.n_samples;
				n_frames = s.n_frames;
				n_width = s.n_width;
				n_height = s.n_height;
				n_channels = s.n_channels;
			}
			Shape(int size[]) {
				n_samples = size[0];
				n_frames = size[1];
				n_width = size[2];
				n_height = size[3];
				n_channels = size[4];
			}
			Shape(int s, int f, int w, int h, int c) {
				n_samples = s;
				n_frames = f;
				n_width = w;
				n_height = h;
				n_channels = c;
			}
			void print() {
				printf_s("Shape(%d, %d, %d, %d, %d)\n", n_samples, n_frames, n_width, n_height, n_channels);
			}
			inline void set(int k, int axis) {
				switch (axis) {
				case 0: n_samples = k; break;
				case 1: n_frames = k; break;
				case 2: n_width = k; break;
				case 3: n_height = k; break;
				case 4: n_channels = k; break;
				default: n_channels = k; break;
				}
			}
			inline void set(int s, int f, int w, int h, int c) {
				n_samples = s;
				n_frames = f;
				n_width = w;
				n_height = h;
				n_channels = c;
			}
			inline int operator[](int k) const {
				int dims[] = { n_samples, n_frames, n_width, n_height, n_channels };
				return dims[k];
			}
			Shape& flatten(int axis = 3) {
				int m_channels;
				switch (axis) {
				case 0: // merge all dimensions
					m_channels = n_samples * n_frames * n_width * n_height * n_channels;
					set(1, 1, 1, 1, n_channels);
					break;
				case 1: // merge last four dimensions
					m_channels = n_frames * n_width * n_height * n_channels;
					set(1, 1, 1, n_samples, n_channels);
					break;
				case 2: // merge last three dimensions
					m_channels = n_width * n_height * n_channels;
					set(1, 1, n_samples, n_frames, m_channels);
					break;
				case 3: // merge last two dimensions
					m_channels = n_height * n_channels;
					set(1, n_samples, n_frames, n_width, m_channels);
					break;
				default: // do not merge
					break;
				}
				return (*this);
			}
			int size() {
				return (n_samples * n_frames * n_width * n_height * n_channels);
			}
			inline int sub2ind(int s, int f, int w, int h, int c) {
				return (((((s*n_frames) + f)*n_width + w)*n_height + h)*n_channels) + c;
			}
			inline int* ind2sub(int idx) {
				int dims[] = { n_samples, n_frames, n_width, n_height, n_channels };
				int *sub = new int[5];
				for (int i = 4; i >= 0; i--) {
					sub[i] = idx % dims[i];
					idx /= dims[i];
				}
				return sub;
			}
			inline int operator() (int s, int f, int w, int h, int c) {
				return sub2ind(s, f, w, h, c);
			}
			friend istream& operator >> (istream& in, Shape &shape) {
				in >> shape.n_samples;
				in >> shape.n_frames;
				in >> shape.n_width;
				in >> shape.n_height;
				in >> shape.n_channels;
				return in;
			}
			friend ostream& operator << (ostream& out, Shape &shape) {
				out << shape.n_samples;
				out << shape.n_frames;
				out << shape.n_width;
				out << shape.n_height;
				out << shape.n_channels;
				return out;
			}
		};
	}
}
#endif // !_SHAPE_H