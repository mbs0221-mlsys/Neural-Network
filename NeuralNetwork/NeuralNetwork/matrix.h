#pragma once

#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
inline T __sigmoid_(T x) { return 1 / (1 + exp(x)); }

template<class T>
inline T __pow_(T x, int y) { return pow(x, y); }

template<class T>
inline T __relu_(T x) { return max(x, 0); }

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

// Matrix class definition
template<class T>
class Matrix {
private:

	// broadcast-ops
	void __broadcast_by_column_(Matrix<T> &out, Matrix<T> &m, T (*func)(T, T)) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				out.data[i][j] = func(data[i][j], m.data[i][0]);
			}
		}
	}
	void __broadcast_by_row_(Matrix<T> &out, Matrix<T> &m, T (*func)(T, T)) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				out.data[i][j] = func(data[i][j], m.data[0][j]);
			}
		}
	}
	void __element_wise_ops_(Matrix<T> &out, Matrix<T> &m, T (*func)(T, T)) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				out.data[i][j] = func(data[i][j], m.data[i][j]);
			}
		}
	}
	
	template<class K>
	void __element_wise_ops_(Matrix<T> &out, K func) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				out.data[i][j] = func(data[i][j]);
			}
		}
	}
	
	// __slice_
	Matrix<T> __slice_by_row_(int start, int end) {
		Matrix<T> out(end - start, col);
		for (int i = start; i < end; i++) {
			for (int j = 0; j < col; j++) {
				out.data[i - start][j] = data[i][j];
			}
		}
		return out;
	}
	Matrix<T> __slice_by_column_(int start, int end) {
		Matrix<T> out(row, end - start);
		for (int i = 0; i < row; i++) {
			for (int j = start; j < end; j++) {
				out.data[i][j - start] = data[i][j];
			}
		}
		return out;
	}
	
	// __reduce_
	Matrix<T> __reduce_by_row_() {
		Matrix<T> out(1, col); // reduce to a row
		for (int j = 0; j < col; j++) {
			double t = 0;
			for (int i = 0; i < row; i++) {
				t = t + data[i][j];
			}
			out.data[0][j] = t;
		}
		return out;
	}
	Matrix<T> __reduce_by_column_() {
		Matrix<T> out(row, 1); // reduce to a column
		for (int i = 0; i < row; i++) {
			double t = 0;
			for (int j = 0; j < col; j++) {
				t = t + data[i][j];
			}
			out.data[i][0] = t;
		}
		return out;
	}

	// __allocate_
	void __free_() {
		if (data == nullptr)
			return;
		for (int i = 0; i < row; i++) {
			if (data[i] == nullptr)
				continue;
			delete data[i];
		}
		delete[] data;
	}
	void __reallocate_(int m_row, int m_col) {
		if (row != m_row || col != m_col) {			
			row = m_row, col = m_col;
			data = new T*[row];
			for (int i = 0; i < row; i++) {
				data[i] = new T[col];
			}
		}
	}

protected:
	//template<class K>
	Matrix<T> element_wise_ops(Matrix<T> &m, T (*func)(T, T)) {
		Matrix<T> out(row, col);
		if (m.col == 1) {
			__broadcast_by_column_(out, m, func);
		} else if (m.row == 1) {		
			__broadcast_by_row_(out, m, func);
		} else {
			__element_wise_ops_(out, m, func);
		}
		return out;
	}
	template<class K>
	Matrix<T> element_wise_ops(K func) {
		Matrix<T> out(row, col);
		__element_wise_ops_(out, func);
		return out;
	}

public:
	char name[8];
	int row;
	int col;
	T **data;

	// constructor
	Matrix() {
		strcpy_s(name, "temp");
		row = col = 0;
		data = nullptr;
	}
	Matrix(int row, int col) {
		strcpy_s(name, "temp");
		this->row = row;
		this->col = col;
		data = new T*[row];
		for (int i = 0; i < row; i++) {
			data[i] = new T[col];
		}
	}
	Matrix(const Matrix<T> &m) {
		// re-allocate
		__reallocate_(m.row, m.col);
		strcpy_s(name, m.name);
		// load data
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				data[i][j] = m.data[i][j];
			}
		}
	}
	Matrix(Shape &shape) {
		strcpy_s(name, "temp");
		row = shape[0];
		col = shape[1];
		data = new T*[row];
		for (int i = 0; i < row; i++) {
			data[i] = new T[col];
		}
	}
	
	// destructor
	~Matrix() {
		//__free_();
	}
	
	// operators
	Matrix<T> one_hot() {
		int size[] = { row, 10 };
		Shape shape(size);
		Matrix<T> m = Matrix<T>::zeros(shape);
		for (int i = 0; i < m.row; i++) {
			int k = (int) data[i][0];
			m.data[i][k] = 1.0f;
		}
		return m;
	}
	Matrix<T> add(Matrix<T> &m) {
		return (*this) + m;
	}
	Matrix<T> sub(Matrix<T> &m) {
		return (*this) - m;
	}
	Matrix<T> matmul(Matrix<T> &m) {
		Matrix<T> out(row, m.col);
		for (int k = 0; k < m.col; k++) {
			for (int i = 0; i < row; i++) {
				double t = 0;
				for (int j = 0; j < col; j++) {
					t += data[i][j] * m.data[j][k];
				}
				out.data[i][k] = t;
			}
		}
		return out;
	}
	Matrix<T> Transpose() {
		Matrix<T> out(col, row);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				out.data[j][i] = data[i][j];
			}
		}
		return out;
	}
	
	// reduce
	Matrix<T> reduce_sum(int axis) {
		return (axis == 0) ? __reduce_by_row_() : __reduce_by_column_();
	}
	Matrix<T> reduce_mean(int axis) {
		Matrix<T> out = reduce_sum(axis);
		return (axis == 0) ? (out / row) : (out / col);
	}

	// element-wise-ops
	Matrix<T> operator +(T b) {	return element_wise_ops([=](T x) { return x + b; }); }
	Matrix<T> operator -(T b) {	return element_wise_ops([=](T x) { return x - b; }); }
	Matrix<T> operator *(T b) {	return element_wise_ops([=](T x) { return x * b; }); }
	Matrix<T> operator /(T b) {	return element_wise_ops([=](T x) { return x / b; });	}
	
	// element-wise-ops
	Matrix<T> sigmoid() { return element_wise_ops([=](T x) { return __sigmoid_(x); }); }
	Matrix<T> exp() { return element_wise_ops([=](T x) { return __exp_(x); }); }
	Matrix<T> pow(int k) { return element_wise_ops([=](T x) { return __pow_(x, k); }); }
	Matrix<T> relu() { return element_wise_ops([=](T x) { return __relu_(x); }); }
	Matrix<T> relu(double max_value, double threshold = 0.0f, double negative_slop=0.1f) {
		return element_wise_ops([=](T x) {
			return relu(x, max_value, threshold, negative_slop);
		});
	}
	Matrix<T> hinge(T t) {		
		return element_wise_ops(x, [=](T x) {
			int y = 1 - t * x;
			return max(0, y);
		});
	}
	Matrix<T> tanh() {
		return element_wise_ops([=](T x) {
			return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
		});
	}
	Matrix<T> neg() {
		return element_wise_ops([=](T x) {
			return -x;
		})
	}

	// _matrix_matrix_
	Matrix<T> operator +(Matrix<T> &x) {
		return element_wise_ops(x, [](T a, T b) { return a + b; });
	}	
	Matrix<T> operator -(Matrix<T> &x) {
		return element_wise_ops(x, [](T a, T b) { return a - b; });
	}
	Matrix<T> operator *(Matrix<T> &x) {
		return element_wise_ops(x, [](T a, T b) { return a * b; });
	}
	Matrix<T> operator /(Matrix<T> &x) {
		return element_wise_ops(x, [](T a, T b) { return a / b; });
	}

	// assign operator
	Matrix<T>& operator =(Matrix<T> &x) {
		__reallocate_(x.row, x.col);
		strcpy_s(name, x.name);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				data[i][j] = x.data[i][j];
			}
		}
		return (*this);
	}

	// slice
	Matrix<T> slice(int start, int end, int axis) {
		if (axis == 0) {
			return __slice_by_row_(start, end);
		} else {
			return __slice_by_column_(start, end);
		}
	}
	
	// matrix operation	
	bool operator ==(Matrix<T> &a) {
		bool result = true;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (fabs(data[i][j] - a.data[i][j]) > 10e-6) {
					result = false;
					break;
				}
			}
		}
		return result;
	}
	void randomize() {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				data[i][j] = RANDOM;
			}
		}
	}
	void print() {
		printf_s("print(%d, %d, %s)\n", row, col, name);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				printf("%.2f\t", data[i][j]);
			}
			printf("\n");
		}
	}
	void print_shape() {
		printf_s("Matrix(%d, %d)\n", row, col);
	}
	Shape shape() {
		int size[] = { row, col, 0, 0 };
		return Shape(size);
	}

	// serialize
	void load(FILE *fp) {
		int m_row, m_col;
		fscanf_s(fp, "%d %d %s", &m_row, &m_col, &name);
		// re-allocate
		__reallocate_(m_row, m_col);
		// load data
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				fscanf_s(fp, "%lf", &data[i][j]);
			}
		}
	}
	void save(FILE *fp) {
		fprintf_s(fp, "%d %d %s\n", row, col, name);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				fprintf_s(fp, "%lf ", data[i][j]);
			}
			fprintf_s(fp, "\n");
		}
	}
	void loadmat(char *path) {
		FILE file;
		FILE *fp = &file;
		fopen_s(&fp, path, "r");
		load(fp);
		fclose(fp);
	}
	void savemat(char *path) {
		FILE file;
		FILE *fp = &file;
		fopen_s(&fp, path, "w");
		save(fp);
		fclose(fp);
	}
	
	// static method
	static Matrix<T> ones(Shape &shape) {
		Matrix<T> out(shape);
		for (int i = 0; i < out.row; i++) {
			for (int j = 0; j < out.col; j++) {
				out.data[i][j] = 1.0f;
			}
		}
		return out;
	}
	static Matrix<T> zeros(Shape &shape) {
		Matrix<T> out(shape);
		for (int i = 0; i < out.row; i++) {
			for (int j = 0; j < out.col; j++) {
				out.data[i][j] = 0.0f;
			}
		}
		return out;
	}
	static Matrix<T> eye(int n) {
		Matrix<T> out(n, n);
		for (int i = 0; i < out.row; i++) {
			for (int j = 0; j < out.col; j++) {
				out.data[i][j] = (i == j ? 1.0 : 0.0);
			}
		}
		return out;
	}
	static Matrix<T> mask(Shape &shape, double rate, int axis = 0) {
		Matrix<T> out = Matrix<T>::ones(shape);
		if (axis == 0) {
			for (int i = 0; i < out.row; i++) {
				if (RANDOM < rate) {
					for (int j = 0; j < out.col; j++) {
						out.data[i][j] = 0;
					}
				}
			}
		}
		if (axis == 1) {
			for (int j = 0; j < out.col; j++) {
				if (RANDOM < rate) {
					for (int i = 0; i < out.row; i++) {
						out.data[i][j] = 0;
					}
				}
			}
		}
		return out;
	}
	
	// _scalar_matrix_
	Matrix<T> _scalar_matrix_(T b, T(*func)(T, T)) {
		Matrix<T> out(row, col);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				out.data[i][j] = func(b, data[i][j]);
			}
		}
		return out;
	}
	
};

// friend functions
template<class T>
Matrix<T> operator +(T x, Matrix<T> &y) {
	return y._scalar_matrix_(x, [](T a, T b) { return a + b; });
}

template<class T>
Matrix<T> operator -(T x, Matrix<T> &y) {
	return y._scalar_matrix_(x, [](T a, T b) { return a - b; });
}

template<class T>
Matrix<T> operator *(T x, Matrix<T> &y) {
	return y._scalar_matrix_(x, [](T a, T b) { return a * b; });
}

template<class T>
Matrix<T> operator /(T x, Matrix<T> &y) {
	return y._scalar_matrix_(x, [](T a, T b) { return a / b; });
}

template<class T>
void print(Matrix<T> &x, Matrix<T> &y, int sz) {
	for (int i = 0; i < x.row; i++) {
		//COLORREF Color = HSLtoRGB(y.data[0][j], 1, 1);
		COLORREF Color = y.data[i][0] > 0.5 ? RED : GREEN;
		setlinecolor(Color);
		circle((int)(x.data[i][0] * 60) + 200, (int)(x.data[i][1] * 20) + 200, sz);
	}
}

#endif // !_MATRIX_H_