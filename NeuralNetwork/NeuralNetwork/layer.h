#pragma once

#ifndef _LAYER_H_
#define _LAYER_H_

#include <map>

#include "ops.h"

namespace layers {

	using namespace std;

	template<class T>
	class Layer {
	private:
		bool require_grad = true;
	protected:
		Shape shape;
		string name;
	public:
		Matrix<T> value;
		Layer<T>* input;
		Layer<T>* output;
		Layer(Layer<T>* input): input(input) { if (input != nullptr) shape = input->shape; }
		Matrix<T> getValue() { return value; }
		Shape getShape() { return shape; }
		virtual Matrix<T> forward() { if(input != nullptr) return input->forward(); }
		virtual void backward(Matrix<T> &delta) {  }
	};

	template<class T>
	class Input : public Layer<T> {
	public:
		Input(const Shape &shape) : Layer<T>::Layer(NULL) { this->shape = shape; }
		void feed(const Matrix<T> &x) { value = x; }
		virtual Matrix<T> forward() { return value; }
	};

	template<class T>
	class Linear : public Layer<T> {
		Matrix<T> w, b;
	public:
		Linear(Layer<T>* input, int n_output) : Layer<T>(input) {
			Shape input_shape = input->getShape();
			int size[] = { input_shape[1], n_output };
			w = Matrix<T>( input_shape[0], n_output);
			b = Matrix<T>(1, n_output);
			shape = Shape(size);
		}
		virtual Matrix<T> forward() {
			Matrix<T> x = Layer<T>::forward();
			value = x.matmul(w) + b;
			return value;
		}
		virtual void backward(Matrix<T> &delta) {
			Matrix<T> x = input->getValue();
			auto grad_w = x.Transpose().matmul(delta);
			auto grad_b = delta.reduce_sum(0);
			input->backward(delta.matmul(w.Transpose()));
		}
	};

	template<class T>
	class Sigmoid : public Layer<T> {
		Matrix<T> ones;
	public:
		Sigmoid(Layer<T>* input) : Layer<T>(input) {
			shape = Shape(input->shape);
			ones = Matrix<T>::ones(shape);
		}
		virtual Matrix<T> forward() {
			Matrix<T> x = Layer<T>::forward();
			value = x.sigmoid();
			return value;
		}
		virtual void backward(Matrix<T> &delta) {
			input->backward(delta * ops::grad_sigmoid(value));
		}
	};

	template<class T>
	class ReLU : public Layer<T> {
	public:
		ReLU(Layer<T>* input) : Layer<T>(input) {

		}
		virtual Matrix<T> forward() {
			Matrix<T> x = Layer<T>::forward();
			value = x.relu();
			return value;
		}
		virtual void backward(Matrix<T> &delta) {
			Matrix<T> x = input->getValue();
			input->backward(delta * ops::grad_relu(x));
		}
	};

	template<class T>
	class LeakyReLU : public Layer<T> {
	private:
		double max_value;
		double threshold;
		double negative_slope;
	public:
		LeakyReLU(Layer<T> *input, double max_value, double threshold, double negative_slope)
			: Layer<T>(input), max_value(max_value), threshold(threshold), negative_slope(negative_slope) {

		}
		virtual Matrix<T> forward() {
			Matrix<T> x = Layer<T>::forward();
			o = ops::relu(x, max_value, threshold, negative_slope);
		}
		virtual void backward(Matrix<T> &delta) {

		}
	};

	template<class T>
	class Loss : Layer<T> {
	private:
		Matrix<T> y;
	public:
		Loss(Layer<T> *input, Matrix<T> y) :Layer<T>(input), y(y) { ; }
		double getLossValue() { return value.data[0][0]; }
		virtual void forward() {
			Matrix<T> x = Layer<T>::forward();
			value = ops::mse(x, y);
		}
		virtual void backward(Matrix<T> &delta) {
			Matrix<T> x = input->getValue();
			input->backward(x - y);
		}
	};
}

#endif // !_LAYER_H_