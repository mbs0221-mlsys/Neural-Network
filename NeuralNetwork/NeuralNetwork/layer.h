#pragma once

#ifndef _LAYER_H_
#define _LAYER_H_

#include <map>

#include "ops.h"

namespace layers {

	using namespace std;

	template<class T>
	class Layer {
	public:
		string name;
		Shape shape;
		Matrix<T> value;
		Layer<T>* input;
		Layer() {  }
		Layer(Layer<T> *input) { setInput(input); }
		Matrix<T> getValue() { return value; }
		Shape getShape() { return shape; }
		virtual void setInput(Layer<T> *input) {
			this->input = input;
		}
		virtual void feed(Matrix<T> &x) {
			if (input != nullptr) {
				input->feed(x);
			}
			value = x;
		}
		virtual Matrix<T> forward() { 
			if (input != nullptr) {
				return input->forward();
			}
		}
		virtual Matrix<T> backward(Matrix<T> &delta) {
			if (input != nullptr)
				return input->backward(delta);
			else
				return delta;
		}
		virtual void update() { ; }
	};

	template<class T>
	class Input : public Layer<T> {
	public:
		Input(Shape &shape) : Layer<T>(nullptr) {
			this->shape = shape;
		}
		Input(int size[]) : Layer<T>(nullptr) {
			this->shape = Shape(size);
		}
		virtual Matrix<T> forward() {
			return value; // 前面没有其他输
		}
	};

	template<class T>
	class Linear : public Layer<T> {
	private:
		Matrix<T> w, b;
		Matrix<T> grad_w, grad_b;
	public:
		Linear(int n_output) : Layer<T>(NULL) {
			shape.setDims(n_output, 1);
		}
		virtual void setInput(Layer<T> *input) {
			Layer<T>::setInput(input);
			int n_output = shape[1];
			Shape input_shape = input->getShape();
			int size[] = { input_shape[0], shape[1] };
			shape = Shape(size);
			w = Matrix<T>(input_shape[1], n_output);
			b = Matrix<T>(1, n_output);
		}
		virtual Matrix<T> forward() {
			Matrix<T> x = Layer<T>::forward();
			value = x.matmul(w) + b;
			return value;
		}
		virtual Matrix<T> backward(Matrix<T> &delta) {
			Matrix<T> x = input->getValue();
			grad_w = x.Transpose().matmul(delta);
			grad_b = delta.reduce_sum(0);
			Matrix<T> delta_b = delta.matmul(w.Transpose());
			return Layer<T>::backward(delta_b);
		}
		virtual void update() {
			Layer<T>::update();
			w = w - 0.01*grad_w;
			b = b - 0.01*grad_b;
		}
	};

	template<class T>
	class FullConnected : public Linear<T> {
	private:
		Matrix<T> grad;
		string activation;
	public:
		FullConnected(int n_output, string activation) : Linear<T>(n_output), activation(activation) { ; }
		virtual void setInput(Layer<T> *input) {
			Linear<T>::setInput(input);
		}
		virtual Matrix<T> forward() {
			Matrix<T> x = Linear<T>::forward();
			if (activation == "sigmoid") {
				value = x.sigmoid();
				grad = ops::grad_sigmoid(value);
			}
			if (activation == "relu") {
				value = x.relu();
				grad = ops::grad_relu(x);
			}
			//if (activation == "leaky_relu") {
			//	value = x.relu(x, max_value, threshold, negative_slope);
			//	grad = ops::grad_relu(x, max_value, threshold, negative_slop);
			//}
			return value;

		}
		virtual Matrix<T> backward(Matrix<T> &delta) {
			return Linear<T>::backward(delta * grad);
		}
		virtual void update() {
			Linear<T>::update();
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
		virtual Matrix<T> backward(Matrix<T> &delta) {
			Matrix<T> x = input->getValue();
			return input->backward(x - y);
		}
	};
}

#endif // !_LAYER_H_