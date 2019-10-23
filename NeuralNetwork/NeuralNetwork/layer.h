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
		bool require_grad = false;
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
		}
		virtual Matrix<T> forward() { 
			if (input != nullptr)
				return input->forward(); 
		}
		virtual void backward(Matrix<T> &delta) { ; }
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
		virtual void feed(Matrix<T> &x) { 
			if (input == nullptr)
				value = x; // 前面没有其他输入
			else
				Layer<T>::feed(x); // 前面有其他输入
		}
		virtual Matrix<T> forward() {
			if (input != nullptr)
				return Layer<T>::forward(); // 前面有其他输入
			else			
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
			require_grad = true;
			shape.setDims(n_output, 1);
		}
		virtual void setInput(Layer<T> *input) {
			Shape input_shape = input->getShape();
			shape.setDims(input_shape[1], 0);
			w = Matrix<T>(shape);
			b = Matrix<T>(1, shape[0]);
			Layer<T>::setInput(input);
		}
		virtual Matrix<T> forward() {
			Matrix<T> x = Layer<T>::forward();
			value = x.matmul(w) + b;
			return value;
		}
		virtual void backward(Matrix<T> &delta) {
			Matrix<T> x = input->getValue();
			grad_w = x.Transpose().matmul(delta);
			grad_b = delta.reduce_sum(0);
			input->backward(delta.matmul(w.Transpose()));
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
		virtual void backward(Matrix<T> &delta) {
			Linear<T>::backward(delta * grad);
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
		virtual void backward(Matrix<T> &delta) {
			Matrix<T> x = input->getValue();
			input->backward(x - y);
		}
	};
}

#endif // !_LAYER_H_