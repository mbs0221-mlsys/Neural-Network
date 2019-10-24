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
		Layer<T> *input;
		Layer() {  }
		Layer(Layer<T> *input) { setInput(input); }
		Shape getShape() { return shape; }
		virtual Matrix<T> getValue() { return value; }
		virtual void setInput(Layer<T> *input) {
			this->input = input;
		}
		virtual Matrix<T> forward(Matrix<T> &data) {
			if (input != nullptr)
				return input->forward(data);// 让前面的输入节点向前传递data
			else
				return data;// 若是无输入节点，则返回
		}
		virtual Matrix<T> backward(Matrix<T> &delta) {
			if (input != nullptr)
				return input->backward(delta); // 让前面的输入节点向前传递delta
			else
				return delta;// 若是无输入节点，则返回
		}
		virtual void update() { ; }
	};

	// 输入层
	template<class T>
	class Input : public Layer<T> {
	public:
		Input() : Layer<T>(nullptr) { ; }
		Input(Shape &shape) : Layer<T>(nullptr) {
			this->shape = shape;
		}
		Input(int size[]) : Layer<T>(nullptr) {
			this->shape = Shape(size);
		}
		virtual Matrix<T> forward(Matrix<T> &data) {
			value = data;// 设置value为data，供后续节点使用
			return value;// input节点不会有输入，所以直接返回x
		}
		virtual Matrix<T> backward(Matrix<T> &delta) {
			return delta;// input节点不会有输入，所以直接返回delta
		}
	};

	// 线性变换层
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
		virtual Matrix<T> forward(Matrix<T> data) {
			Matrix<T> x = Layer<T>::forward(data);
			value = x.matmul(w) + b;
			return value;
		}
		virtual Matrix<T> backward(Matrix<T> &delta) {
			Matrix<T> in = input->getValue();
			grad_w = in.Transpose().matmul(delta);
			grad_b = delta.reduce_sum(0);
			return Layer<T>::backward(delta.matmul(w.Transpose()));
		}
		virtual void update() {
			Layer<T>::update();
			w = w - 0.01*grad_w;
			b = b - 0.01*grad_b;
		}
	};

	// 全连接层
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
		virtual Matrix<T> forward(Matrix<T> &data) {
			Matrix<T> x = Linear<T>::forward(data);
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

	// 损失层
	template<class T>
	class Loss : Layer<T> {
	private:
		Matrix<T> y;
	public:
		Loss(Layer<T> *input, Matrix<T> y) :Layer<T>(input), y(y) { ; }
		double getLossValue() { return value.data[0][0]; }
		virtual Matrix<T> forward(Matrix<T> &data) {
			Matrix<T> x = Layer<T>::forward(data);
			value = ops::mse(x, y);
		}
		virtual Matrix<T> backward(Matrix<T> &delta) {
			Matrix<T> x = input->getValue();
			return input->backward(x - y);
		}
	};
}

#endif // !_LAYER_H_