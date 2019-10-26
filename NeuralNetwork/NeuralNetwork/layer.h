#pragma once

#ifndef _LAYER_H_
#define _LAYER_H_

#include <map>
#include <string>

#include "ops.h"

namespace layers {

	using namespace std;
	using namespace tensor;
	using namespace ops;

	template<class T>
	class Layer {
	public:
		string name;
		Shape shape;
		Layer<T> *input;
		Layer() : input(nullptr) {  }
		Layer(Layer<T> *input) { setInput(input); }
		Shape getShape() { return shape; }
		virtual void setInput(Layer<T> *input) {
			this->input = input;
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			if (input != nullptr)
				return input->forward(data);// 让前面的输入节点向前传递data
			else
				return data;// 若是无输入节点，则返回
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			if (input != nullptr)
				return input->backward(delta); // 让前面的输入节点向前传递delta
			else
				return delta;// 若是无输入节点，则返回
		}
		virtual void update() { ; }
	};

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
		virtual Tensor<T> forward(Tensor<T> &data) {
			return data;// input节点不会有输入，所以直接返回data
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return delta;// input节点不会有输入，所以直接返回delta
		}
	};

	template<class T>
	class Conv3D :public Layer<T> {
	private:
		Tensor<T> filter;
		int stride;
	public:
		Conv3D(int width=3, int stride=1, int n_filters=1): Layer<T>(){
			Shape filter_shape(n_filters, width, width, 1);
			filter = Tensor<T>(filter_shape);
		}
		virtual void setInput(Layer<T> *input) {
			Layer<T>::setInput(input);
			Shape input_shape = input->getShape();
			Shape filter_shape = filter.getShape();
			filter_shape.set(3, input_shape[3]);// n_channels
			filter = Tensor<T>(filter_shape);
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			Tensor<T> x = Layer<T>::forward(data);
			return x.conv3d(filter, stride);
		}
	};

	enum POOLING { MAX_POOLING, MIN_POOLING, AVG_POOLING };

	template<class T>
	class Pooling : public Layer<T> {
	private:
		Tensor<T> input_value;
		POOLING type;
		int size;
	public:
		Pooling(int size, POOLING type=MAX_POOLING) : Layer<T>(), size(size), (type) { ; }
		virtual void setInput(Layer<T> *input) {
			Layer<T>::setInput(input);
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			// average pooling by default
			input_value = Layer<T>::forward(data);			
			if (type == POOLING::MAX_POOLING)
				return input_value.max_pooling(size);
			else if (type == POOLING::MIN_POOLING)
				return input_value.min_pooling(size);
			else if (type == POOLING::AVG_POOLING)
				return input_value.avg_pooling(size);
			else
				return input_value.avg_pooling(size);
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			// average upsampling by default
			if (type == POOLING::MAX_POOLING)
				return Layer<T>::backward(delta.max_upsampling(input_value, size));
			else if (type == POOLING::MIN_POOLING)
				return Layer<T>::backward(delta.min_upsampling(input_value, size));
			else if (type == POOLING::AVG_POOLING)
				return Layer<T>::backward(delta.avg_upsampling(input_value, size));
			else
				return Layer<T>::backward(delta.avg_upsampling(input_value, size));
		}
	};

	template<class T>
	class Flatten : public Layer<T> {
	private:
		Shape before;
	public:
		Flatten() : Layer<T>() { ; }
		virtual void setInput(Layer<T> *input) {
			Layer<T>::setInput(input);
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			Tensor<T> x = Layer<T>::forward(data);
			before = x.getShape();// privious shape
			value = x.flatten(); // merge last two dimensions
			return value;
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return Layer<T>::backward(delta.reshape(before)); // restore privious shape
		}
	};

	template<class T>
	class Linear : public Layer<T> {
	private:
		Tensor<T> x, w, b;
		Tensor<T> grad_w, grad_b;
	public:
		Linear(int n_output) : Layer<T>() {
			shape.setDims(n_output, 3);
		}
		virtual void setInput(Layer<T> *input) {
			Layer<T>::setInput(input);
			Shape input_shape = input->getShape();
			shape.set(NULL, NULL, input_shape[3], shape[3]);
			// output_shape
			w = Tensor<T>(1, 1, input_shape[3], shape[3]);// (1,1,row,col)
			b = Tensor<T>(1, 1, 1, shape[3]);// (1,1,1,col)
		}
		virtual Tensor<T> forward(Tensor<T> data) {
			x = Layer<T>::forward(data);
			value = x.matmul(w) + b;// TODO: matrix+vector
			return value;
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			grad_w = x.Transpose().matmul(delta);
			grad_b = delta.reduce_sum(0);
			return Layer<T>::backward(delta.matmul(w.Transpose()));
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
		Tensor<T> grad;
		string activation;
	public:
		FullConnected(int n_output, string activation) : Linear<T>(n_output), activation(activation) { ; }
		virtual void setInput(Layer<T> *input) {
			Linear<T>::setInput(input);
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			Tensor<T> x = Linear<T>::forward(data);
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
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return Linear<T>::backward(delta * grad);
		}
		virtual void update() {
			Linear<T>::update();
		}
	};

	template<class T>
	class Loss : Layer<T> {
	private:
		Tensor<T> y, m_delta;
	public:
		Loss(Layer<T> *input, Tensor<T> y) :Layer<T>(input), y(y) { ; }
		double getLossValue() {
			return value.data[0][0];
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			Tensor<T> x = Layer<T>::forward(data);
			value = ops::mse(x, y);
			m_delta = x - y;
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return Layer<T>::backward(m_delta);
		}
	};
}

#endif // !_LAYER_H_