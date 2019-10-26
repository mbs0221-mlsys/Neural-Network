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
	
	//----------------------------------------ABSTRACT LAYER DEFINATION-------

	template<class T>
	class Layer {
	protected:
		string name;
		Shape shape;
		Layer<T> *input;
	public:
		Layer() : input(nullptr) {  }
		Layer(Layer<T> *input) { setInput(input); }
		Shape getShape() { return shape; }
		virtual void setInput(Layer<T> *input) {
			this->input = input;
		}
		virtual Layer<T>* getInput() {
			return input;
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
	//----------------------------------------INPUT LAYER---------------------

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

	// activation layer
	template<class T>
	class Activation {
	private:
		Tensor<T> grad;
		string activation;
	public:
		Activation(string activation) : activation(activation) { ; }
		virtual Tensor<T> forward(Tensor<T> &data) {
			if (activation == "sigmoid") {
				Tensor<T> value = data.sigmoid();
				grad = ops::grad_sigmoid(value);
				return value;
			} else if (activation == "relu") {
				Tensor<T> value = data.relu();
				grad = ops::grad_relu(data);
				return value;
			} else if (activation == "leaky_relu") {
				value = x.relu(x, max_value, threshold, negative_slope);
				grad = ops::grad_relu(x, max_value, threshold, negative_slop);
			} else {
				throw(std::exception());
			}

		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return delta * grad;
		}
	};

	//----------------------------------------CONVOLUTION LAYER---------------

	template<class T>
	class Convolution : public Layer<T> {
	protected:
		Tensor<T> filter;
		int padding;
		int stride;
		string activation;
	public:
		Convolution(int width = 3, int padding=0, int stride = 1,
			int n_filters = 1, string activation = "sigmoid") 
			: Layer<T>(), padding(padding), stride(stride), activation(activation) {
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
			Tensor<T> m_data = Layer<T>::forward(data);
			return m_data.padding(padding);
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			Tensor<T> m_delta = delta.clipping(padding);
			return Layer<T>::backward(m_delta);
		}
	};

	template<class T>
	class Conv2D : public Convolution<T> {
	private:
		Tensor<T> input_value;
	public:
		Conv2D(int width = 3, int padding=0, int stride = 1,
			int n_filters = 1, string activation = "sigmoid")
			: Convolution<T>(width, padding, stride, n_filters, activation) {
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			input_value = Convolution<T>::forward(data);
			return input_value.conv2d(filter, stride);
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			Tensor<T> m_delta = input_value.conv2d(delta.rotate180(), stride);
			return Convolution<T>::backward(m_delta.rotate180());
		}
	};

	template<class T>
	class Conv3D : public Convolution<T> {
	private:
		Tensor<T> filter;
		int stride;
		string activation;
	public:
		Conv3D(int width = 3, int padding = 0, int stride = 1,
			int n_filters = 1, string activation = "sigmoid")
			: Convolution<T>(width, padding, stride, n_filters, activation) {
			Shape filter_shape(n_filters, width, width, 1);
			filter = Tensor<T>(filter_shape);
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			Tensor<T> x = Convolution<T>::forward(data);
			return x.conv3d(filter, stride);
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return Convolution<T>::backward(data);
		}
	};

	//----------------------------------------POOLING LAYER-------------------
	template<class T>
	class Pooling : public Layer<T> {
	protected:
		string activation;
		int width;
	public:
		Pooling(int size, string activation = "sigmoid")
			: Layer<T>(), activation(activation), width(width) { ; }
		virtual void setInput(Layer<T> *input) {
			Layer<T>::setInput(input);
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			return Layer<T>::forward(data);
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return Layer<T>::backward(delta);
		}
	};

	template<class T>
	class MaxPooling : public Pooling<T> {
	private:
		Tensor<T> input_value;
	public:
		MaxPooling(int width, string activation = "sigmoid")
			: Pooling<T>(width, activation) { ; }
		virtual Tensor<T> forward(Tensor<T> &data) {
			input_value = Pooling<T>::forward(data);
			return input_value.max_pooling(size);
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return Pooling<T>::backward(delta.max_upsampling(input_value, width));
		}
	};

	template<class T>
	class MinPooling : public Pooling<T> {
	private:
		Tensor<T> input_value;
	public:
		MinPooling(int width, string activation = "sigmoid")
			: Pooling<T>(width, activation) { ; }
		virtual Tensor<T> forward(Tensor<T> &data) {
			input_value = Pooling<T>::forward(data);
			return input_value.min_pooling(width);
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return Pooling<T>::backward(delta.min_upsampling(input_value, width));
		}
	};

	template<class T>
	class AvgPooling : public Pooling<T> {
	public:
		AvgPooling(int width, string activation = "sigmoid")
			: Pooling<T>(width, activation) { ; }
		virtual Tensor<T> forward(Tensor<T> &data) {
			Pooling<T>::forward(data);
			return input_value.avg_pooling(width);
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			return Pooling<T>::backward(delta.avg_upsampling(width));
		}
	};

	//----------------------------------------FLATTEN LAYER-------------------
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
	class FullConnected : public Layer<T> {
	private:
		Tensor<T> x, w, b;
		Tensor<T> grad_w, grad_b;
	public:
		FullConnected(int n_output) : Layer<T>() {
			shape.setDims(n_output, 3);
		}
		virtual void setInput(Layer<T> *input) {
			Layer<T>::setInput(input);
			Shape input_shape = input->getShape();
			shape.set(NULL, NULL, input_shape[3], shape[3]);
			w = Tensor<T>(1, 1, input_shape[3], shape[3]);
			b = Tensor<T>(1, 1, 1, shape[3]);
		}
		virtual Tensor<T> forward(Tensor<T> data) {
			x = Layer<T>::forward(data);
			return x.matmul(w) + b;// TODO: matrix+vector
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