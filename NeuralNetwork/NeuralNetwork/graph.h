#pragma once

#include <list>

#include "tensor.h"

namespace AutoGrad {

	using namespace std;
	using namespace tensor;

	enum NodeType { VARIABLE, PLACEHOLDER, OPERATION };

	template<class T>
	class Node {
	private:
		Tensor<T> output;// 输入值
		list<Node*> consumers;// 消费者
	public:
		Node() { ; }
		void setOutput(Tensor<T> &output) {
			this->output;
		}
		virtual NodeType getNodeType() = 0;
	};

	template<class T>
	class Variable : Node<T> {
	private:
		Tensor<T> value, grad;
		bool require_grad;
	public:
		Variable(Shape &shape, bool require_grad) : Node<T>(), value(Tensor<T>(shape)), require_grad(require_grad) { ; }
		virtual NodeType getNodeType() { return VARIABLE; }
		Tensor<T> getValue() { return value; }
	};

	template<class T>
	class Placeholder : Node<T> {
	private:
		Shape shape;
	public:
		Placeholder(Shape &shape) : Node<T>(), shape(shape) { ; }
		virtual NodeType getNodeType() { return PLACEHOLDER; }
	};

	template<class T>
	class Operation : Node<T> {
	private:
		list<Node<T>*> input_nodes;
	public:
		Operation() { ; }
		Operation(list<Node<T>*> input_nodes) {
			this->input_nodes.push_back(input_nodes);
			this->consumers.clear();
			list<Node<T>*>::iterator iter;
			for (iter = input_nodes.begin(); iter != input_nodes.end(); iter++) {
				iter->consumers.push_back(this);
			}
		}
		list<Tensor<T>> getInputValues() {
			list<Tensor<T>> inputs;
			list<Node<T>*>::iterater parent;
			for (parent = input_nodes.begin(); parent != input_nodes.end(); parent++) {
				inputs.push_back(parent->output);
			};
		}
		virtual NodeType getNodeType() { return OPERATION; }
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) = 0;
	};
	//----------------------------------------MATH OPERATION-----------------------

	template<class T>
	class Add : Operation<T> {
	public:
		Add(Node<T>* x, Node<T> *y) {
			list<Node<T>*> input_nodes;
			input_nodes.push_back(x);
			input_nodes.push_back(y);
			Operation<T>(input_nodes);
		}
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> y = inputs[1];
			return x + y;
		}
	};

	template<class T>
	class MatMul : Operation<T> {
	public:
		MatMul(Node<T>* x, Node<T> *y) {
			list<Node<T>*> input_nodes;
			input_nodes.push_back(x);
			input_nodes.push_back(y);
			Operation<T>(input_nodes);
		}
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> y = inputs[1];
			return x.matmul(y);
		}
	};

	//----------------------------------------CONVOLUTION--------------------------
	template<class T>
	class Convolution : Operation<T> {
	private:
		int width;
		int n_filters;
		int padding;
		int stride;
		string activation;
		Variable<T> filter;
		Variable<T> bias;
	public:
		Convolution(Node<T> *x, int width, int padding,
			int stride, int n_filters, string activation)
			: width(width), padding(padding), stride(stride)
			, n_filters(n_filters), activation(activation) {
			list<Node<T>*> input_nodes;
			input_nodes.push_back(x);
			input_nodes.push_back(&filter);
			input_nodes.push_back(&bias);
			Operation<T>(input_nodes);
		}
	};

	template<class T>
	class Conv2D : Convolution<T> {
	public:
		Conv2D(Node<T> *x, int width, int padding,
			int stride, int n_filters, string activation)
			: Convolution(x, width, padding, stride, n_filters, activation) {
			;
		}
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> filter = inputs[1];
			Tensor<T> bias = inputs[2];
			return x.padding(padding).conv2d(filter, bias, stride);
		}
	};

	template<class T>
	class Conv3D : Convolution<T> {
	public:
		Conv3D(Node<T> *x, int width, int padding,
			int stride, int n_filters, string activation)
			: Convolution(x, width, padding, stride, n_filters, activation) {
			;
		}
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> filter = inputs[1];
			Tensor<T> bias = inputs[2];
			return x.conv3d(filter, bias, stride);
		}
	};

	//----------------------------------------POOLING OPEARION---------------------
	template<class T>
	class Pooling : public Operation<T> {
	protected:
		string activation;
		int width;
	public:
		Pooling(Node<T> *x, int width)
			: activation(activation), width(width) {
			list<Node<T>*> input_nodes;
			input_nodes.push_back(x);
			Operation<T>(input_nodes);
		}
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) = 0;
	};

	template<class T>
	class MaxPooling : public Pooling<T> {
	public:
		MaxPooling(Node<T> *x, int width)
			: Pooling<T>(x, width, activation) { ; }
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.max_pooling(width);// Pooling<T>::backward(delta.upsampling(input_value, width));
		}
	};

	template<class T>
	class MinPooling : public Pooling<T> {
	public:
		MinPooling(Node<T> *x, int width)
			: Pooling<T>(x, width, activation) { ; }
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.min_pooling(width);
		}
	};

	template<class T>
	class AvgPooling : public Pooling<T> {
	public:
		AvgPooling(Node<T> *x, int width)
			: Pooling<T>(x, width, activation) { ; }
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.avg_pooling(width);
		}
	};

	//----------------------------------------FLATTEN OPERATION---------------------
	template<class T>
	class Flatten : public Operation<T> {
	private:
		Shape before;
	public:
		Flatten(Node *x) {
			list<Node<T>*> input_nodes;
			input_nodes.push_back(x);
			Operation<T>(input_nodes);
		}
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			before = x.getShape();
			return x.flatten();
		}
	};

	template<class T>
	class Sigmoid : public Operation<T> {
	public:
		Sigmoid(Node<T> *x){}
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return  x.sigmoid();
		}
	};

	template<class T>
	class ReLU : public Operation<T> {
	public:
		ReLU(Node<T> *x) {
			list<Node<T>*> input_nodes;
			input_nodes.push_back(x);
			Operation<T>(input_nodes);
		}
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.relu();
		}
	};

	template<class T>
	class LeakyReLU : public Operation<T> {
	private:
		T max_value;
		T threshold;
		T negative_slope;
	public:
		LeakyReLU(Node<T> *x, T max_value, T threshold, T negative_slop)
		: max_value(max_value), threshold(threshold) , negative_slop(negative_slop) {
			list<Node<T>*> input_nodes;
			input_nodes.push_back(x);
			Operation<T>(input_nodes);
		}
		virtual Tensor<T> compute(list<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.relu(max_value, threshold, negative_slop);
		}
	};
	//----------------------------------------COMPUTATIONAL GRAPH-------------------
	template<class T>
	class Graph {
	private:
		list<Placeholder<T>*> placeholders;
		list<Variable<T>*> variables;
		list<Operation<T>*> operations;
	public:
		void collect(Node<T> *root) {
			if (root->getNodeType() == PLACEHOLDER) {
				placeholders.push_back(root);
			}
			if (root->getNodeType() == VARIABLE) {
				variables.push_back(root);
			}
			if (root->getNodeType() == OPERATION) {
				list<Node<T>*>::iterater iter;
				list<Node<T>*> input_nodes = ((Operation<T>*)root)->input_nodes;
				for (iter = input_nodes.begin(); iter != input_nodes.end(); iter++) {
					collect(iter);
				}
				operations.push_back(root);
			}
		}
	};

	template<class T>
	class Session {
	private:
		Graph graph;
	public:
		void Session(Operation<T> *operation) {
			graph.collect(operation);
		}
		void run(map<Placeholder<T>*, Tensor<T>> &feed_dict) {
			list<Node<T>*> operations = graph.operations;
			list<Node<T>*>::iterater iter;
			for (iter = operations.begin(); iter != operations.end(); iter++) {
				switch (iter->getNodeType()) {
				case PLACEHOLDER:
					iter->setOutput(feed_dict[*iter]);
					break;
				case VARIABLE:
					iter->setOutput(((Variable<T>*)iter)->getValue());
					break;
				case OPERATION:
					iter->setOutput(iter->compute(((Operation<T>*)iter)->getInputValues()));
					break;
				default:
					cout << "Wrong Node Type" << endl;
					break;
				}
			}
		}
	};

	//----------------------------------------FUNCTIONS-----------------------------
	template<class T>
	Node<T>* add(Node<T> *x, Node<T>* y) {
		return new Add(x, y);
	}

	template<class T>
	Node<T>* matmul(Node<T> *x, Node<T>* y) {
		return new MatMul(x, y);
	}

	template<class T>
	Node<T>* sigmoid(Node<T> x) {
		return new Sigmoid(x);
	}

	template<class T>
	Node<T>* relu(Node<T> x) {
		return new ReLU(x);
	}

	template<class T>
	Node<T>* leaky_relu(Node<T> x, T max_value, T threshold, T negative_slop) {
		return new LeakyReLU(x, max_value, threshold, negative_slop);
	}

	template<class T>
	Node<T>* conv2d(Node<T> *x, int width, int padding,	int stride, int n_filters, string activation = "sigmoid") {
		return new Conv2D(x, width, padding, stride, n_filters, activation);
	}

	template<class T>
	Node<T>* conv3d(Node<T> *x, int width, int padding, int stride, int n_filters, string activation = "sigmoid") {
		return new Conv3D(x, width, padding, stride, n_filters, activation);
	}

	template<class T>
	Node<T>* maxpooling(Node<T> *x, int width, string activation = "sigmoid") {
		return new MaxPooling(x, width, activation);
	}

	template<class T>
	Node<T>* minpooling(Node<T> *x, int width, string activation = "sigmoid") {
		return new MaxPooling(x, width, activation);
	}

	template<class T>
	Node<T>* avgpooling(Node<T> *x, int width, string activation = "sigmoid") {
		return new AvgPooling(x, width, activation);
	}

	template<class T>
	Node<T>* flatten(Node<T> x, string activation = "sigmoid") {
		return new Flatten(x);
	}

}