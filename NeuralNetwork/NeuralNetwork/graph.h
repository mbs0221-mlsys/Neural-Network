#pragma once

#include <vector>

#include "tensor.h"

namespace AutoGrad {

	using namespace std;
	using namespace tensor;

	enum NodeType { VARIABLE, PLACEHOLDER, OPERATION };

	template<class T>
	class Node {
	public:
		Tensor<T> output;// 输入值
		vector<Node*> consumers;// 消费者
		Node() {  }
		void getOuput() {
			return this->output; 
		}
		void setOutput(Tensor<T> output) {
			this->output = output;
		}
		virtual NodeType getNodeType() = 0;
	};

	template<class T>
	class Variable : public Node<T> {
	private:
		string name;
		Tensor<T> value, grad;
		bool require_grad;
	public:
		Variable(Tensor<T> value, string name, bool require_grad=true) 
			: name(name), require_grad(require_grad) { ; }
		Variable(Shape shape, string name, bool require_grad=true)
			: name(name), require_grad(require_grad) { ; }
		virtual NodeType getNodeType() { return VARIABLE; }
		Tensor<T> getValue() { return value; }
	};

	template<class T>
	class Placeholder : public Node<T> {
	private:
		Shape shape;
	public:
		Placeholder(Shape &shape) : shape(shape) { ; }
		virtual NodeType getNodeType() { return PLACEHOLDER; }
	};

	template<class T>
	class Operation : public Node<T> {
	public:
		vector<Node<T>*> input_nodes;// only operation has inputs
		Operation(initializer_list<Node<T>*> inputs) {
			for (auto nodes : inputs) {
				input_nodes.push_back(nodes);
			}
			vector<Node<T>*>::iterator iter;
			for (iter = input_nodes.begin(); iter != input_nodes.end(); iter++) {
				(*iter)->consumers.push_back(this);
			}
		}
		virtual void build(Shape &input_shape) { ; }
		Variable<T>* addWeight(string name, Shape &shape, bool trainable=true) {
			Tensor<T> ones = Tensor<T>::ones(shape);
			input_nodes.push_back(new Variable<T>(ones, name, trainable));
		}
		vector<Tensor<T>> getInputValues() {
			vector<Tensor<T>> inputs;
			vector<Node<T>*>::iterator parent;
			for (parent = input_nodes.begin(); parent != input_nodes.end(); parent++) {
				inputs.push_back((*parent)->output);
			};
		}
		virtual NodeType getNodeType() { return OPERATION; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) { ; }
	};


	//----------------------------------------MATH OPERATION-----------------------

	template<class T>
	class Add : public Operation<T> {
	public:
		Add(Node<T>* x, Node<T> *y) :Operation<T>({ x, y }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> y = inputs[1];
			return x + y;
		}
	};

	template<class T>
	class MatMul : public Operation<T> {
	public:
		MatMul(Node<T>* x, Node<T> *y) : Operation<T>({ x, y }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> y = inputs[1];
			return x.matmul(y);
		}
	};


	//----------------------------------------CONVOLUTION--------------------------

	template<class T>
	class Convolution : public Operation<T> {
	protected:
		int width;
		int n_filters;
		int padding;
		int stride;
		Variable<T>* m_filter;
		Variable<T>* m_bias;
	public:
		Convolution(Node<T> *x, int width, int padding, int stride, int n_filters)
			: Operation<T>({ x }), width(width), padding(padding), stride(stride), n_filters(n_filters) { ; }
		virtual void build(Shape &input_shape) {
			Shape kernel_shape(n_filters, input_shape[1], width, width, input_shape[4]);
			Shape bias_shape(1, 1, 1, 1, n_filters);
			m_filter = addWeight("kernel", kernel_shape);
			m_bias = addWeight("bias", bias_shape);
		}
	};

	template<class T>
	class Conv2D : public Convolution<T> {
	public:
		Conv2D(Node<T> *x, int width, int padding, int stride, int n_filters)
			: Convolution<T>(x, width, padding, stride, n_filters) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> filter = m_filter->getValue();
			Tensor<T> bias = m_bias->getValue();
			return x.padding(padding).conv2d(filter, bias, stride);
		}
	};

	template<class T>
	class Conv3D : public Convolution<T> {
	public:
		Conv3D(Node<T> *x, int width, int padding, int stride, int n_filters)
			: Convolution<T>(x, width, padding, stride, n_filters) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> filter = m_filter->getValue();
			Tensor<T> bias = m_bias->getValue();
			return x.padding(padding).conv2d(filter, bias, stride);
		}
	};


	//----------------------------------------POOLING OPEARION---------------------

	template<class T>
	class Pooling : public Operation<T> {
	protected:
		int width;
	public:
		Pooling(Node<T> *x, int width) : Operation<T>({ x }), width(width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) { ; }
	};

	template<class T>
	class MaxPooling : public Pooling<T> {
	public:
		MaxPooling(Node<T> *x, int width) : Pooling<T>(x, width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.max_pooling(width);
		}
	};

	template<class T>
	class MinPooling : public Pooling<T> {
	public:
		MinPooling(Node<T> *x, int width) : Pooling<T>(x, width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.min_pooling(width);
		}
	};

	template<class T>
	class AvgPooling : public Pooling<T> {
	public:
		AvgPooling(Node<T> *x, int width) : Pooling<T>(x, width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.avg_pooling(width);
		}
	};


	//----------------------------------------FLATTEN OPERATION---------------------

	template<class T>
	class Reshape : public Operation<T> {
	private:
		Shape input_shape, output_shape;
	public:
		Reshape(Node *x) : Operation<T>({ x }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			input_shape = x.getShape();
			return x.reshape(output_shape);
		}
	};

	template<class T>
	class Flatten : public Operation<T> {
	private:
		Shape before;
	public:
		Flatten(Node *x) : Operation<T>({ x }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			before = x.getShape();
			return x.flatten();
		}
	};

	template<class T>
	class FullConnect : public Operation<T> {
	private:
		int n_outputs;
		Variable<T> *weight, *bias;
	public:
		FullConnect(Node *x, int n_outputs) : Operation<T>({ x }), n_outputs(n_outputs) { ; }
		virtual void build(Shape &input_shape) {
			Shape weight_shape(1, 1, 1, input_shape[4], n_outputs);
			Shape bias_shape(1, 1, 1, 1, n_outputs);
			weight = addWeight("weight", weight_shape, true);
			bias = addWeight("bias", bias_shape, true);
		}
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> w = weight->getValue();
			Tensor<T> b = bias->getValue();
			return  x.matmul(w).add(b);
		}
	};

	template<class T>
	class Sigmoid : public Operation<T> {
	public:
		Sigmoid(Node<T> *x) : Operation<T>({ x }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return  x.sigmoid();
		}
	};

	template<class T>
	class ReLU : public Operation<T> {
	public:
		ReLU(Node<T> *x) : Operation<T>({ x }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
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
			: Operation<T>({ x }), max_value(max_value), threshold(threshold), negative_slop(negative_slop) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.relu(max_value, threshold, negative_slop);
		}
	};


	//----------------------------------------COMPUTATIONAL GRAPH-------------------

	template<class T>
	class Graph {
	private:
		vector<Node<T>*> placeholders;
		vector<Node<T>*> variables;
		vector<Node<T>*> operations;
	public:
		Graph() { ; }
		~Graph() { 
			placeholders.clear();
			variables.clear();
			operations.clear();
		}
		void collect(Node<T> *root) {
			if (root->getNodeType() == PLACEHOLDER) {
				placeholders.push_back(root);
			}
			if (root->getNodeType() == VARIABLE) {
				variables.push_back(root);
			}
			if (root->getNodeType() == OPERATION) {
				vector<Node<T>*>::iterator iter;
				vector<Node<T>*> input_nodes = ((Operation<T>*)root)->input_nodes;
				for (iter = input_nodes.begin(); iter != input_nodes.end(); iter++) {
					collect(*iter);
				}
				operations.push_back(root);
			}
		}
		vector<Node<T>*> getOperations() { return operations; }
	};

	template<class T>
	class Session {
	private:
		Graph<T> graph;
	public:
		Session(Node<T> *operation) {
			graph.collect(operation);
		}
		void run(map<Placeholder<T>*, Tensor<T>> &feed_dict) {
			vector<Node<T>*> operations = graph.getOperations();
			vector<Node<T>*>::iterator iter;
			for (iter = operations.begin(); iter != operations.end(); iter++) {
				Node<T>* node = (*iter);
				switch (node->getNodeType()) {
				case PLACEHOLDER:
					node->setOutput(feed_dict[(Placeholder<T>*)node]);
					break;
				case VARIABLE:
					node->setOutput(((Variable<T>*)node)->getValue());
					break;
				case OPERATION:
					node->setOutput(((Operation<T>*)node)->compute(((Operation<T>*)node)->getInputValues()));
					break;
				default:
					cout << "Wrong Node Type" << endl;
					break;
				}
			}
		}
	};

	//----------------------------------------FUNCTIONS-----------------------------

	namespace ops {

		template<class T>
		Operation<T>* add(Node<T> *x, Node<T>* y) {
			return new Add<T>(x, y);
		}

		template<class T>
		Operation<T>* matmul(Node<T> *x, Node<T>* y) {
			return new MatMul<T>(x, y);
		}

		template<class T>
		Operation<T>* sigmoid(Node<T> *x) {
			return new Sigmoid<T>(x);
		}

		template<class T>
		Operation<T>* relu(Node<T> *x) {
			return new ReLU<T>(x);
		}

		template<class T>
		Operation<T>* leaky_relu(Node<T> *x, T max_value, T threshold, T negative_slop) {
			return new LeakyReLU<T>(x, max_value, threshold, negative_slop);
		}

		template<class T>
		Operation<T>* conv2d(Node<T> *x, int width, int padding, int stride, int n_filters) {
			return new Conv2D<T>(x, width, padding, stride, n_filters);
		}

		template<class T>
		Operation<T>* conv3d(Node<T> *x, int width, int padding, int stride, int n_filters) {
			return new Conv3D<T>(x, width, padding, stride, n_filters);
		}

		template<class T>
		Operation<T>* maxpooling(Node<T> *x, int width) {
			return new MaxPooling<T>(x, width);
		}

		template<class T>
		Operation<T>* minpooling(Node<T> *x, int width) {
			return new MaxPooling<T>(x, width);
		}

		template<class T>
		Operation<T>* avgpooling(Node<T> *x, int width) {
			return new AvgPooling<T>(x, width);
		}

		template<class T>
		Operation<T>* reshape(Node<T> *x, Shape &shape) {
			return new Reshape<T>(x, shape);
		}

		template<class T>
		Operation<T>* flatten(Node<T> *x) {
			return new Flatten<T>(x);
		}
	
		template<class T>
		Operation<T>* full_connect(Node<T> *x, int n_outputs) {
			return new FullConnect<T>(x, n_outputs);
		}

		template<class T>
		void test() {
			
			Shape input_shape(NULL, 1, 28, 28, 3);
			Shape output_shape(NULL, 1, 1, 1, 10);
			
			Placeholder<T> *x = new Placeholder<T>(input_shape);
			Placeholder<T> *y = new Placeholder<T>(output_shape);
			
			map<Placeholder<T>*, Tensor<T>> feed_dict;
			feed_dict[x] = Tensor<T>(1, 1000, 28, 28, 3);
			feed_dict[y] = Tensor<T>(1, 1, 1, 1000, 10);

			Operation<T> *net = conv2d(x, 3, 1, 1, 10);
			// conv1
			net = conv2d(net, 3, 1, 1, 10);
			net = maxpooling(net, 3);
			net = sigmoid(net);
			// conv2
			net = conv2d(net, 3, 1, 1, 10);
			net = maxpooling(net, 3);
			net = sigmoid(net);
			// conv3
			net = conv2d(net, 3, 1, 1, 10);
			net = maxpooling(net, 3);
			net = sigmoid(net);
			// fc_layer
			net = flatten(net);
			net = full_connect(net, 10);
			net = sigmoid(net);

			Session<T> session(net);
			session.run(feed_dict);
		}
	}
}