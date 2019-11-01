#pragma once

#include <vector>
#include <stdarg.h>

#include "tensor.h"
#include "ops.h"

namespace AutoGrad {

	using namespace std;
	using namespace tensor;

	enum NodeType { VARIABLE, PLACEHOLDER, OPERATION };

	template<class T>
	class Node {
	private:
		Tensor<T> value;// every node has an output
	public:
		vector<Node*> consumers;// the consummer of this node
		Node() {  }
		void addInput() {}
		Tensor<T> getValue() { return value; }
		void setValue(Tensor<T> &value) { this->value = value; }
		virtual NodeType getNodeType() = 0;
		virtual Tensor<T> backward() {
			// collect updates from consumers
			vector<Tensor<T>> deltas;
			for (Node<T>* consumer : consumers) {
				Tensor<T> delta = consumer->backward();
				deltas.push_back(delta);
			}
			// accumulates all deltas
			Tensor<T> value = Tensor<T>::zeros(value.getShape());
			Tensor<T> delta = accumulate(deltas.begin(), deltas.end(), value);
			return delta;
		}
	};

	template<class T>
	class Variable : public Node<T> {
	private:
		string name;
		bool require_grad;
		Tensor<T> grad;
	public:
		Variable(Tensor<T> &tensor, bool require_grad = true) 
			:value(tensor), require_grad(require_grad) { ; }
		Variable(string name, Shape shape, bool require_grad=true)
			: name(name), require_grad(require_grad) {
			value = Tensor<T>::random(shape);
			grad = Tensor<T>::zeros(shape);
		}
		virtual NodeType getNodeType() { return VARIABLE; }
		void update(Tensor<T> delta){
			this->grad = delta;
		}
		virtual Tensor<T> backward() {
			grad = Node<T>::backward();
		}
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
	protected:
		void add_weight(string name, Shape &shape, bool trainable = true) {
			input_nodes.push_back(new Variable<T>(name, shape, trainable));
		}
	public:
		vector<Node<T>*> input_nodes; // only operation has inputs
		Operation(initializer_list<Node<T>*> inputs) {
			// add input nodes
			for (auto nodes : inputs) {
				input_nodes.push_back(nodes);
			}
			// consumer
			vector<Node<T>*>::iterator iter;
			for (iter = input_nodes.begin(); iter != input_nodes.end(); iter++) {
				(*iter)->consumers.push_back(this);
			}
		}
		virtual void build(Shape &input_shape) { ; }
		Tensor<T> getInput(int i) {
			return input_nodes[i]->getOutput();
		}
		vector<Tensor<T>> getInputs() {
			vector<Tensor<T>> inputs;
			vector<Node<T>*>::iterator parent;
			for (parent = input_nodes.begin(); parent != input_nodes.end(); parent++) {
				inputs.push_back((*parent)->getOutput());
			};
			return inputs;
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
		int width, n_filters, padding, stride;
	public:
		Convolution(Node<T> *x, int width, int padding, int stride, int n_filters)
			: Operation<T>({ x }), width(width), padding(padding), stride(stride), n_filters(n_filters) { ; }
		virtual void build(Shape &input_shape) {
			Shape kernel_shape(n_filters, input_shape[1], width, width, input_shape[4]);
			Shape bias_shape(1, 1, 1, 1, n_filters);
			add_weight("kernel", kernel_shape);
			add_weight("bias", bias_shape);
		}
	};

	template<class T>
	class Conv2D : public Convolution<T> {
	public:
		Conv2D(Node<T> *x, int width, int padding, int stride, int n_filters)
			: Convolution<T>(x, width, padding, stride, n_filters) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			return inputs[0].padding(padding).conv2d(inputs[1], inputs[2], stride);
		}
		virtual Tensor<T> backward() {
			Tensor<T> grad = Node<T>::backward();
			Tensor<T> x = getInput(0);
			Tensor<T> filter = getInput(1);
			Tensor<T> bias = getInput(2);
			// pass the delta to the filter and bias
			((Variable<T>*)input_nodes[1])->update(x.conv2d(grad, stride));
			((Variable<T>*)input_nodes[2])->update(grad.reduce_sum(0).reduce_sum(1).reduce_sum(2).reduce_sum(3));
			return grad.padding(width).conv2d(filter.rotate180(), stride);
		}
	};

	template<class T>
	class Conv3D : public Convolution<T> {
	public:
		Conv3D(Node<T> *x, int width, int padding, int stride, int n_filters)
			: Convolution<T>(x, width, padding, stride, n_filters) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			return inputs[0].padding(padding).conv2d(inputs[1], inputs[2], stride);
		}
		virtual Tensor<T> backward() {
			Tensor<T> grad = Node<T>::backward();
			Tensor<T> x = getInput(0);
			Tensor<T> filter = getInput(1);
			Tensor<T> bias = getInput(2);
			// pass the delta to the flter and bias
			((Variable<T>*)input_nodes[1])->update(grad.reduce_sum(0).reduce_sum(1).reduce_sum(2).reduce_sum(3));
			((Variable<T>*)input_nodes[2])->update(x.conv2d(grad, stride));
			return grad.padding(width).conv2d(filter.rotate180(), stride);
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
			return inputs[0].max_pooling(width);
		}
		virtual Tensor<T> backward(){
			Tensor<T> delta = Node<T>::backward();
			Tensor<T> input = getInput(0);
			return delta.upsampling(input, width);
		}
	};

	template<class T>
	class MinPooling : public Pooling<T> {
	public:
		MinPooling(Node<T> *x, int width) : Pooling<T>(x, width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			return inputs[0].min_pooling(width);
		}
		virtual Tensor<T> backward(){
			Tensor<T> delta = Node<T>::backward();
			Tensor<T> input = getInput(0);
			return delta.upsampling(input, width); 
		}
	};

	template<class T>
	class AvgPooling : public Pooling<T> {
	public:
		AvgPooling(Node<T> *x, int width) : Pooling<T>(x, width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			return inputs[0].avg_pooling(width);
		}
		virtual Tensor<T> backward(){
			Tensor<T> delta = Node<T>::backward();
			return delta.avg_upsampling(width);	
		}
	};
	//----------------------------------------RNN CELL------------------------------
	template<class T>
	class RNNCell : public Operation<T> {
	private:
	public:
		RNNCell(Node<T> *x, Node<T> *state) : Operation<T>({ x, state }) { ; }
		virtual void build(Shape &input_shape) {
			// needing debug
			Shape U_shape(1, 1, 1, input_shape[4], input_shape[4]);
			Shape b_shape(1, 1, 1, 1, input_shape[4]);
			add_weight("U", U_shape); add_weight("W", U_shape);
			add_weight("b", b_shape); add_weight("V", U_shape);
			add_weight("c", b_shape);
		}
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			// needing debug
			Tensor<T> x, s0, U, W, b, V, c, ot;
			x = inputs[0]; s0 = inputs[1];
			U = inputs[2]; W = inputs[3]; b = inputs[4];
			V = inputs[5]; c = inputs[6];
			s1 = ops::sigmoid(x.matmul(U) + W.matmul(s0) + b);
			ot = V.matmul(s1) + c;
			return y_t = ot.softmax();
		}
	};

	//----------------------------------------FLATTEN OPERATION---------------------

	template<class T>
	class Reshape : public Operation<T> {
	private:
		Shape input_shape, output_shape;
	public:
		Reshape(Node<T> *x, Shape &output_shape)
			: Operation<T>({ x }), output_shape(output_shape) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> input = inputs[0];
			input_shape = input.getShape();
			return input.reshape(output_shape);
		}
		virtual Tensor<T> backward(){
			Tensor<T> delta = Node<T>::backward();
			return delta.reshape(input_shape);
		}
	};

	template<class T>
	class Flatten : public Operation<T> {
	private:
		Shape before;
	public:
		Flatten(Node<T> *x) : Operation<T>({ x }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			before = x.getShape();
			return x.flatten();
		}
		virtual Tensor<T> backward(){
			Tensor<T> delta = Node<T>::backward();
			return delta.reshape(before);
		}
	};

	template<class T>
	class FullyConnected : public Operation<T> {
	private:
		int n_outputs;
	public:
		FullyConnected(Node<T> *x, int n_outputs)
			: Operation<T>({ x }), n_outputs(n_outputs) { ; }
		virtual void build(Shape &input_shape) {
			Shape weight_shape(1, 1, 1, input_shape[4], n_outputs);
			Shape bias_shape(1, 1, 1, 1, n_outputs);
			add_weight("weight", weight_shape, true);
			add_weight("bias", bias_shape, true);
		}
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> w = inputs[1];
			Tensor<T> b = inputs[2];
			return  x.matmul(w).add(b);
		}
		virtual Tensor<T> backward(){
			Tensor<T> delta = Node<T>::backward();
			// calculate the delta of the weight and bias
			Tensor<T> x = getInput(0);
			Tensor<T> w = getInput(1);
			Tensor<T> b = getInput(2);
			// update weight delta
			((Variable<T>*)input_nodes[1])->update(x.Transpose().matmul(delta));
			((Variable<T>*)input_nodes[2])->update(delta.reduce_sum(0).reduce_sum(1).reduce_sum(2).reduce_sum(3));
			return delta.matmul(w.Transpose());
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
		virtual Tensor<T> backward() {
			Tensor<T> delta = Node<T>::backward();
			// calculate the delta of the weight and bias
			Tensor<T> y = this->getOutput();
			Tensor<T> e = Tensor<T>::ones(y.getShape());
			return delta * (y * (e - y));
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
		virtual Tensor<T> backward() {
			Node<T>::backward();
			Tensor<T> input = getInput(0);
			return ops::grad_relu(input);
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
		virtual Tensor<T> backward() {
			Tensor<T> input = getInput(0);
			return ops::grad_relu(input, max_value, threshold, negative_slop);
		}
	};

	template<class T>
	class Softmax : public Operation<T> {
	public:
		Softmax(Node<T> *x) : Operation<T>({ x }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.softmax();
		}
	};

	template<class T>
	class MSE : public Operation<T> {
	public:
		MSE(Node<T> *output, Node<T> *target)
			: Operation<T>({ output, target }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> y_ = inputs[0];
			Tensor<T> y = inputs[1];
			Tensor<T> error = (y_ - y).pow(2);
			return error.reduce_mean();
		}
		virtual Tensor<T> backward() {
			Tensor<T> y_ = getInput(0);
			Tensor<T> y = getInput(1);
			return (y_ - y);
		}
	};

	template<class T>
	class CrossEntrpy : public Operation<T> {
	public:
		CrossEntrpy(Node<T> *output, Node<T> *target)
			: Operation<T>({ output, target }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> y_ = inputs[0];
			Tensor<T> y = inputs[1];
			Tensor<T> error = ((T)0.0f - (y*y_.log() + ((T)1.0f - y)*((T)1.0f - y_).log()));
			return error.reduce_mean();
		}
		virtual Tensor<T> backward() {
			Tensor<T> y_ = getInput(0);
			Tensor<T> y = getInput(1);
			return (y_ - y);
		}
	};
	//----------------------------------------COMPUTATIONAL GRAPH-------------------

	template<class T>
	class Graph {
	private:
		vector<Placeholder<T>*> placeholders;
		vector<Variable<T>*> variables;
		vector<Operation<T>*> operations;
	public:
		Graph() { ; }
		~Graph() { 
			placeholders.clear();
			variables.clear();
			operations.clear();
		}
		void init_variables() {
			vector<Variable<T>*>::iterator iter;
			for (iter = variables.begin(); iter != variables.end(); iter++) {
				Variable<T>* node = (Variable<T>*)(*iter);
				node->setValue(node->getValue());
			}
		}
		void feed_dict(map<Placeholder<T>*, Tensor<T>> &feed_dict) {
			vector<Placeholder<T>*>::iterator iter;
			for (iter = placeholders.begin(); iter != placeholders.end(); iter++) {
				Placeholder<T>* node = (Placeholder<T>*)(*iter);
				node->setValue(feed_dict[node]);
			}
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
			graph.init_variables();
			graph.feed_dict(feed_dict);
			vector<Operation<T>*> operations = graph.getOperations();
			vector<Operation<T>*>::iterator iter;
			for (iter = operations.begin(); iter != operations.end(); iter++) {
				Operation<T>* node = (*iter);
				node->setValue(node->compute(node->getInputs()));
			}
		}
		void backward() {
			vector<Node<T>*> operations = graph.getOperations();
			vector<Node<T>*> reverse_order = reverse(operations.begin(), operations.end());
		}
	};

	//----------------------------------------FUNCTIONS-----------------------------

	namespace layers {

		template<class T>
		Operation<T>* add(Node<T> *x, Node<T>* y) {
			return new Add<T>(x, y);
		}

		template<class T>
		Operation<T>* matmul(Node<T> *x, Node<T>* y) {
			return new MatMul<T>(x, y);
		}

		// activation function
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
		Operation<T>* activation_func(Node<T> *x, string &func,...) {
			//va_list args;
			//va_start(args, func);
			//if (func == "leaky_relu") {
			//	T max_value = va_arg(args, T);
			//	T threshold = va_arg(args, T);
			//	T negative_slop = va_arg(args, T);
			//	return leaky_relu(x, max_value, threshold, negative_slop);
			//}
			if (func == "relu")
				return relu(x);
			if (func == "sigmoid")
				return sigmoid(x);
		}

		// convolution
		template<class T>
		Operation<T>* conv2d(Node<T> *x, int width, int padding, 
			int stride, int n_filters, string activation="relu") {
			Operation<T>* res = new Conv2D<T>(x, width, padding, stride, n_filters);
			return activation_func(res, activation);
		}

		template<class T>
		Operation<T>* conv3d(Node<T> *x, int width, int padding, 
			int stride, int n_filters, string activation="relu") {
			Operation<T>* res = new Conv3D<T>(x, width, padding, stride, n_filters);
			return activation_func(res, activation);
		}

		// pooling
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

		// basic operation
		template<class T>
		Operation<T>* reshape(Node<T> *x, Shape &shape) {
			return new Reshape<T>(x, shape);
		}

		template<class T>
		Operation<T>* flatten(Node<T> *x) {
			return new Flatten<T>(x);
		}
	
		template<class T>
		Operation<T>* fully_connected(Node<T> *x, int n_outputs, string activation="sigmoid") {
			Operation<T>* res = new FullyConnected<T>(x, n_outputs);
			return activation_func(res, activation);
		}

		template<class T>
		Operation<T>* softmax(Node<T> *x) {
			return new Softmax<T>(x);
		}

		template<class T>
		Operation<T>* mse(Node<T> *x, Node<T> *y) {
			return new MSE<T>(x, y);
		}

		template<class T>
		Operation<T>* cross_entopy(Node<T> *y_, Node<T> *y) {
			return new CrossEntrpy<T>(y_, y);
		}

	}

	template<class T>
	void test() {

		using namespace layers;

		Shape input_shape(NULL, 1, 28, 28, 3);
		Shape output_shape(NULL, 1, 1, 1, 10);

		Placeholder<T> *x = new Placeholder<T>(input_shape);
		Placeholder<T> *y = new Placeholder<T>(output_shape);

		Operation<T> *net;
		// conv1
		net = conv2d(x, 3, 1, 1, 10);
		net = maxpooling(net, 3);
		// conv2
		net = conv2d(net, 3, 1, 1, 10);
		net = maxpooling(net, 3);
		// conv3
		net = conv2d(net, 3, 1, 1, 10);
		net = maxpooling(net, 3);
		// fc_layer
		net = flatten(net);
		net = fully_connected(net, 10);
		net = softmax(net);

		Operation<T> *loss = cross_entopy(net, y);
		Session<T> session(loss);

		//map<Placeholder<T>*, Tensor<T>> feed_dict;
		//feed_dict[x] = Tensor<T>(1, 1000, 28, 28, 3);
		//feed_dict[y] = Tensor<T>(1, 1, 1, 1000, 10);

		//session.run(feed_dict);
	}
}
