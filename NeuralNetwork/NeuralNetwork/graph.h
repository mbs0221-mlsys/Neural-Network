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
		Shape getShape() { return value.getShape(); }
		void setShape(Shape &shape) { value = Tensor<T>::zeros(shape); }
		Tensor<T> getValue() { return value; }
		void setValue(Tensor<T> &value) { this->value = value; }
		void add_consumer(Node<T> *consumer) { consumers.push_back(consumer); }
		vector<Node*> get_consumers() { return consumers; }
		virtual NodeType getNodeType() = 0;
	};

	template<class T>
	class Variable : public Node<T> {
	private:
		string name;
		Shape shape;
		bool require_grad;
	public:
		Variable(string name, Shape &shape, bool require_grad=true)
			: name(name), shape(shape), require_grad(require_grad) {
		}
		virtual NodeType getNodeType() { return VARIABLE; }
		bool is_require_grad() { return require_grad; }
		void initialize() {
			setValue(Tensor<T>::random(shape));
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
		vector<Node<T>*> input_nodes; // only operation has inputs
		void add_weight(string name, Shape &shape, bool trainable = true) {
			Variable<T> *variable = new Variable<T>(name, shape, trainable);
			input_nodes.push_back(variable);
			variable->add_consumer(this);
		}
	public:
		Operation(initializer_list<Node<T>*> inputs) {
			// build bi-directional linkage
			for (Node<T>* input : inputs) {
				input_nodes.push_back(input);
				input->add_consumer(this);
			}
		}
		Tensor<T> getInput(int i) {
			return input_nodes[i]->getValue();
		}
		vector<Tensor<T>> getInputs() {
			vector<Tensor<T>> inputs;
			vector<Node<T>*>::iterator parent;
			for (parent = input_nodes.begin(); parent != input_nodes.end(); parent++) {
				inputs.push_back((*parent)->getValue());
			};
			return inputs;
		}
		vector<Node<T>*> getInputNodes() { return input_nodes; }
		virtual NodeType getNodeType() { return OPERATION; }
		virtual void build(Shape &input_shape) { ; }// initialize weights
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) = 0; // compute output
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) = 0; // back propagation
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D;
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			vector<Tensor<T>> inputs = getInputs();
			if (V == input_nodes[0])
				return inputs[1].matmul(D);
			if (V == input_nodes[1])
				return inputs[0].matmul(D);
			return D;
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			Tensor<T> x = getInput(0);
			Tensor<T> filter = getInput(1);
			Tensor<T> bias = getInput(2);
			// pass the delta to the flter and bias
			if (V == input_nodes[0])
				return D.padding(width).conv2d(filter.rotate180(), stride);
			if (V == input_nodes[1])
				return x.conv2d(D, stride);
			if (V == input_nodes[2])
				return D.reduce_sum(0).reduce_sum(1).reduce_sum(2).reduce_sum(3);
			return D;
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			Tensor<T> x = getInput(0);
			Tensor<T> filter = getInput(1);
			Tensor<T> bias = getInput(2);
			// pass the delta to the flter and bias
			if (V == input_nodes[0])
				return D.padding(width).conv2d(filter.rotate180(), stride);
			if (V == input_nodes[1])
				return x.conv2d(D, stride);
			if (V == input_nodes[2])
				return D.reduce_sum(0).reduce_sum(1).reduce_sum(2).reduce_sum(3);
			return D;
		}
	};


	//----------------------------------------POOLING OPEARION---------------------

	template<class T>
	class Pooling : public Operation<T> {
	protected:
		int width;
	public:
		Pooling(Node<T> *x, int width) : Operation<T>({ x }), width(width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) = 0;
	};

	template<class T>
	class MaxPooling : public Pooling<T> {
	public:
		MaxPooling(Node<T> *x, int width) : Pooling<T>(x, width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			return inputs[0].max_pooling(width);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D.upsampling(V->getValue(), width);
		}
	};

	template<class T>
	class MinPooling : public Pooling<T> {
	public:
		MinPooling(Node<T> *x, int width) : Pooling<T>(x, width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			return inputs[0].min_pooling(width);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D.upsampling(V->getValue(), width);
		}
	};

	template<class T>
	class AvgPooling : public Pooling<T> {
	public:
		AvgPooling(Node<T> *x, int width) : Pooling<T>(x, width) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			return inputs[0].avg_pooling(width);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D.avg_upsampling(V->getValue(), width);
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
		Shape output_shape;
	public:
		Reshape(Node<T> *x, Shape &output_shape)
			: Operation<T>({ x }), output_shape(output_shape) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> input = inputs[0];
			return input.reshape(output_shape);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D.reshape(V->getShape());
		}
	};

	template<class T>
	class Flatten : public Operation<T> {
	public:
		Flatten(Node<T> *x) : Operation<T>({ x }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			return inputs[0].flatten();
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D.reshape(V->getShape());
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			// calculate the delta of the weight and bias
			Tensor<T> x = getInput(0);
			Tensor<T> w = getInput(1);
			// update weight delta
			if (V == input_nodes[0]) // x
				return D.matmul(w.Transpose());
			if (V == input_nodes[1]) // w
				return x.Transpose().matmul(D);
			if (V == input_nodes[2]) // b
				return D.reduce_sum(0).reduce_sum(1).reduce_sum(2).reduce_sum(3);
			return D;
		}
	};


	//----------------------------------------ACTIVATION OPERATION---------------------
	template<class T>
	class Sigmoid : public Operation<T> {
	public:
		Sigmoid(Node<T> *x) : Operation<T>({ x }) { ; }
		virtual Tensor<T> compute(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return  x.sigmoid();
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			Tensor<T> y = this->getValue();
			Tensor<T> e = Tensor<T>::ones(y.getShape());
			return D * (y * (e - y));
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D * ops::grad_relu(V->getValue());
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D * ops::grad_relu(V->getValue(), max_value, threshold, negative_slop);
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return Tensor<T>::zeros(D.getShape());
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			Tensor<T> y_ = V->getValue();
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
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			Tensor<T> y_ = V->getValue();
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
		map<Node<T>*, Tensor<T>> grad_table;
	protected:
		Tensor<T> build_grad(map<Node<T>*, Tensor<T>> &grad_table, Node<T> *V) {

			if (grad_table.find(V) != grad_table.end()) {
				return grad_table[V];
			}

			// calculate the gradients from all consumers of V

			vector<Tensor<T>> gradients;
			for (Node<T>* consumer : V->get_consumers()) {
				Tensor<T> D = build_grad(grad_table, consumer);
				Tensor<T> gradient = ((Operation<T>*)consumer)->bprop(V, D);
				gradients.push_back(gradient);
			}

			Tensor<T> zero = Tensor<T>::zeros(gradients[0].getShape());
			Tensor<T> G = accumulate(gradients.begin(), gradients.end(), zero);

			grad_table[V] = G;// record the gradient of V

			return G;
		}
	public:
		Graph() { ; }
		~Graph() {
			placeholders.clear();
			variables.clear();
			operations.clear();
		}
		void initialize_all_variables() {
			for (Variable<T>* variable : variables) {
				variable->initialize();
			}
		}
		void feed_dict(map<Placeholder<T>*, Tensor<T>> &feed_dict) {
			for (Placeholder<T>* placeholder : placeholders) {
				placeholder->setValue(feed_dict[placeholder]);
			}
		}
		void run() {
			for (Operation<T>* operation : operations) {
				vector<Tensor<T>> inputs = operation->getInputs();
				Tensor<T> value = operation->compute(inputs);
				operation->setValue(value);
			}
		}
		void build_grad() {
			for (Variable<T>* variable : variables) {
				if (variable->is_require_grad()) {
					build_grad(grad_table, variable);
				}
			}
		}
		void collect(Node<T> *root) {
			if (root->getNodeType() == PLACEHOLDER) {
				placeholders.push_back((Placeholder<T>*)root);
			}
			if (root->getNodeType() == VARIABLE) {
				variables.push_back((Variable<T>*)root);
			}
			if (root->getNodeType() == OPERATION) {
				vector<Node<T>*> inputs = ((Operation<T>*)root)->getInputNodes();
				for (Node<T>* input : inputs) {
					collect(input);
				}
				operations.push_back((Operation<T>*)root);
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
			graph.initialize_all_variables();
			graph.feed_dict(feed_dict);
			graph.run();
			//graph.build_grad();
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
		Operation<T>* activation_func(Node<T> *x, string &func,...) {
			//va_list args;
			//va_start(args, func);
			//if (func == "leaky_relu") {
			//	T max_value = va_arg(args, T);
			//	T threshold = va_arg(args, T);
			//	T negative_slop = va_arg(args, T);
			//	return new LeakyReLU<T>(x, max_value, threshold, negative_slop);
			//}
			if (func == "relu")
				return new ReLU<T>(x);
			return new Sigmoid<T>(x);
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
		net = conv2d(x, 3, 0, 1, 32);
		net = maxpooling(net, 2);
		// conv2
		net = conv2d(net, 3, 0, 1, 64);
		net = maxpooling(net, 2);
		// conv3
		net = conv2d(net, 3, 0, 1, 32); 
		net = maxpooling(net, 3);
		// fc_layer
		net = flatten(net);
		net = fully_connected(net, 10);
		net = softmax(net);

		//Operation<T> *loss = cross_entopy(net, y);
		Session<T> session(net);

		map<Placeholder<T>*, Tensor<T>> feed_dict;
		feed_dict[x] = Tensor<T>(1, 1000, 28, 28, 3);
		feed_dict[y] = Tensor<T>(1, 1, 1, 1000, 10);

		session.run(feed_dict);
	}
}
