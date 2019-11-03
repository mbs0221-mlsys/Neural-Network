#pragma once

#include <vector>
#include <stdarg.h>
#include <map>

#include "tensor.h"
#include "ops.h"

namespace AutoGrad {

	using namespace std;
	using namespace tensor;

	enum NodeType { VARIABLE, PLACEHOLDER, OPERATION };

	template<class T>
	class Node {
	protected:
		Shape m_Shape;
		Tensor<T> m_Value;// every node has an output
		vector<Node*> m_Consumers;// the consumers of current node
	public:
		void setShape(Shape &shape) { m_Shape = shape; }
		void setValue(Tensor<T> &value) { m_Value = value; }
		void addConsumer(Node<T> *consumer) { m_Consumers.push_back(consumer); }
		Shape getShape() { return m_Shape; }
		Tensor<T> getValue() { return m_Value; }
		vector<Node*> getConsumers() { return m_Consumers; }
		virtual NodeType getNodeType() = 0;
	};

	template<class T>
	class Variable : public Node<T> {
	private:
		string m_Name;
		bool m_RequireGrad;
	public:
		Variable(string name, Shape shape, bool require_grad=true)
			: m_Name(name), m_RequireGrad(require_grad) {
			m_Shape = shape;
		}
		virtual NodeType getNodeType() { return VARIABLE; }
		bool isRequireGrad() { return m_RequireGrad; }
		void initialize() {
			m_Value = Tensor<T>::random(m_Shape);
		}
	};

	template<class T>
	class Placeholder : public Node<T> {
	public:
		Placeholder(Shape &shape) { 
			m_Shape = shape;
		}
		virtual NodeType getNodeType() { return PLACEHOLDER; }
	};

	template<class T>
	class Operation : public Node<T> {
	protected:
		vector<Node<T>*> m_InputNodes; // only operation has inputs
		void addWeight(string name, Shape &shape, bool trainable = true) {
			Variable<T> *variable = new Variable<T>(name, shape, trainable);
			m_InputNodes.push_back(variable);
			variable->addConsumer(this);
		}
	public:
		Operation(initializer_list<Node<T>*> inputNodes) {
			// build bi-directional linkage
			m_InputNodes.clear();
			for (Node<T>* InputNode : inputNodes) {
				m_InputNodes.push_back(InputNode);
				InputNode->addConsumer(this);
			}
		}
		Tensor<T> getInput(int i) {
			return m_InputNodes[i]->getValue();
		}
		vector<Tensor<T>> getInputs() {
			vector<Tensor<T>> inputs;
			int n = m_InputNodes.size();
			for (int i = 0; i < n;i++) {
				Node<T>* InputNode = m_InputNodes[i];
				inputs.push_back(InputNode->getValue());
			};
			return inputs;
		}
		vector<Node<T>*> getInputNodes() { return m_InputNodes; }
		virtual NodeType getNodeType() { return OPERATION; }
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) = 0; // forward output
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) = 0; // back propagation
		virtual void build(Shape &shape) { ; }
	};


	//----------------------------------------MATH OPERATION-----------------------

	template<class T>
	class Add : public Operation<T> {
	public:
		Add(Node<T>* x, Node<T> *y) :Operation<T>({ x, y }) { ; }
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
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
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> y = inputs[1];
			return x.matmul(y);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			vector<Tensor<T>> inputs = getInputs();
			if (V == m_InputNodes[0])
				return inputs[1].matmul(D);
			if (V == m_InputNodes[1])
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
			: Operation<T>({ x }), width(width), padding(padding), stride(stride), n_filters(n_filters) {
		}
		virtual void build(Shape &shape) {
			// build weights
			Shape filter_shape(n_filters, shape[1], width, width, shape[4]);
			Shape bias_shape(1, 1, 1, 1, n_filters);
			addWeight("filter", filter_shape);
			addWeight("bias", bias_shape);
		}
	};

	template<class T>
	class Conv2D : public Convolution<T> {
	public:
		Conv2D(Node<T> *x, int width, int padding, int stride, int n_filters)
			: Convolution<T>(x, width, padding, stride, n_filters) {
			build(x->getShape());
		}
		virtual void build(Shape &shape) {
			Convolution<T>::build(shape);
			// calculate output shape
			int n_samples = shape[0];
			int n_frames = shape[1];
			int n_width = (shape[2] - width) / stride + 1;
			int n_height = (shape[3] - width) / stride + 1;
			int n_channels = n_filters;
			m_Shape = Shape(n_samples, n_frames, n_width, n_height, n_channels);
		}
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			Tensor<T> filter = inputs[1];
			Tensor<T> bias = inputs[2];
			return x.padding(padding).conv2d(filter, bias, stride);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			Tensor<T> x = getInput(0);
			Tensor<T> filter = getInput(1);
			Tensor<T> bias = getInput(2);
			// pass the delta to the flter and bias
			if (V == m_InputNodes[0])
				return D.padding(width).conv2d(filter.rotate180(), stride);
			if (V == m_InputNodes[1])
				return x.conv2d(D, stride);
			if (V == m_InputNodes[2])
				return D.reduce_sum(0).reduce_sum(1).reduce_sum(2).reduce_sum(3);
			return D;
		}
	};

	template<class T>
	class Conv3D : public Convolution<T> {
	public:
		Conv3D(Node<T> *x, int width, int padding, int stride, int n_filters)
			: Convolution<T>(x, width, padding, stride, n_filters) { 
			build(x->getShape());
		}
		virtual void build(Shape &shape) {
			Convolution<T>::build(shape);
			// calculate output shape
			int n_samples = shape[0];
			int n_frames = (shape[1] - width) / stride + 1;
			int n_width = (shape[2] - width) / stride + 1;
			int n_height = (shape[3] - width) / stride + 1;
			int n_channels = n_filters;
			m_Shape = Shape(n_samples, n_frames, n_width, n_height, n_channels);
		}
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
			return inputs[0].padding(padding).conv2d(inputs[1], inputs[2], stride);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			Tensor<T> x = getInput(0);
			Tensor<T> filter = getInput(1);
			Tensor<T> bias = getInput(2);
			// pass the delta to the flter and bias
			if (V == m_InputNodes[0])
				return D.padding(width).conv2d(filter.rotate180(), stride);
			if (V == m_InputNodes[1])
				return x.conv2d(D, stride);
			if (V == m_InputNodes[2])
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
		Pooling(Node<T> *x, int width) : Operation<T>({ x }), width(width) { 
			build(x->getShape());
		}
		virtual void build(Shape &shape) {
			// calculate output shape
			int n_samples = shape[0];
			int n_frames = shape[1];
			int n_width = shape[2] / width;
			int n_height= shape[3] / width;
			int n_channels = shape[4];
			m_Shape = Shape(n_samples, n_frames, n_width, n_height, n_channels);
		}
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) = 0;
	};

	template<class T>
	class MaxPooling : public Pooling<T> {
	public:
		MaxPooling(Node<T> *x, int width) : Pooling<T>(x, width) { ; }
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
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
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
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
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
			return inputs[0].avg_pooling(width);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D.avg_upsampling(width);
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
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
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
	public:
		Reshape(Node<T> *x, Shape &shape)
			: Operation<T>({ x }) { 
			setShape(shape);
		}
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
			Tensor<T> input = inputs[0];
			return input.reshape(m_Shape);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D.reshape(V->getShape());
		}
	};

	template<class T>
	class Flatten : public Operation<T> {
	public:
		Flatten(Node<T> *x) : Operation<T>({ x }) { 
			build(x->getShape());
		}
		virtual void build(Shape &shape) {
			setShape(shape.flatten());
		}
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
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
			: Operation<T>({ x }), n_outputs(n_outputs) {
			build(x->getShape());
		}
		virtual void build(Shape &shape) {
			// build weights
			Shape weight_shape(1, 1, 1, shape[4], n_outputs);
			Shape bias_shape(1, 1, 1, 1, n_outputs);
			addWeight("weight", weight_shape, true);
			addWeight("bias", bias_shape, true);
			// calculate output shape
			int n_samples = shape[0];
			int n_frames = shape[1];
			int n_width = shape[2];
			int n_height = shape[4];
			int n_channels = n_outputs;
			m_Shape = Shape(n_samples, n_frames, n_width, n_height, n_channels);
		}
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
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
			if (V == m_InputNodes[0]) // x
				return D.matmul(w.Transpose());
			if (V == m_InputNodes[1]) // w
				return x.Transpose().matmul(D);
			if (V == m_InputNodes[2]) // b
				return D.reduce_sum(0).reduce_sum(1).reduce_sum(2).reduce_sum(3);
			return D;
		}
	};


	//----------------------------------------ACTIVATION OPERATION---------------------
	template<class T>
	class Activation : public Operation<T> {
	public:
		Activation(Node<T> *x) : Operation<T>({ x }) { 
			m_Shape = x->getShape();
		}
	};

	template<class T>
	class Sigmoid : public Activation<T> {
	public:
		Sigmoid(Node<T> *x) : Activation<T>(x) { ; }
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
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
	class ReLU : public Activation<T> {
	public:
		ReLU(Node<T> *x) : Activation<T>(x) { ; }
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.relu();
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D * ops::grad_relu(V->getValue());
		}
	};

	template<class T>
	class LeakyReLU : public Activation<T> {
	private:
		T max_value;
		T threshold;
		T negative_slope;
	public:
		LeakyReLU(Node<T> *x, T max_value, T threshold, T negative_slop)
			: Activation<T>(x), max_value(max_value), threshold(threshold), negative_slop(negative_slop) { ; }
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.relu(max_value, threshold, negative_slop);
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return D * ops::grad_relu(V->getValue(), max_value, threshold, negative_slop);
		}
	};

	template<class T>
	class Softmax : public Activation<T> {
	public:
		Softmax(Node<T> *x) : Activation<T>(x) { ; }
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
			Tensor<T> x = inputs[0];
			return x.softmax();
		}
		virtual Tensor<T> bprop(Node<T>* V, Tensor<T> &D) {
			return Tensor<T>::zeros(D.getShape());
		}
	};

	//
	template<class T>
	class Loss : public Operation<T> {
	public:
		Loss(Node<T> *output, Node<T> *target) 
			: Operation<T>({ output, target }) {
			m_Shape = output->getShape();
		}
	};

	template<class T>
	class MSE : public Loss<T> {
	public:
		MSE(Node<T> *output, Node<T> *target)
			: Loss<T>(output, target) { ; }
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
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
	class CrossEntrpy : public Loss<T> {
	public:
		CrossEntrpy(Node<T> *output, Node<T> *target) 
			: Loss<T>(output, target) { ; }
		virtual Tensor<T> forward(vector<Tensor<T>> &inputs) {
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
			for (Node<T>* consumer : V->getConsumers()) {
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
		// basic function
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
		void feed_dict(map<Placeholder<T>*, Tensor<T>*> &feed_dict) {
			// initialize placaeholders
			for (Placeholder<T>* placeholder : placeholders) {
				Tensor<T> tensor = *feed_dict[placeholder];
				placeholder->setValue(tensor);
			}
		}
		void initialize_all_variables() {
			// initialize variables
			for (Variable<T>* variable : variables) {
				variable->initialize();
			}
		}
		void run() {
			// forward evaluation
			for (Operation<T>* operation : operations) {
				vector<Tensor<T>> inputs = operation->getInputs();
				Tensor<T> value = operation->forward(inputs);
				operation->setValue(value);
			}
		}
		void build_grad() {
			// initialize the gradient of loss
			int N = operations.size();
			Operation<T> *loss = operations[N - 1];
			Shape shape = loss->getShape();
			grad_table[loss] = Tensor<T>::eye(shape[4]);
			// update the gradients of other variables
			for (Variable<T>* variable : variables) {
				if (variable->isRequireGrad()) {
					build_grad(grad_table, variable);
				}
			}
		}
		// getter
		vector<Placeholder<T>*> get_placeholders() { return placeholders; }
		vector<Variable<T>*> get_variables() { return variables; }
		vector<Operation<T>*> get_operations() { return operations; }
	};

	//----------------------------------------OPTIMIZER----------------------------
	template<class T>
	class Session {
	private:
		Graph<T> graph;
	public:
		Session(Node<T> *operation) {
			graph.collect(operation);
		}
		void run(map<Placeholder<T>*, Tensor<T>*> &feed_dict) {
			graph.initialize_all_variables();
			graph.feed_dict(feed_dict);
			graph.run(); 
			graph.build_grad();
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

		Operation<T> *loss = cross_entopy(net, y);
		Session<T> session(loss);

		map<Placeholder<T>*, Tensor<T>*> feed_dict;
		feed_dict[x] = new Tensor<T>(1000, 1, 28, 28, 3);
		feed_dict[y] = new Tensor<T>(1, 1, 1, 1000, 10);

		session.run(feed_dict);
	}
}
