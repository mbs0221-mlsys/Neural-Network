#pragma once

#ifndef _MODEL_H
#define _MODEL_H

#define _EDG_ 1

#include "layer.h"
#include "optimizer.h"

namespace model {

	using namespace std;
	using namespace layers;
	using namespace optimizer;
	
	template<class T>
	class Model : public Layer<T> {
		std::string loss_func;
		std::vector<double> losses;
		Optimizer<T> optimizer;
	public:
		Model(Layer<T>* input, Layer<T>* output) : Layer<T>(input) {
			this->output = output;
		}
		void compile(std::string &loss, Optimizer<T> &optimizer) {
			this->loss_func = loss;
			this->optimizer = optimizer;
		}
		void train(Matrix<T> x, Matrix<T> y) {
			((Input<T> *)input)->feed(x);
			//optimizer.optimize(loss);

			output->forward();
			Matrix<T> loss = ops::mse(output->value, y);
			losses.push_back(loss.data[0][0]);

			grad = ops::sub(y, output->value);
			output->backward(grad);
		}
	};
	

	template<class T>
	class Network {
	private:
		int epochs = 1000;
		int batch_size = 10;
		double learning_rate = 0.01;
	public:
		std::map<std::string, Matrix<T>> weights;
		Network() {
			printf_s("Network()\n");
			weights["w0"] = Matrix<T>(2, 5);
			weights["w1"] = Matrix<T>(5, 8);
			weights["w2"] = Matrix<T>(8, 10);
			weights["b0"] = Matrix<T>(1, 5);
			weights["b1"] = Matrix<T>(1, 8);
			weights["b2"] = Matrix<T>(1, 10);
			randomize();
		}
		void load_weights(string path) {
			ifstream inf;
			inf.open(path, ios::in);
			if (inf.is_open()) {
				std::map<std::string, Matrix<T>>::iterator iter;
				for (iter = weights.begin(); iter != weights.end(); iter++) {
					iter->second.load(inf);
				}
				inf.close();
			}
		}
		void save_weights(string path) {			
			ofstream outf;
			outf.open(path, ios::out);
			if (outf.is_open()) {
				std::map<std::string, Matrix<T>>::iterator iter;
				for (iter = weights.begin(); iter != weights.end(); iter++) {
					iter->second.save(outf);
				}
				outf.close();
			}
		}
		void fit(Matrix<T> &x_train, Matrix<T> &y_train) {
			printf_s("fit()\n");
			y_train.savemat("y_train-one_hot.txt");
			double last = INFINITE;
			std::map<std::string, Matrix<T>> values, gradients;
			for (int i = 0; i < epochs; i++) {
				for (int start = 0; start < 100; start += batch_size) {
					values["X0"] = x_train.slice(start, start + batch_size, 0);
					values["Y0"] = y_train.slice(start, start + batch_size, 0);

					// 正向传播
					values["T0"] = values["X0"].matmul(weights["w0"]).add(weights["b0"]);
					values["O1"] = values["T0"].relu();

					values["T1"] = values["O1"].matmul(weights["w1"]).add(weights["b1"]);
					values["O2"] = values["T1"].relu();
					
					values["T2"] = values["O2"].matmul(weights["w2"]).add(weights["b2"]);
					values["O3"] = values["T2"].sigmoid();

					// 反向传播
					values["D2"] = ops::grad_sigmoid(values["O3"]) * (values["O3"] - values["Y0"]);
					values["D1"] = ops::matmul(ops::grad_relu(values["T1"]) * values["D2"], weights["w2"].Transpose());
					values["D0"] = ops::matmul(ops::grad_relu(values["T0"]) * values["D1"], weights["w1"].Transpose());

					//gradients.clear();
					gradients["w2"] = values["O2"].Transpose().matmul(values["D2"]);
					gradients["w1"] = values["O1"].Transpose().matmul(values["D1"]);
					gradients["w0"] = values["X0"].Transpose().matmul(values["D0"]);
					gradients["b2"] = values["D2"].reduce_sum(0);
					gradients["b1"] = values["D1"].reduce_sum(0);
					gradients["b0"] = values["D0"].reduce_sum(0);

					optimize(gradients, -learning_rate);
				}

				values["y_pred"] = predict(x_train);
				values["loss"] = ops::mse(values["y_pred"], y_train);
				printf("epochs:%5d\t loss:%.8f\n", i, values["loss"].data[0][0]);
				last = values["loss"].data[0][0];
			}
		}
		void randomize() {
			std::map<std::string, Matrix<T>>::iterator iter;
			for (iter = weights.begin(); iter != weights.end(); iter++) {
				std::string name = iter->first;
				weights[name].randomize();
			}
		}
		void optimize(std::map<std::string, Matrix<T>> &gradients, double learning_rate) {
			std::map<std::string, Matrix<T>>::iterator iter;
			for (iter = weights.begin(); iter != weights.end(); iter++) {
				std::string name = iter->first;
				weights[name] = weights[name] - learning_rate * gradients[name];
			}
		}
		Matrix<T> predict(Matrix<T> &x_test) {
			Matrix<T> net;
			net = x_test.matmul(weights["w0"]).relu().add(weights["b0"]);
			net = net.matmul(weights["w1"]).relu().add(weights["b1"]);
			net = net.matmul(weights["w2"]).sigmoid().add(weights["b2"]);
			return net;
		}
	};

	template<class T>
	void test_model() {
		// Data
		Matrix<T> x_train(10000, 2);
		Matrix<T> y_train(10000, 1);
		Matrix<T> x_test(40000, 2);
		// Model
		printf("Model:\n");
		int size[] = { NULL, 10 };
		Shape shape(size);
		Input<T> input(shape);

		Linear<T> linear1((Layer<T>*)&input, 6);
		Sigmoid<T> layer1((Layer<T>*)&linear1);

		Linear<T> linear2((Layer<T>*)&layer1, 9);
		Sigmoid<T> layer2((Layer<T>*)&linear2);

		Linear<T> linear3((Layer<T>*)&layer2, 15);
		ReLU<T> output((Layer<T>*)&linear3);

		Model<T> model((Layer<T>*)&input, (Layer<T>*)&output);
		printf("Model.compile:\n");
		model.compile(std::string("least_squares"), Optimizer<T>(0.9));
		printf("Model.train:\n");
		model.train(x_train, y_train);
	}

	template<class T>
	void test() {

		Matrix<T> x_train(100, 2);
		Matrix<T> y_train(100, 1);
		Matrix<T> x_test(100, 2);

		srand((unsigned int)time(NULL));

		x_train.print_shape();
		y_train.print_shape();
		x_test.print_shape();

		x_train.loadmat("x_train.txt");
		y_train.loadmat("y_train.txt");
		x_test.loadmat("x_test.txt");

		//x_train.savemat("x_train.txt");
		//y_train.savemat("y_train.txt");
		//x_test.savemat("x_test.txt");

		Network<T> net;
		net.load_weights("Text.txt");
		net.fit(x_train, y_train.one_hot());
		net.save_weights("Text.txt");

		Matrix<T> y_test = net.predict(x_test);
		Matrix<T> out = ops::softmax(y_test);
		out.savemat("softmax.txt");
	}
}

#endif // !_MODEL_H