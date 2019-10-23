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
	private:
		Layer<T> *output;
		std::string loss_func;
		std::vector<double> losses;
		Optimizer<T> optimizer;
		void __init_model_(vector<Layer<T>*> &layers) {
			int num = layers.size();
			for (int i = 1; i < num; i++) {
				Layer<T> *layer0 = layers[i - 1];
				Layer<T> *layer1 = layers[i];
				layer1->setInput(layer0);
			}
		}
	public:
		Model(vector<Layer<T>*> &layers) : Layer<T>(layers[0]) {
			int num = layers.size();
			setOutput(layers[num-1]);
			__init_model_(layers);
		}
		void compile(std::string &loss, Optimizer<T> &optimizer) {
			this->loss_func = loss;
			this->optimizer = optimizer;
		}
		void train(Matrix<T> &x, Matrix<T> &y) {
			feed(x);
			for (int i = 0; i < 100; i++) {

				Matrix<T> y_ = forward();
				
				Matrix<T> delta = y_ - y;
				Matrix<T> loss = ops::cross_entropy_loss(ops::softmax(output->value), y);
				printf("Loss: %.8f\n", loss.data[0][0]);
				
				output->backward(delta);

				for (Layer<T> *p = output; p != NULL; p = p->input) {
					if (p->require_grad) {
						p->update();
					}
				}
			}
			printf("training finished");
		}
		virtual void setOutput(Layer<T> *output) {
			this->output = output;
		}
		virtual void feed(Matrix<T> &x) {
			Layer<T>::feed(x);
		}
		virtual Matrix<T> forward() {
			return output->forward();
		}
		virtual void backward(Matrix<T> &delta) {
			output->backward(delta);
		}
	};

	namespace bp_network {

		template<class T>
		class Network {
		private:
			int epochs = 5000;
			int batch_size = 60;
			double learning_rate = 10000;
		public:
			std::map<std::string, Matrix<T>> weights;
			Network() {
				printf_s("Network()\n");
				weights["w0"] = Matrix<T>(13, 11);
				weights["w1"] = Matrix<T>(11, 7);
				weights["w2"] = Matrix<T>(7, 3);
				weights["b0"] = Matrix<T>(1, 11);
				weights["b1"] = Matrix<T>(1, 7);
				weights["b2"] = Matrix<T>(1, 3);
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
				Shape shape = y_train.shape();
				int num_samples = shape[0];
				double last = INFINITE;
				std::map<std::string, Matrix<T>> gradients;
				for (int i = 0; i < epochs; i++) {
					for (int start = 0; start < num_samples; start += batch_size) {
						int end = start + batch_size;
						end = min(end, num_samples);
						Matrix<T> X0 = x_train.slice(start, end, 0);
						Matrix<T> Y0 = y_train.slice(start, end, 0);

						// 正向传播
						Matrix<T> T0 = X0.matmul(weights["w0"]).add(weights["b0"]);
						Matrix<T> O1 = T0.sigmoid();

						Matrix<T> T1 = O1.matmul(weights["w1"]).add(weights["b1"]);
						Matrix<T> O2 = T1.sigmoid();

						Matrix<T> T2 = O2.matmul(weights["w2"]).add(weights["b2"]);
						Matrix<T> O3 = T2.sigmoid();
						Matrix<T> O4 = ops::softmax(O3);

						// 反向传播
						Matrix<T> D2 = ops::grad_sigmoid(O3) * (O3 - Y0);
						Matrix<T> D1 = ops::grad_sigmoid(O2) * D2.matmul(weights["w2"].Transpose());
						Matrix<T> D0 = ops::grad_sigmoid(O1) * D1.matmul(weights["w1"].Transpose());

						//gradients.clear();
						gradients["w2"] = O2.Transpose().matmul(D2);
						gradients["w1"] = O1.Transpose().matmul(D1);
						gradients["w0"] = X0.Transpose().matmul(D0);
						gradients["b2"] = D2.reduce_sum(0);
						gradients["b1"] = D1.reduce_sum(0);
						gradients["b0"] = D0.reduce_sum(0);

						optimize(gradients, learning_rate);
						gradients.clear();
					}

					Matrix<T> y_pred = predict(x_train);
					Matrix<T> loss = ops::cross_entropy_loss(y_pred, y_train);
					printf("epochs:%5d\t loss:%.8f\n", i, loss.data[0][0]);
					last = loss.data[0][0];
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
				net = x_test.matmul(weights["w0"]).add(weights["b0"]).sigmoid();
				net = net.matmul(weights["w1"]).add(weights["b1"]).sigmoid();
				net = net.matmul(weights["w2"]).add(weights["b2"]).sigmoid();
				net = ops::softmax(net);
				return net;
			}
		};

		template<class T>
		void test(Matrix<T> &x_train, Matrix<T> &y_train, Matrix<T> &x_test) {

			printf("bp_network::test()\n");

			Network<T> net;
			net.load_weights("Text.txt");
			net.fit(x_train, y_train.one_hot(3));
			net.save_weights("Text.txt");

			Matrix<T> y_test = net.predict(x_test);
			y_test.savemat("softmax.txt");
		}
	}

	namespace fc_network {

		template<class T>
		Model<T>* create_fc_network() {

			int size[] = { NULL, 13 };

			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));
			layers.push_back(new FullConnected<T>(11, "sigmoid"));
			layers.push_back(new FullConnected<T>(9, "sigmoid"));
			layers.push_back(new FullConnected<T>(7, "sigmoid"));
			layers.push_back(new FullConnected<T>(5, "sigmoid"));
			layers.push_back(new FullConnected<T>(3, "sigmoid"));
			
			return new Model<T>(layers);
		}

		template<class T>
		void test(Matrix<T> &x_train, Matrix<T> &y_train, Matrix<T> &x_test) {

			printf("fc_network::test()\n");

			Model<T> *fc_network = create_fc_network<T>();
			fc_network->compile(std::string("least_squares"), Optimizer<T>(0.9));
			fc_network->train(x_train, y_train.one_hot(3));
		}
	}

	namespace auto_encoder {

		template<class T>
		Model<T>* create_encoder() {

			int size[] = { NULL, 13 };

			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));
			layers.push_back(new FullConnected<T>(11, "sigmoid"));
			layers.push_back(new FullConnected<T>(9, "sigmoid"));
			layers.push_back(new FullConnected<T>(7, "sigmoid"));
			layers.push_back(new FullConnected<T>(5, "sigmoid"));
			layers.push_back(new FullConnected<T>(3, "sigmoid"));

			return new Model<T>(layers);
		}

		template<class T>
		Model<T>* create_decoder() {
			int size[] = { NULL, 3 };
			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));
			layers.push_back(new FullConnected<T>(5, "sigmoid"));
			layers.push_back(new FullConnected<T>(7, "sigmoid"));
			layers.push_back(new FullConnected<T>(9, "sigmoid"));
			layers.push_back(new FullConnected<T>(11, "sigmoid"));
			layers.push_back(new FullConnected<T>(13, "sigmoid"));
			return new Model<T>(layers);
		}

		template<class T>
		Model<T>* create_auto_encoder() {

			Model<T>* encoder = create_encoder<T>();
			Model<T>* decoder = create_decoder<T>();

			vector<Layer<T>*> layers;
			layers.push_back(encoder);
			layers.push_back(decoder);

			return new Model<T>(layers);
		}

		template<class T>
		void test(Matrix<T> &x_train, Matrix<T> &y_train, Matrix<T> &x_test) {

			printf("auto_encoder::test()\n");

			Model<T>* auto_encoder = create_auto_encoder<T>();
			auto_encoder->compile(std::string("least_squares"), Optimizer<T>(0.9));
			auto_encoder->train(x_train, x_train);
		}
	}

	namespace gan {

		template<class T>
		Model<T>* create_generator() {

			int size[] = { NULL, 3 };

			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));
			layers.push_back(new FullConnected<T>(5, "sigmoid"));
			layers.push_back(new FullConnected<T>(7, "sigmoid"));
			layers.push_back(new FullConnected<T>(9, "sigmoid"));
			layers.push_back(new FullConnected<T>(11, "sigmoid"));
			layers.push_back(new FullConnected<T>(13, "sigmoid"));

			return new Model<T>(layers);
		}

		template<class T>
		Model<T>* create_discriminator() {

			int size[] = { NULL, 13 };

			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));
			layers.push_back(new FullConnected<T>(11, "sigmoid"));
			layers.push_back(new FullConnected<T>(9, "sigmoid"));
			layers.push_back(new FullConnected<T>(7, "sigmoid"));
			layers.push_back(new FullConnected<T>(5, "sigmoid"));
			layers.push_back(new FullConnected<T>(1, "sigmoid"));

			return new Model<T>(layers);
		}

		template<class T>
		Model<T>* create_gan() {

			Model<T> *generator = create_generator<T>();
			Model<T> *discriminator = create_discriminator<T>();

			vector<Layer<T>*> layers;
			layers.push_back(generator);
			layers.push_back(discriminator);

			return new Model<T>(layers);
		}

		template<class T>
		void test(Matrix<T> &x_train, Matrix<T> &y_train, Matrix<T> &x_test) {

			printf("gan::test()\n");

			Model<T>* gan = create_gan<T>();
			gan->compile(std::string("least_squares"), Optimizer<T>(0.9));
			//gan->train(x_train, x_train);
		}
	}


	template<class T>
	void load_data(Matrix<T> &x_train, Matrix<T> &y_train, Matrix<T> &x_test) {


		x_train.loadmat("x_train.txt");
		y_train.loadmat("y_train.txt");
		x_test.loadmat("x_test.txt");

		x_train.print_shape();
		y_train.print_shape();
		x_test.print_shape();

		//x_train.savemat("x_train.txt");
		//y_train.savemat("y_train.txt");
		//x_test.savemat("x_test.txt");
	}

	template<class T>
	void test() {

		srand((unsigned int)time(NULL));

		Matrix<T> x_train, y_train, x_test;

		load_data<T>(x_train, y_train, x_test);
		// run test module
		printf("model::test()\n");
		//bp_network::test<T>(x_train, y_train, x_test);
		//fc_network::test<T>(x_train, y_train, x_test);
		auto_encoder::test<T>(x_train, y_train, x_test);
		//gan::test<T>(x_train, y_train, x_test);
	}
}

#endif // !_MODEL_H