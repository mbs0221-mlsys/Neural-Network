#pragma once

#ifndef _MODEL_H
#define _MODEL_H

#define _EDG_ 1

#include <vector>

#include "layer.h"
#include "optimizer.h"
#include "image.h"

namespace model {

	using namespace std;
	using namespace tensor;
	using namespace layers;
	using namespace optimizer;
	using namespace image;
	
	template<class T>
	class Model : public Layer<T> {
	private:
		std::string loss_func;
		std::vector<double> losses;
		Optimizer<T> optimizer;
		Layer<T>* output;
		void __init_model_(vector<Layer<T>*> &layers) {
			int num = layers.size();
			// 双向链表
			for (int i = 1; i < num; i++) {
				layers[i]->setInput(layers[i - 1]);// 建立链接关系
			}
		}
	protected:
		virtual void get_variables(Map &variable) {
			for (Layer<T> *p = output; p != nullptr; p = p->input) {
				p->get_variables(variable);
			}
		}
		virtual void get_gradients(Map &gradients) {
			for (Layer<T> *p = output; p != nullptr; p = p->input) {
				p->update(gradients);
			}
		}
	public:
		Model(vector<Layer<T>*> &layers) : Layer<T>(NULL) {
			int num = layers.size();
			setInput(layers[0]);
			setOutput(layers[num-1]);
			__init_model_(layers); 
		}
		void compile(std::string &loss, Optimizer<T> &optimizer) {
			this->loss_func = loss;
			this->optimizer = optimizer;
		}
		void train(Tensor<T> &x, Tensor<T> &y) {
			Map<T> variable, gradients;
			get_variables(variable);// collect variables
			for (int i = 0; i < 100; i++) {
				
				Tensor<T> y_ = forward(x);
				
				Tensor<T> delta = y_ - y;
				Tensor<T> loss = ops::cross_entropy_loss(ops::softmax(y_), y);
				printf("Loss: %.8f\n", loss(0));
				
				Tensor<T> delta_b = backward(delta);

				get_gradients(gradients);// collect gradients
			}
			printf("training finished");
		}
		virtual void setInput(Layer<T> *input) {
			this->input = input;
		}
		virtual void setOutput(Layer<T> *output) {
			this->output = output;
		}
		virtual Tensor<T> forward(Tensor<T> &data) {
			// 对于Model类，先执行父类forward方法，再执行output的forward方法；
			return output->forward(Layer<T>::forward(data));
		}
		virtual Tensor<T> backward(Tensor<T> &delta) {
			// 对于Model类，先执行output的backward方法，再执行父类的forward方法；
			return Layer<T>::backward(output->backward(delta));
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
			std::map<std::string, Tensor<T>> weights;
			Network() {
				printf_s("Network()\n");
				weights["w0"] = Tensor<T>(13, 11);
				weights["w1"] = Tensor<T>(11, 7);
				weights["w2"] = Tensor<T>(7, 3);
				weights["b0"] = Tensor<T>(1, 11);
				weights["b1"] = Tensor<T>(1, 7);
				weights["b2"] = Tensor<T>(1, 3);
				randomize();
			}
			void fit(Tensor<T> &x_train, Tensor<T> &y_train) {
				printf_s("fit()\n");
				y_train.save("y_train-one_hot.txt");
				Shape shape = y_train.shape();
				int num_samples = shape[0];
				double last = INFINITE;
				std::map<std::string, Tensor<T>> gradients;
				for (int i = 0; i < epochs; i++) {
					for (int start = 0; start < num_samples; start += batch_size) {
						int end = start + batch_size;
						end = min(end, num_samples);
						Tensor<T> X0 = x_train.slice(start, end, 1);
						Tensor<T> Y0 = y_train.slice(start, end, 1);

						// 正向传播
						Tensor<T> T0 = X0.matmul(weights["w0"]).add(weights["b0"]);
						Tensor<T> O1 = T0.sigmoid();

						Tensor<T> T1 = O1.matmul(weights["w1"]).add(weights["b1"]);
						Tensor<T> O2 = T1.sigmoid();

						Tensor<T> T2 = O2.matmul(weights["w2"]).add(weights["b2"]);
						Tensor<T> O3 = T2.sigmoid();
						Tensor<T> O4 = ops::softmax(O3);

						// 反向传播
						Tensor<T> D2 = ops::grad_sigmoid(O3) * (O3 - Y0);
						Tensor<T> D1 = ops::grad_sigmoid(O2) * D2.matmul(weights["w2"].Transpose());
						Tensor<T> D0 = ops::grad_sigmoid(O1) * D1.matmul(weights["w1"].Transpose());

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

					Tensor<T> y_pred = predict(x_train);
					Tensor<T> loss = ops::cross_entropy_loss(y_pred, y_train);
					printf("epochs:%5d\t loss:%.8f\n", i, loss.data[0][0]);
					last = loss.data[0][0];
				}
			}
			void randomize() {
				std::map<std::string, Tensor<T>>::iterator iter;
				for (iter = weights.begin(); iter != weights.end(); iter++) {
					std::string name = iter->first;
					weights[name].randomize();
				}
			}
			void optimize(std::map<std::string, Tensor<T>> &gradients, double learning_rate) {
				std::map<std::string, Tensor<T>>::iterator iter;
				for (iter = weights.begin(); iter != weights.end(); iter++) {
					std::string name = iter->first;
					weights[name] = weights[name] - learning_rate * gradients[name];
				}
			}
			Tensor<T> predict(Tensor<T> &x_test) {
				Tensor<T> net;
				net = x_test.matmul(weights["w0"]).add(weights["b0"]).sigmoid();
				net = net.matmul(weights["w1"]).add(weights["b1"]).sigmoid();
				net = net.matmul(weights["w2"]).add(weights["b2"]).sigmoid();
				net = ops::softmax(net);
				return net;
			}
			friend ostream& operator << (ostream& out, Network<T> &network) {
				std::map<std::string, Tensor<T>> weights = network.weights;
				std::map<std::string, Tensor<T>>::iterator iter;
				for (iter = weights.begin(); iter != weights.end(); iter++) {
					out >> iter->second;
				}
				return out;
			}
			friend istream& operator >> (istream& in, Network<T> &network) {
				std::map<std::string, Tensor<T>>::iterator iter;
				for (iter = weights.begin(); iter != weights.end(); iter++) {
					in << iter->second;
				}
				return in;
			}
		};

		template<class T>
		void test(Tensor<T> &x_train, Tensor<T> &y_train, Tensor<T> &x_test) {

			printf("bp_network::test()\n");

			Network<T> net;
			net.load_weights("Text.txt");
			net.fit(x_train, y_train.one_hot(3));
			net.save_weights("Text.txt");

			Tensor<T> y_test = net.predict(x_test);
			y_test.savemat("softmax.txt");
		}
	}

	namespace fc_network {

		template<class T>
		Model<T>* create_fc_network() {

			int size[] = { NULL, NULL, 28, 28, 3 };

			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));
			layers.push_back(new Conv2D(3, 0, 4, 96));// width=3, pad=0, stride=4, n_filters=96
			layers.push_back(new MaxPooling<T>(5)); // width=5
			layers.push_back(new Flatten<T>());
			layers.push_back(new FullyConnected<T>(10, "sigmoid"));
			
			return new Model<T>(layers);
		}

		template<class T>
		void test(Tensor<T> &x_train, Tensor<T> &y_train, Tensor<T> &x_test) {

			printf("fc_network::test()\n");
			int size[] = { 500, 1, 28, 28, 3 };
			Model<T> *fc_network = create_fc_network<T>();
			fc_network->compile(std::string("least_squares"), Optimizer<T>(0.9));
			Tensor<T> x = x_train.slice(0, 500, 1).reshape(size);
			Tensor<T> y = y_train.one_hot(3);
			fc_network->train(x, y);
		}
	}

	namespace auto_encoder {

		template<class T>
		Model<T>* create_encoder() {

			int size[] = { NULL, 13 };

			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));// value
			layers.push_back(new FullyConnected<T>(11, "sigmoid"));
			layers.push_back(new FullyConnected<T>(9, "sigmoid"));
			layers.push_back(new FullyConnected<T>(7, "sigmoid"));
			layers.push_back(new FullyConnected<T>(5, "sigmoid"));
			layers.push_back(new FullyConnected<T>(3, "sigmoid"));// calculate

			return new Model<T>(layers);
		}

		template<class T>
		Model<T>* create_decoder() {
			
			int size[] = { NULL, 3 };
			
			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));
			layers.push_back(new FullyConnected<T>(5, "sigmoid"));
			layers.push_back(new FullyConnected<T>(7, "sigmoid"));
			layers.push_back(new FullyConnected<T>(9, "sigmoid"));
			layers.push_back(new FullyConnected<T>(11, "sigmoid"));
			layers.push_back(new FullyConnected<T>(13, "sigmoid"));
			
			return new Model<T>(layers);
		}

		template<class T>
		Model<T>* create_auto_encoder() {

			Model<T>* encoder = create_encoder<T>();
			Model<T>* decoder = create_decoder<T>();

			int size[] = { NULL, 13 };
			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));
			layers.push_back(encoder);
			layers.push_back(decoder);
			
			return new Model<T>(layers);
		}

		template<class T>
		void test(Tensor<T> &x_train, Tensor<T> &y_train, Tensor<T> &x_test) {

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
			layers.push_back(new FullyConnected<T>(5, "sigmoid"));
			layers.push_back(new FullyConnected<T>(7, "sigmoid"));
			layers.push_back(new FullyConnected<T>(9, "sigmoid"));
			layers.push_back(new FullyConnected<T>(11, "sigmoid"));
			layers.push_back(new FullyConnected<T>(13, "sigmoid"));

			return new Model<T>(layers);
		}

		template<class T>
		Model<T>* create_discriminator() {

			int size[] = { NULL, 13 };

			vector<Layer<T>*> layers;
			layers.push_back(new Input<T>(size));
			layers.push_back(new FullyConnected<T>(11, "sigmoid"));
			layers.push_back(new FullyConnected<T>(9, "sigmoid"));
			layers.push_back(new FullyConnected<T>(7, "sigmoid"));
			layers.push_back(new FullyConnected<T>(5, "sigmoid"));
			layers.push_back(new FullyConnected<T>(1, "sigmoid"));

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
		void test(Tensor<T> &x_train, Tensor<T> &y_train, Tensor<T> &x_test) {

			printf("gan::test()\n");

			Model<T>* gan = create_gan<T>();
			gan->compile(std::string("least_squares"), Optimizer<T>(0.9));
			//gan->train(x_train, x_train);
		}
	}

	template<class T>
	void test();
}

#endif // !_MODEL_H