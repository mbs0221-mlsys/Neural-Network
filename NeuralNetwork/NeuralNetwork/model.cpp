#include "model.h"

template void model::test<double>();

template<class T>
void model::test() {
	// run test module
	printf("model::test()\n");

	srand((unsigned int)time(NULL));

	//Tensor<T> x_train, y_train, x_test;

	//x_train.load("x_train.txt");
	//y_train.load("y_train.txt");
	//x_test.load("x_test.txt");

	//Mat im = imread("timg.jpg", 1);
	//imshow("ͼƬ", im);
	//Tensor<T> tensor = image::im2tensor<T>(im);

	//x_train.print();
	//y_train.print();
	//x_test.print();

	//x_train.save("x_train.txt");
	//y_train.save("y_train.txt");
	//x_test.save("x_test.txt");

	//bp_network::test<T>(x_train, y_train, x_test);
	//fc_network::test<T>(x_train, y_train, x_test);
	//auto_encoder::test<T>(x_train, y_train, x_test);
	//gan::test<T>(x_train, y_train, x_test);
}