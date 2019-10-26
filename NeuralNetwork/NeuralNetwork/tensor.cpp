#include "tensor.h"

using namespace std;
using namespace shape;
using namespace tensor;

template class Tensor<double>;
template void tensor::test<double>();


template<class T>
void tensor::test() {

	printf("tensor::test()\n");
	
	int size[] = { 1,1,4,3 };
	Shape shape(size);
	
	{
		Tensor<T> x = Tensor<T>::ones(shape);
		x.print();
		Tensor<T> b = Tensor<T>::zeros(shape);
		b.print();
		Tensor<T> c = Tensor<T>::mask(shape, 0.2);
		c.print();
		cout << "c.Transpose().sigmoid().print();" << endl;
		c.Transpose().sigmoid().print();
		cout << "x.matmul(c.Transpose()).reduce_mean(2).reduce_mean(3).print();" << endl;
		x.matmul(c.Transpose()).reduce_mean(2).reduce_mean(3).print();
	}

	{
		Tensor<T> mat, filter;
		mat.load("mat.txt");
		filter.load("filter.txt");

		cout << "mat.print();" << endl;
		mat.print();
		cout << "filter.print();" << endl;
		filter.print();

		cout << "x_train.padding(1).print();" << endl;
		mat.padding(1).print();

		//cout << "ones.conv2d(filter, 1).print();" << endl;


		int order[] = { 0, 2, 3, 1 };
		filter = filter.permute(order);
		cout << "filter.permute(order);" << endl;
		filter.print();

		int order1[] = { 0, 3, 1, 2 };
		mat.permute(order).padding(1).permute(order1).print();

		mat.permute(order).padding(1).conv3d(filter, 2).permute(order1).print();
		//mat.padding(1).conv3d(filter, 2).permute(order1).print();
		//mat.print();

		int after[] = { 1, 1, 1, 75 };
		int before[] = { 1, 3, 5, 5 };

		cout << "mat.reshape(after).reshape(before).print();" << endl;
		mat.reshape(after).reshape(before).print();

		cout << "mat.flatten().reshape(before).print();" << endl;
		mat.flatten().reshape(before).print();
		getchar();
	}
}