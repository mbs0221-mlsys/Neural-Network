#include "tensor.h"

using namespace std;
using namespace shape;
using namespace tensor;

template class Tensor<double>;

typedef Tensor<double> DoubleTensor;
typedef Tensor<float> FloatTensor;
typedef Tensor<int> IntTensor;

template void tensor::test_basic<double>();
template void tensor::test_conv<double>();
template void tensor::test_pooling<double>();

int after[] = { 0, 1, 3, 4, 2 };
int before[] = { 0, 1, 4, 2, 3 };

template<class T>
void tensor::test_basic() {

	printf("tensor::test()\n");

	int size[] = { 1,1,1,4,3 };
	Shape shape(size);
	Tensor<T> x = Tensor<T>::ones(shape);
	x.print();
	Tensor<T> b = Tensor<T>::zeros(shape);
	b.print();
	Tensor<T> c = Tensor<T>::mask(shape, 0.2);
	c.print();
	Tensor<T> d = Tensor<T>::eye(4);
	d.print();

	cout << "c.Transpose().sigmoid().print();" << endl;
	c.Transpose().sigmoid().print();
	cout << "x.matmul(c.Transpose()).reduce_mean(2).reduce_mean(3).print();" << endl;
	x.matmul(c.Transpose()).reduce_mean(3).reduce_mean(4).print();
}

template<class T>
void tensor::test_conv() {

	Tensor<T> tensor, filter, bias;
	ifstream inf("tensor.txt");
	if (inf.is_open()) {
		inf >> tensor;
		inf >> filter;
		inf >> bias;
		inf.close();
	}

	ofstream outf("tensor.txt");
	if (outf.is_open()) {
		outf << tensor << endl;
		outf << filter << endl;
		outf << bias << endl;
		outf.close();
	}

	getchar();

	cout << "mat;" << endl;
	tensor.padding(1).permute(before).print();

	cout << "filter;" << endl;
	filter.permute(before).print();// (:,:,channel,row,col)

	cout << "bias;" << endl;
	bias.print();

	cout << "filter.rotate180().permute(before);" << endl;
	filter.rotate180().permute(before).print();
	
	cout << "conv2d(no bias);" << endl;
	tensor.padding(1).conv2d(filter, 2).permute(before).print();

	cout << "conv2d;" << endl;
	tensor.padding(1).conv2d(filter, bias, 2).permute(before).print();

	cout << "conv3d;" << endl;
	tensor.padding(1).conv3d(filter, bias, 2).permute(before).print();

	getchar();
}

template<class T>
void tensor::test_pooling() {

	Tensor<T> tensor;

	tensor.load("pooling.txt");
	tensor.save("pooling.txt");

	cout << "pooling;" << endl;
	tensor.permute(before).print();
	
	cout << "rotate180;" << endl;
	tensor.rotate180().permute(before).print();

	cout << "pooling;" << endl;
	tensor.max_pooling(2).permute(before).print();
	tensor.min_pooling(2).permute(before).print();
	tensor.avg_pooling(2).permute(before).print();

	cout << "upsampling;" << endl;
	tensor.max_pooling(2).upsampling(tensor, 2).permute(before).print();
	tensor.min_pooling(2).upsampling(tensor, 2).permute(before).print();
	tensor.avg_pooling(2).avg_upsampling(2).permute(before).print();

}