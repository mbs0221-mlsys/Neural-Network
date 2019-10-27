#include "tensor.h"


using namespace std;
using namespace shape;
using namespace tensor;

template class Tensor<double>;

template void tensor::test_basic<double>();
template void tensor::test_conv<double>();
template void tensor::test_pooling<double>();

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

	Tensor<T> mat, filter, bias;
	ifstream inf("tensor.txt");
	if (inf.is_open()) {
		inf >> mat;
		inf >> filter;
		inf >> bias;
		inf.close();
	}
	int after[] = { 0, 1, 3, 4, 2 };
	int before[] = { 0, 1, 4, 2, 3 };

	//ofstream outf("tensor.txt");
	//if (outf.is_open()) {
	//	outf << mat.permute(after) << endl;
	//	outf << filter.permute(after) << endl;
	//	outf << bias << endl;
	//	outf.close();
	//}

	cout << "mat;" << endl;
	mat.padding(1).permute(before).print();

	cout << "filter;" << endl;
	filter.permute(before).print();// (:,:,channel,row,col)

	cout << "bias;" << endl;
	bias.print();

	cout << "filter.rotate180().permute(before);" << endl;
	filter.rotate180().permute(before).print();

	cout << "conv2d;" << endl;
	mat.padding(1).conv2d(filter, bias, 2).permute(before).print();

	cout << "conv3d;" << endl;
	mat.padding(1).conv3d(filter, bias, 2).permute(before).print();

	getchar();
}

template<class T>
void tensor::test_pooling() {
	int after[] = { 0, 1, 3, 4, 2 };
	int before[] = { 0, 1, 4, 2, 3 };

	Tensor<T> pooling;
	pooling.load("pooling.txt");
	pooling.print();
	cout << "pooling;" << endl;
	pooling.permute(after).max_pooling(2).permute(before).print();
	pooling.permute(after).min_pooling(2).permute(before).print();
	pooling.permute(after).avg_pooling(2).permute(before).print();

	cout << "upsampling;" << endl;
	Tensor<T> x = pooling.permute(after).max_pooling(2);
	x.max_upsampling(pooling.permute(after), 2).permute(before).print();

	Tensor<T> y = pooling.permute(after).min_pooling(2);
	y.min_upsampling(pooling.permute(after), 2).permute(before).print();

	Tensor<T> z = pooling.permute(after);
	z = z.avg_pooling(2);
	z = z.avg_upsampling(2);
	z = z.permute(before);
	z.print();

	pooling.permute(after).rotate180().permute(before).print();
}