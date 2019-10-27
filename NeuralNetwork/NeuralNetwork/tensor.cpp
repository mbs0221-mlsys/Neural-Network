#include "tensor.h"

using namespace std;
using namespace shape;
using namespace tensor;

template class Tensor<double>;
template void tensor::test<double>();

template<class T>
void tensor::test() {

	printf("tensor::test()\n");
	{
		//int size[] = { 1,1,1,4,3 };
		//Shape shape(size);
		//Tensor<T> x = Tensor<T>::ones(shape);
		//x.print();
		//Tensor<T> b = Tensor<T>::zeros(shape);
		//b.print();
		//Tensor<T> c = Tensor<T>::mask(shape, 0.2);
		//c.print();
		//cout << "c.Transpose().sigmoid().print();" << endl;
		//c.Transpose().sigmoid().print();
		//cout << "x.matmul(c.Transpose()).reduce_mean(2).reduce_mean(3).print();" << endl;
		//x.matmul(c.Transpose()).reduce_mean(3).reduce_mean(4).print();
		//getchar();
	}
	int after[] = { 0, 1, 3, 4, 2 };
	int before[] = { 0, 1, 4, 2, 3 };
	{
		/*Tensor<T> mat, filter, bias;
		ifstream inf("mat.txt");
		if (inf.is_open()) {
			inf >> mat;
			inf >> filter;
			inf >> bias;
			inf.close();
		}
		int after[] = { 0, 1, 3, 4, 2 };
		int before[] = { 0, 1, 4, 2, 3 };

		cout << "mat.permute(order).padding(1);" << endl;
		mat.permute(after).padding(1).permute(before).print();

		filter.print();
		filter = filter.permute(after);

		bias.print();

		cout << "mat.permute(after).padding(1).conv2d(filter, 2).permute(before);" << endl;
		mat = mat.permute(after).padding(1).conv2d(filter, bias, 2).permute(before);
		mat.print();
		getchar();*/
	}
	{
		Tensor<T> pooling;
		pooling.load("pooling.txt");
		pooling.print();
		cout << "pooling;" << endl;
		pooling.permute(after).max_pooling(2).permute(before).print();
		pooling.permute(after).min_pooling(2).permute(before).print();
		pooling.permute(after).avg_pooling(2).permute(before).print();

		auto x = pooling.permute(after).max_pooling(2);
		x.max_upsampling(pooling.permute(after), 2).permute(before).print();

		auto y = pooling.permute(after).min_pooling(2);
		y.min_upsampling(pooling.permute(after), 2).permute(before).print();

		auto z = pooling.permute(after).avg_pooling(2);
		z.avg_upsampling(2).permute(before).print();

		//pooling.permute(after).rotate180().permute(before).print();

		//cout << "filter.rotate180().permute(before);" << endl;
		//filter.rotate180().permute(before).print();

	
		getchar();
	}
}