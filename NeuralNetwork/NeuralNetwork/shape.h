#pragma once

#ifndef _SHAPE_H_
#define _SHAPE_H_

class Shape {
private:
	int dims[4] = { 0, 0, 1, 1 };
public:
	Shape() { 	}
	Shape(const Shape & s) {
		dims[0] = s.dims[0];
		dims[1] = s.dims[1];
		dims[2] = s.dims[2];
		dims[3] = s.dims[3];
	}
	Shape(int size[]) {
		dims[0] = size[0];
		dims[1] = size[1];
		dims[2] = size[2];
		dims[3] = size[3];
	}
	void setDims(int k, int axis) {
		dims[axis] = k;
	}
	int operator[](int k) const { return dims[k]; }
	inline int size() {
		return (dims[0] * dims[1] * dims[2] * dims[3]);
	}
};

#endif // !_SHAPE_H