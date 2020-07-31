#include "hiBlob.hpp"
#include <iostream>

using namespace std;
using namespace arma;

Blob::Blob(int n, int c, int h, int w, int cube_fill) : N_(n), C_(c), H_(h), W_(w) {
	arma_rng::set_seed_random(); // 随机初始化种子
	fill_cubes(N_, C_, H_, W_, cube_fill);
}


Blob::Blob(const vector<int>& shape, int cube_fill) : N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3]) {
	arma_rng::set_seed_random();
	fill_cubes(N_, C_, H_, W_, cube_fill);
}


void Blob::fill_cubes(const int n, const int c, const int h, const int w, int cube_fill) {
	if(cube_fill == C_ONES) {
		blob_cubes = vector<cube>(n, cube(h, w, c, fill::ones));
	}
	else if(cube_fill == C_ZEROS) {
		blob_cubes = vector<cube>(n, cube(h, w, c, fill::zeros));
	}
	else if(cube_fill == C_DEFAULT) {
		blob_cubes = vector<cube>(n, cube(h, w, c));
	}
	else if(cube_fill == C_RANDU) {
		for(int i = 0; i < n; i++) {
			blob_cubes.push_back(randu<cube>(h, w, c));
		}
	}
	else if(cube_fill == C_RANDN) {
		for(int i = 0; i < n; i++) {
			blob_cubes.push_back(randn<cube>(h, w, c));
		}
	}
}


void Blob::print_blob(string str) const {
	cout << str << endl;
	if(blob_cubes.empty()) {
		cout << "blob is empty!" << endl;
		return;
	}

	for(int i = 0; i < N_; i++) {
		cout << "N = " << i << endl;
		this->blob_cubes[i].print();
	}
}


cube& Blob::operator[](int i) {
	return blob_cubes[i];
}


const cube& Blob::operator[](int i) const {
	return blob_cubes[i];
}


Blob& Blob::operator=(double val) {
	for(cube& curr_cube : blob_cubes) {
		curr_cube.fill(val);
	}
	return *this;
}


Blob operator+(const Blob& a, const Blob& b) {
	// 确保两个Blob尺寸相同
	vector<int> size_a = a.size();
	vector<int> size_b = b.size();

	for(int i = 0; i < size_a.size(); i++) {
		if(size_a[i] != size_b[i]) {
			throw "two blobs with different size cannot be added!";
		}
	}

	// 对应元素的相加
	Blob res(size_a);
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] + b[i];
	}
	return res;
}


Blob operator+(double val, const Blob& a) {
	// Blob中每个cube都加上val
	Blob res(a.size());
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] + val;
	}
	return res;
}



Blob operator*(const Blob& a, const Blob& b) {
	// 确保两个Blob尺寸相同
	vector<int> size_a = a.size();
	vector<int> size_b = b.size();

	for(int i = 0; i < size_a.size(); i++) {
		if(size_a[i] != size_b[i]) {
			throw "two blobs with different size cannot be multiplied!";
		}
	}

	// 对应元素的相乘
	Blob res(size_a);
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] % b[i];
	}
	return res;
}


Blob operator*(double val, const Blob& a) {
	// Blob中每个cube都乘上val
	Blob res(a.size());
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] * val;
	}
	return res;
}


Blob operator/(const Blob& a, const Blob& b) {
	// 确保两个Blob尺寸相同
	vector<int> size_a = a.size();
	vector<int> size_b = b.size();

	for(int i = 0; i < size_a.size(); i++) {
		if(size_a[i] != size_b[i]) {
			throw "two blobs with different size cannot be multiplied!";
		}
	}

	// 对应元素的相除
	Blob res(size_a);
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] / b[i];
	}
	return res;
}


Blob operator/(const Blob& a, double val) {
	Blob res(a.size());
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] / val;
	}
	return res;
}


Blob sqrt(const Blob& a) {
	Blob res(a.size());
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = arma::sqrt(a[i]);
	}
	return res;
}


Blob square(const Blob& a) {
	Blob res(a.size());
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = arma::square(a[i]);
	}
	return res;
}


double accu(const Blob& a) {
	double sum = 0.0;
	for(int i = 0; i < a.get_N(); i++) {
		sum += arma::accu(a[i]);
	}
	return sum;
}


Blob& Blob::operator*= (const double i) {
	for(cube& curr_cube: blob_cubes) {
		curr_cube *= i;
	}
	return *this;
}


const vector<cube>& Blob::get_cubes() const {
	return blob_cubes;
}


Blob Blob::splitBlob(int start, int end) const {
	// 实现了end > start的情况，方便进行minibatch训练
	if(end >= start) {
		Blob tmp(end - start, C_, H_, W_);
		for(int i = start; i < end; i++) {
			tmp[i - start] = blob_cubes[i];
		}
		return tmp;
	}
	else {
		// [0,1,2,3,4,5] -> [3,1) -> [3,6)+[0,1) -> [3,4,5,0]
		Blob tmp(N_ - start + end, C_, H_, W_);
		for(int i = start; i < N_; i++) {
			tmp[i - start] = blob_cubes[i];
		}
		for(int i = 0; i < end; i++) {
			tmp[i + N_ - start] = blob_cubes[i];
		}
		return tmp;
	}
}


Blob Blob::pad(int pad_size, double pad_val) {
	if(blob_cubes.empty()) {
		throw "current blob is empty!";
	}
	Blob padded(N_, C_, H_ + 2 * pad_size, W_ + 2 * pad_size);
	padded = pad_val; // 初始化
	// 复制原Blob中的值
	for(int n = 0; n < N_; n++) {
		for(int c = 0; c < C_; c++) {
			for(int h = 0; h < H_; h++) {
				for(int w = 0; w < W_; w++) {
					padded[n](h + pad_size, w + pad_size, c) = blob_cubes[n](h, w, c);
				}
			}
		}
	}
	return padded;
}


Blob Blob::de_pad(int pad_size) {
	if(blob_cubes.empty()) {
		throw "current blob is empty!";
	}
	Blob de_padded(N_, C_, H_ - 2 * pad_size, W_ - 2 * pad_size);

	for(int n = 0; n < N_; n++) {
		for(int c = 0; c < C_; c++) {
			for(int h = pad_size; h < H_ - pad_size; h++) {
				for(int w = pad_size; w < W_ - pad_size; w++) {
					// 去除pad后的Blob的宽高从0开始索引
					de_padded[n](h - pad_size, w - pad_size, c) = blob_cubes[n](h, w, c);
				}
			}
		}
	}
	return de_padded;
}


vector<int> Blob::size() const {
	vector<int> shape{N_, C_, H_, W_};
	return shape;
}