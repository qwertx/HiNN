#include "hiBlob.hpp"
#include <iostream>

using namespace std;
using namespace arma;

Blob::Blob(int n, int c, int h, int w, int cube_fill) : N_(n), C_(c), H_(h), W_(w) {
	arma_rng::set_seed_random(); // �����ʼ������
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
	// ȷ������Blob�ߴ���ͬ
	vector<int> size_a = a.size();
	vector<int> size_b = b.size();

	for(int i = 0; i < size_a.size(); i++) {
		if(size_a[i] != size_b[i]) {
			throw "two blobs with different size cannot be added!";
		}
	}

	// ��ӦԪ�ص����
	Blob res(size_a);
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] + b[i];
	}
	return res;
}


Blob operator+(double val, const Blob& a) {
	// Blob��ÿ��cube������val
	Blob res(a.size());
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] + val;
	}
	return res;
}



Blob operator*(const Blob& a, const Blob& b) {
	// ȷ������Blob�ߴ���ͬ
	vector<int> size_a = a.size();
	vector<int> size_b = b.size();

	for(int i = 0; i < size_a.size(); i++) {
		if(size_a[i] != size_b[i]) {
			throw "two blobs with different size cannot be multiplied!";
		}
	}

	// ��ӦԪ�ص����
	Blob res(size_a);
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] % b[i];
	}
	return res;
}


Blob operator*(double val, const Blob& a) {
	// Blob��ÿ��cube������val
	Blob res(a.size());
	for(int i = 0; i < a.get_N(); i++) {
		res[i] = a[i] * val;
	}
	return res;
}


Blob operator/(const Blob& a, const Blob& b) {
	// ȷ������Blob�ߴ���ͬ
	vector<int> size_a = a.size();
	vector<int> size_b = b.size();

	for(int i = 0; i < size_a.size(); i++) {
		if(size_a[i] != size_b[i]) {
			throw "two blobs with different size cannot be multiplied!";
		}
	}

	// ��ӦԪ�ص����
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
	// ʵ����end > start��������������minibatchѵ��
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
	padded = pad_val; // ��ʼ��
	// ����ԭBlob�е�ֵ
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
					// ȥ��pad���Blob�Ŀ�ߴ�0��ʼ����
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