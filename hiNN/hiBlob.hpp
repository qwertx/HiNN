#ifndef __HIBLOB_HPP__
#define __HIBLOB_HPP__
#include <string>
#include <armadillo>

using std::vector;
using std::string;
using arma::cube;

// cubeʵ���Ͼ���tensor
enum CubeFill { 
	C_ONES = 1,    // cube���Ϊ1
	C_ZEROS = 2,   // cube���Ϊ0
	C_RANDU = 3,   // cube�����0-1����ȷֲ�
	C_RANDN = 4,   // cube���Ϊ��׼��̬�ֲ�
	C_DEFAULT = 5  // cube�ṩ��Ĭ�����
};

// ÿһ����һ������Blob, һ���ݶ�Blob
// ÿ��Blob��N��Cube���ɣ�һ��Cube����һ��tensor
class Blob {
public:
	Blob(): N_(0), C_(0), H_(0), W_(0) {}
	Blob(int n, int c, int h, int w, int cube_fill = C_DEFAULT);
	Blob(const vector<int>& shape, int cube_fill = C_DEFAULT);
	void print_blob(string str = "") const;
	const vector<cube>& get_cubes() const;
	cube& operator[](int i);
	const cube& operator[](int i) const;
	Blob& operator=(double val);
	friend Blob operator+(const Blob& a, const Blob& b);
	friend Blob operator+(double val, const Blob& a);
	friend Blob operator*(const Blob& a, const Blob& b);
	friend Blob operator*(double val, const Blob& a);
	friend Blob operator/(const Blob& a, const Blob& b);
	friend Blob operator/(const Blob& a, double val);
	friend Blob sqrt(const Blob& a);
	friend Blob square(const Blob& a);
	friend double accu(const Blob& a); // ��Blob������Ԫ�����
	Blob& operator*= (const double i);
	Blob splitBlob(int start, int end) const;
	Blob pad(int pad_size, double pad_val = 0.0);
	Blob de_pad(int pad_size);
	vector<int> size() const;
	inline int get_N() const {
		return N_;
	}
	inline int get_C() const {
		return C_;
	}
	inline int get_H() const {
		return H_;
	}
	inline int get_W() const {
		return W_;
	}

private:
	int N_; // ������/����˸���
	int C_; // ��Ԫ��/ͨ����
	int H_; // �����/����ĸߣ���FC���Ϊ1
	int W_; // �����/����Ŀ���FC���Ϊ1
	vector<cube> blob_cubes;

	void fill_cubes(const int n, const int c, const int h, const int w, int cube_fill);
};

#endif