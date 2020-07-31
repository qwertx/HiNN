#include <iostream>
#include <string>
#include <memory>
#include "hiNet.hpp"
#include "hiBlob.hpp"
#include "hiLayer.hpp"

using namespace std;

// 以下3个函数用于读取MNIST数据集
// 大端转小端
int ReverseEndian(int i) {  
	unsigned char ch1, ch2, ch3, ch4;  //　一个int有4个char大小
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMnistImages(string path, shared_ptr<Blob> images) {
	ifstream file(path, ios::binary);
	if(file.is_open()) {
		// MNIST原始数据文件中32位的整型值是大端存储，C/C++变量是小端存储，所以读取数据的时候，需要对其进行大小端转换
		// meta info
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseEndian(magic_number);
		cout << "magic_number = " << magic_number << endl;

		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseEndian(number_of_images);
		cout << "number_of_images = " << number_of_images << endl;

		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseEndian(n_rows);
		cout << "n_rows = " << n_rows << endl;

		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseEndian(n_cols);
		cout << "n_cols = " << n_cols << endl;

		// 遍历图片，存储到Blob
		for(int i = 0; i < number_of_images; i++) {
			for(int h = 0; h < n_rows; h++) {
				for(int w = 0; w < n_cols; w++) {
					unsigned char tmp = 0;
					file.read((char*)&tmp, sizeof(tmp));	
					// 写入Blob, 顺便归一化
					(*images)[i](h, w, 0) = static_cast<double>(tmp) / 255;
				}
			}
		}
	}
	else {
		cout << "no data file found :-(" << endl;
	}

}
void ReadMnistLabels(string path, shared_ptr<Blob> labels) {
	ifstream file(path, ios::binary);
	if(file.is_open()) {
		// meta info
		int magic_number = 0;
		int number_of_labels = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseEndian(magic_number);
		cout << "magic_number = " << magic_number << endl;

		file.read((char*)&number_of_labels, sizeof(number_of_labels));
		number_of_labels = ReverseEndian(number_of_labels);
		cout << "number_of_Labels = " << number_of_labels << endl;

		// 遍历标签，存储到Blob 
		for(int i = 0; i < number_of_labels; i++) {
			unsigned char tmp = 0;
			file.read((char*)&tmp, sizeof(tmp)); // tmp值为0-9
			// 写入Blob, 顺便归一化
			(*labels)[i](0, 0, (int)tmp) = 1;
		}
	}
	else {
		cout << "no label file found :-(" << endl;
	}
}


// 使用训练集的一部分作为验证集训练
void train(NetParams& net_params, shared_ptr<Blob> X, shared_ptr<Blob> Y) {
	// 划分训练集和验证集
	shared_ptr<Blob> X_train(new Blob(X->splitBlob(0, 59000)));
	shared_ptr<Blob> Y_train(new Blob(Y->splitBlob(0, 59000)));
	shared_ptr<Blob> X_validate(new Blob(X->splitBlob(59000, 60000)));
	shared_ptr<Blob> Y_validate(new Blob(Y->splitBlob(59000, 60000)));

	// 初始化网络
	Net model(net_params, X_train, Y_train, X_validate, Y_validate);

	// 训练
	cout << "start training..." << endl;
	model.train(net_params);
	cout << "training complete." << endl;
}


// 使用外部验证集训练
void train_with_ex_val(NetParams& net_params, shared_ptr<Blob> X_train, shared_ptr<Blob> Y_train, 
					   shared_ptr<Blob> X_val, shared_ptr<Blob> Y_val) {
	// 初始化网络
	Net model(net_params, X_train, Y_train, X_val, Y_val);

	// 训练
	cout << "start training..." << endl;
	model.train(net_params);
	cout << "training complete." << endl;
}


int main() {
	// 读取训练需要的参数
	string configFile{ "./model_cnn.json" };
	NetParams net_params;
	net_params.readNetParams(configFile);
	
	// 读取训练集
    shared_ptr<Blob> images_train(new Blob(60000, 1, 28, 28, C_ZEROS));
	shared_ptr<Blob> labels_train(new Blob(60000, 10, 1, 1, C_ZEROS));

	ReadMnistImages("./mnist_data/train/train-images.idx3-ubyte", images_train);
	ReadMnistLabels("./mnist_data/train/train-labels.idx1-ubyte", labels_train);

	// 读取测试集
	shared_ptr<Blob> images_test(new Blob(10000, 1, 28, 28, C_ZEROS));
	shared_ptr<Blob> labels_test(new Blob(10000, 10, 1, 1, C_ZEROS));

	ReadMnistImages("./mnist_data/test/t10k-images.idx3-ubyte", images_test);
	ReadMnistLabels("./mnist_data/test/t10k-labels.idx1-ubyte", labels_test);

	// 仅使用1000张图片训练，测试正则化效果
	shared_ptr<Blob> images_few(new Blob(images_train->splitBlob(0, 1000)));
	shared_ptr<Blob> labels_few(new Blob(labels_train->splitBlob(0, 1000)));
	shared_ptr<Blob> images_few_test(new Blob(images_test->splitBlob(0, 1000)));
	shared_ptr<Blob> labels_few_test(new Blob(labels_test->splitBlob(0, 1000)));
	train_with_ex_val(net_params, images_few, labels_few, images_few_test, labels_few_test);

	//train(net_params, images_train, labels_train);

	return 0;
}