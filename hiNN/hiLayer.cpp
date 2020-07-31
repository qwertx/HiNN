#include "hiLayer.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace arma;


// 将arma::Mat转为cv::Mat
template<typename T>
void arma2cv(const arma::Mat<T>& arma_mat, cv::Mat_<T>& cv_mat) {
	cv::transpose(cv::Mat_<T>(static_cast<int>(arma_mat.n_cols),
				  static_cast<int>(arma_mat.n_rows),
			      const_cast<T*>(arma_mat.memptr())), cv_mat);
}


// 将cube中所有Mat转为cv中的Mat，存入vector
void visualize(const cube& in_cube, vector<cv::Mat_<double>>& mat_vec) {
	int num = static_cast<int>(in_cube.n_slices);
	for(int i = 0; i < num; ++i) {
		cv::Mat_<double> cv_mat;
		arma::mat arma_mat = in_cube.slice(i);
		arma2cv<double>(arma_mat, cv_mat);
		mat_vec.push_back(cv_mat);
	}
}


void ConvLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	cout << "initiating ConvLayer" << endl;
	//  卷积核的N, C, H, W
	int N = curr_params.conv_num;
	int C = in_shape[1];
	int H = curr_params.conv_height;
	int W = curr_params.conv_width;

	// 初始化Blob
	// W: curr_blobs[1]
	if(!curr_blobs[1]) {
		curr_blobs[1].reset(new Blob(N, C, H, W, C_RANDN));
		if(curr_params.conv_init == "Gaussian") {
			// 将标准差改为0.01
			(*curr_blobs[1]) *= 1e-2;
		}
		else if(curr_params.conv_init == "He") { // He初始化
			(*curr_blobs[1]) *= std::sqrt(2 / static_cast<double>(in_shape[1] * in_shape[2] * in_shape[3]));
		}
		else if(curr_params.conv_init == "Xavier") { // Xavier初始化
			(*curr_blobs[1]) *= std::sqrt(1 / static_cast<double>(in_shape[1] * in_shape[2] * in_shape[3]));
		}
		else {
			throw "wrong init method!";
		}
	}
	// b: curr_blobs[2]，每个卷积核只有一个b
	if(!curr_blobs[2]) {
		curr_blobs[2].reset(new Blob(N, 1, 1, 1, C_RANDN));
		(*curr_blobs[2]) *= 1e-2;
	}
}


void ConvLayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	// 输入尺寸
	int N_in = in_shape[0];
	int C_in = in_shape[1];
	int H_in = in_shape[2];
	int W_in = in_shape[3];
	// 卷积核尺寸
	int c_N = curr_params.conv_num;
	int c_H = curr_params.conv_height;
	int c_W = curr_params.conv_width;
	int c_P = curr_params.conv_pad;
	int c_S = curr_params.conv_stride;
	// 输出 
	out_shape[0] = N_in; // 样本数
	out_shape[1] = c_N; // 输出通道数为卷积核个数
	out_shape[2] = (H_in + 2 * c_P - c_H) / c_S + 1;
	out_shape[3] = (W_in + 2 * c_P - c_W) / c_S + 1;
}


void ConvLayer::forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output,
							 const LayerParams& curr_params, string mode) {
	//cout << "ConvLayer::forward_prop" << endl;
	if(output) {
		output.reset();
	}
	if(input[0]->get_C() != input[1]->get_C()) {
		throw "input channels don't match conv channels!";
	}
	// 获取相关尺寸（输入/卷积核/输出）
	// 输入尺寸(样本个数/输入通道数/高/宽)
	int N_in = input[0]->get_N();
	int C_in = input[0]->get_C();
	int H_in = input[0]->get_H();
	int W_in = input[0]->get_W();
	// 卷积核尺寸，卷积核通道数 == 输入通道数，因此省略
	int N_conv = input[1]->get_N();
	int H_conv = input[1]->get_H();
	int W_conv = input[1]->get_W();
	int stride = curr_params.conv_stride;
	int pad_size = curr_params.conv_pad;
	// 输出尺寸
	int H_out = (H_in + 2 * pad_size - H_conv) / stride + 1;
	int W_out = (W_in + 2 * pad_size - W_conv) / stride + 1;

	// padding
	Blob X_padded = input[0]->pad(pad_size);

	// 可视化第一张图片
	//arma::mat mat0_arma = X_padded[0].slice(0);
	//cv::Mat_<double> mat0_cv;
	//arma2cv(mat0_arma, mat0_cv);

	// 卷积后通道数就是卷积核个数
	output.reset(new Blob(N_in, N_conv, H_out, W_out));
	// 卷积运算，遍历计算输出的每一个值
	for(int n = 0; n < N_in; n++) {
		for(int c = 0; c < N_conv; c++) {
			for(int h = 0; h < H_out; h++) {
				for(int w = 0; w < W_out; w++) {
					// 截取padding后的输入，3个span分别为起止row/col/slice(channel)，此处左右均为闭合区间
					cube slice = X_padded[n](span(h* stride, h* stride + H_conv - 1),
											 span(w* stride, w* stride + W_conv - 1),
											 span::all);
					// 卷积, sum(slice * conv_core) + b
					// 因为输出通道数就是卷积核个数，这里要遍历所有卷积核，因此索引为c
					// b: (c, 1, 1, 1) c=卷积核个数
					(*output)[n](h, w, c) = accu(slice % (*input[1])[c]) + as_scalar((*input[2])[c]);
				}
			}
		}
	}

	// 测试
	// 可视化第一个卷积核
	//vector<cv::Mat_<double>> mat_vec_c_w1;
	//visualize((*input[1])[0], mat_vec_c_w1);
	// 可视化第一个偏置b
	//vector<cv::Mat_<double>> mat_vec_c_b1;
	//visualize((*input[2])[0], mat_vec_c_b1);
	// 可视化第一个卷积核的输出
	//vector<cv::Mat_<double>> mat_vec_c_o1;
	//visualize((*output)[0], mat_vec_c_o1);
}


void ConvLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
							  shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	//cout << "ConvLayer::backward_prop" << endl;
	// 卷积层梯度尺寸
	grads[0].reset(new Blob(cache[0]->size(), C_ZEROS));
	grads[1].reset(new Blob(cache[1]->size(), C_ZEROS));
	grads[2].reset(new Blob(cache[2]->size(), C_ZEROS));

	// 上层输入梯度的尺寸(样本个数，卷积核个数，高，宽)
	int N_d = d_input->get_N();
	int C_d = d_input->get_C();
	int H_d = d_input->get_H();
	int W_d = d_input->get_W();

	// 卷积核参数
	int H_conv = curr_params.conv_height;
	int W_conv = curr_params.conv_width;
	int stride = curr_params.conv_stride;
	int pad_size = curr_params.conv_pad;

	// 参与实际反向传播计算的应该是padding后的输入X
	// dX要与X尺寸一致，因此dX也要进行padding
	Blob X_padded = cache[0]->pad(pad_size);
	Blob dX_padded(X_padded.size(), C_ZEROS);

	// 遍历上一层每个输入的梯度值(dY)
	for(int n = 0; n < N_d; n++) {
		for(int c = 0; c < C_d; c++) {
			for(int h = 0; h < H_d; h++) {
				for(int w = 0; w < W_d; w++) {
					// 由于前向计算时也是通过遍历该层的输出(Y)进行计算，因此这里片段截取和前向传播完全一致
					// 截取要计算的梯度dX的片段
					cube slice = X_padded[n](span(h* stride, h* stride + H_conv - 1),
											 span(w* stride, w* stride + W_conv - 1),
											 span::all);
					// dX，使用每一个输入梯度乘以对应的卷积核得到
					dX_padded[n](span(h* stride, h* stride + H_conv - 1),
								 span(w* stride, w* stride + W_conv - 1),
								 span::all) += (*d_input)[n](h, w, c) * (*cache[1])[c];
					// 对应卷积核的dW，使用输入特征片段X乘以输入梯度dY得到
					(*grads[1])[c] += (*d_input)[n](h, w, c) * slice / N_d;
					// 对应卷积核的db，也就是输入梯度dY
					(*grads[2])[c](0, 0, 0) += (*d_input)[n](h, w, c) / N_d;
				}
			}
		}
	}

	// 去掉dX的padding并存入grads[0]
	*grads[0] = dX_padded.de_pad(pad_size);

	// 测试用(使用的卷积核个数为3，卷积核宽高为2，通道为1)
	// 第一个样本的输入梯度
	//(*d_input)[0].slice(0).print("d_input = ");
	//(*d_input)[0].slice(1).print("d_input = ");
	//(*d_input)[0].slice(2).print("d_input = ");
	// 三个卷积核
	//(*cache[1])[0].slice(0).print("W1 = ");
	//(*cache[1])[1].slice(0).print("W2 = ");
	//(*cache[1])[2].slice(0).print("W3 = ");
	// 第一个样本的输出梯度
	//dX_padded[0].slice(0).print("dX_padded = ");
}


void ReLULayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	cout << "initiating ReLULayer" << endl;
}


void ReLULayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	out_shape.assign(in_shape.begin(), in_shape.end()); // deep copy
}


void ReLULayer::forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output,
							 const LayerParams& curr_params, string mode) {
	//cout << "ReLULayer::forward_prop" << endl;
	if(output) {
		output.reset();
	}
	double val = clip_val;

	output.reset(new Blob(*input[0]));
	// ReLU: 大于0不变，小于0清0
	int N = output->get_N();
	for(int i = 0; i < N; i++) {
		(*output)[i].transform([](double e) { return e > 0.0 ? e : 0.0; });
		if(clip) {
			(*output)[i].transform([val](double e) { return e > val ? val : e; });
		}
	}
	

	// 可视化ReLU第一个输入cube
	//vector<cv::Mat_<double>> mat_vec_r_i1;
	//visualize((*input[0])[0], mat_vec_r_i1);
	// 可视化ReLU后第一个输出cube
	//vector<cv::Mat_<double>> mat_vec_r_o1;
	//visualize((*output)[0], mat_vec_r_o1);
}


void ReLULayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
							  shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	//cout << "ReLULayer::backward_prop" << endl;
	// ReLU层梯度dX，直接初始化为输入X以方便计算
	grads[0].reset(new Blob(*cache[0]));

	// ReLU使用当前层输入X得到梯度，大于0为1，否则为0
	// 因此此处将利用输入获得掩码cube，并直接写入grads[0]
	// 遍历所有样本
	int N = grads[0]->get_N();
	double val = clip_val;

	if(clip) {
		for(int n = 0; n < N; n++) {
			(*grads[0])[n].transform([val](double e) { return (e > 0 && e < val) ? 1 : 0; });
		}
	}
	else {
		for(int n = 0; n < N; n++) {
			(*grads[0])[n].transform([](double e) { return e > 0 ? 1 : 0; });
		}
	}

	// 将掩码和输入梯度(dY)相乘获取输出梯度(dX)
	(*grads[0]) = (*grads[0]) * (*d_input);

	// 测试
	//(*d_input)[0].slice(0).print("d_input = ");
	//(*cache[0])[0].slice(0).print("cache = ");
	//(*grads[0])[0].slice(0).print("grads = ");
}


void TanhLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	cout << "initiating TanhLayer" << endl;
}


void TanhLayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	out_shape.assign(in_shape.begin(), in_shape.end());
}


void TanhLayer::forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, 
							 const LayerParams& curr_params, string mode) {
	if(output) {
		output.reset();
	}

	output.reset(new Blob(*input[0]));
	int N = input[0]->get_N();

	for(int n = 0; n < N; n++) {
		cube exp_x = arma::exp((*input[0])[n]);
		cube exp_neg_x = arma::exp(-(*input[0])[n]);
		cube res = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
		forward_cache.push_back(res);
		(*output)[n] = res;
	}

	// 清空forward_cache
	if(mode == "test") {
		forward_cache.clear();
	}
}


void TanhLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, 
							  shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	grads[0].reset(new Blob(*cache[0]));

	int N = grads[0]->get_N();
	assert(forward_cache.size() == N);

	for(int n = 0; n < N; n++) {
		(*grads[0])[n] = (*d_input)[n] % (1 - arma::square(forward_cache[n]));
	}

	// 完成一次迭代后，需要清空forward_cache
	forward_cache.clear();
}


void PoolLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	cout << "initiating PoolLayer" << endl;
}


void PoolLayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	// 输入尺寸
	int N_in = in_shape[0];
	int C_in = in_shape[1];
	int H_in = in_shape[2];
	int W_in = in_shape[3];
	// 池化核尺寸
	int p_H = curr_params.pool_height;
	int p_W = curr_params.pool_width;
	int p_S = curr_params.pool_stride;
	// 输出 
	out_shape[0] = N_in; // 样本数
	out_shape[1] = C_in;
	out_shape[2] = (H_in - p_H) / p_S + 1;
	out_shape[3] = (W_in - p_W) / p_S + 1;
}


void PoolLayer::forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output,
							 const LayerParams& curr_params, string mode) {
	//cout << "PoolLayer::forward_prop" << endl;
	if(output) {
		output.reset();
	}
	// 获取相关尺寸（输入/池化核/输出）
	// 输入尺寸(样本个数/卷积核数/高/宽)
	int N_in = input[0]->get_N();
	int C_in = input[0]->get_C();
	int H_in = input[0]->get_H();
	int W_in = input[0]->get_W();
	// 池化核尺寸和步长
	int H_pool = curr_params.pool_height;
	int W_pool = curr_params.pool_width;
	int stride = curr_params.pool_stride;
	// 输出尺寸
	int H_out = (H_in - H_pool) / stride + 1;
	int W_out = (W_in - W_pool) / stride + 1;

	// 池化后N和C不变
	output.reset(new Blob(N_in, C_in, H_out, W_out));
	// 卷积运算，遍历计算输出的每一个值
	for(int n = 0; n < N_in; n++) {
		for(int c = 0; c < C_in; c++) {
			for(int h = 0; h < H_out; h++) {
				for(int w = 0; w < W_out; w++) {
					// 截取输入片段，注意此处通道要单独计算（因为不需要合并通道）
					cube slice = (*input[0])[n](span(h* stride, h* stride + H_pool - 1),
												span(w* stride, w* stride + W_pool - 1),
												span(c, c));
					// 最大池化
					(*output)[n](h, w, c) = slice.max();
				}
			}
		}
	}
	// 可视化第一个池化层输入
	//vector<cv::Mat_<double>> mat_vec_p_i1;
	//visualize((*input[0])[0], mat_vec_p_i1);
	// 可视化第一个池化层输出
	//vector<cv::Mat_<double>> mat_vec_p_o1;
	//visualize((*output)[0], mat_vec_p_o1);
}


void PoolLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
							  shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	//cout << "PoolLayer::backward_prop" << endl;
	// 池化层梯度尺寸dX(池化层没有W和b)
	grads[0].reset(new Blob(cache[0]->size(), C_ZEROS));

	// 上层输入梯度的尺寸(样本个数，通道数，高，宽)
	int N_d = d_input->get_N();
	int C_d = d_input->get_C();
	int H_d = d_input->get_H();
	int W_d = d_input->get_W();

	// 池化核参数
	int H_pool = curr_params.pool_height;
	int W_pool = curr_params.pool_width;
	int stride = curr_params.pool_stride;

	// 遍历上一层每个输入的梯度值(dY)
	for(int n = 0; n < N_d; n++) {
		for(int c = 0; c < C_d; c++) {
			for(int h = 0; h < H_d; h++) {
				for(int w = 0; w < W_d; w++) {
					// 由于前向计算时也是通过遍历该层的输出(Y)进行计算，因此这里两处片段截取和前向传播完全一致
					// 截取该层前向计算时输入的片段(X)，制作掩码
					mat X_slice = (*cache[0])[n](span(h* stride, h* stride + H_pool - 1),
												 span(w* stride, w* stride + W_pool - 1),
												 span(c, c));
					double max_value = X_slice.max();
					mat mask = conv_to<mat>::from(max_value == X_slice); // 返回掩码矩阵
					// 计算梯度，并将结果写入该层待输出的梯度片段(dX)
					(*grads[0])[n](span(h* stride, h* stride + H_pool - 1),
								   span(w* stride, w* stride + W_pool - 1),
								   span(c, c)) += mask * (*d_input)[n](h, w, c);
				}
			}
		}
	}
	// 测试
	//(*d_input)[0].slice(0).print("d_input = ");
	//(*cache[0])[0].slice(0).print("cache = ");
	//(*grads[0])[0].slice(0).print("grads = ");
}


void FCLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	cout << "initiating FCLayer" << endl;
	// 全连接层中，将W和b使用四维的Blob表示
	// 那么W的形状就是神经元数和输入的C, H, W连接
	// b的形状第一维是神经元数，剩下全是1
	int N = curr_params.fc_neuron;
	int C = in_shape[1];
	int H = in_shape[2];
	int W = in_shape[3];

	// 初始化Blob
	// W: curr_blobs[1]
	if(!curr_blobs[1]) {
		curr_blobs[1].reset(new Blob(N, C, H, W, C_RANDN));
		if(curr_params.fc_init == "Gaussian") {
			(*curr_blobs[1]) *= 1e-2;
		}
		else if(curr_params.fc_init == "He") {
			(*curr_blobs[1]) *= std::sqrt(2 / static_cast<double>(in_shape[1] * in_shape[2] * in_shape[3]));
		}
		else if(curr_params.fc_init == "Xavier") {
			(*curr_blobs[1]) *= std::sqrt(1 / static_cast<double>(in_shape[1] * in_shape[2] * in_shape[3]));
		}
		else {
			throw "wrong init method!";
		}
	}
	// b: curr_blobs[2]
	if(!curr_blobs[2]) {
		curr_blobs[2].reset(new Blob(N, 1, 1, 1, C_ZEROS));
	}
}


void FCLayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	// 输出尺寸
	int N_out = in_shape[0];
	int C_out = curr_params.fc_neuron; // 神经元个数
	int H_out = 1;
	int W_out = 1;
	// 输出 
	out_shape[0] = N_out; // 样本数
	out_shape[1] = C_out;
	out_shape[2] = H_out;
	out_shape[3] = W_out;
}


void FCLayer::forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output,
						   const LayerParams& curr_params, string mode) {
	// 测试用
	//vector<shared_ptr<Blob>> input(3, nullptr);
	//input[0].reset(new Blob(2, 2, 2, 2, C_ONES));
	//input[1].reset(new Blob(3, 2, 2, 2, C_RANDU));
	//input[2].reset(new Blob(3, 1, 1, 1, C_RANDU));

	//cout << "FCLayer::forward_prop" << endl;
	if(output) {
		output.reset();
	}
	// 全连接层中，将W和b使用四维的Blob表示
	// 那么W的形状就是神经元数和输入的C, H, W连接，并将神经元称为全连接核
	// 可以理解为就是步长为0，宽高和输入相等的卷积核
	// b的形状第一维是神经元数，剩下全是1
	// X*W就是将每个输入和每个全连接核相乘求和
	// 输出尺寸为 (N, 神经元个数(全连接核数), 1, 1)

	// 确保X和W通道相同
	if(input[0]->get_C() != input[1]->get_C()) {
		throw "input channels don't match W's channels!";
	}

	// 获取相关尺寸
	// 输入尺寸
	int N_in = input[0]->get_N();
	int C_in = input[0]->get_C();
	int H_in = input[0]->get_H();
	int W_in = input[0]->get_W();
	// 全连接核数量和尺寸
	int N_fc = input[1]->get_N();
	int H_fc = input[1]->get_H();
	int W_fc = input[1]->get_W();

	// 确保全连接核宽高与输入一致
	if(H_in != H_fc || W_in != W_fc) {
		throw "input size don't match W's size!";
	}

	// 输出尺寸
	int H_out = 1;
	int W_out = 1;

	// 卷积后通道数就是卷积核个数
	output.reset(new Blob(N_in, N_fc, H_out, W_out));

	// 全连接运算，遍历计算输出的每一个值
	// 宽和高都是1，因此无需遍历
	for(int n = 0; n < N_in; n++) {
		for(int c = 0; c < N_fc; c++) {
			// 无需截取，直接对整个cube进行计算
			cube slice = (*input[0])[n];
			(*output)[n](0, 0, c) = accu(slice % (*input[1])[c]) + as_scalar((*input[2])[c]);
		}
	}

	// 测试
	// 可视化第一个全连接层输入
	//vector<cv::Mat_<double>> mat_vec_f_i1;
	//visualize((*input[0])[0], mat_vec_f_i1);
	// 可视化第一个全连接核
	//vector<cv::Mat_<double>> mat_vec_f_w1;
	//visualize((*input[1])[0], mat_vec_f_w1);
	// 可视化第一个偏置b
	//vector<cv::Mat_<double>> mat_vec_f_b1;
	//visualize((*input[2])[0], mat_vec_f_b1);
	// 可视化第一个全连接层输出
	//vector<cv::Mat_<double>> mat_vec_f_o1;
	//visualize((*output)[0], mat_vec_f_o1);
}


void FCLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
							shared_ptr<Blob>& d_input, const LayerParams& curr_params) {

	// 测试用
	//shared_ptr<Blob> d_input(new Blob(2, 2, 1, 1, C_RANDU));
	//vector<shared_ptr<Blob>> cache(3, nullptr);
	//cache[0].reset(new Blob(2, 2, 2, 2, C_ONES));
	//cache[1].reset(new Blob(2, 2, 2, 2, C_RANDU));
	//cache[2].reset(new Blob(2, 1, 1, 1, C_RANDU));

	//cout << "FCLayer::backward_prop" << endl;
	// 初始化grads
	grads[0].reset(new Blob(cache[0]->size(), C_ZEROS));
	grads[1].reset(new Blob(cache[1]->size(), C_ZEROS));
	grads[2].reset(new Blob(cache[2]->size(), C_ZEROS));

	// 样本数N和神经元数C
	int N = grads[0]->get_N();
	int C = grads[1]->get_N();

	// 遍历每个样本和神经元，输入梯度形状是(样本数，神经元数, 1, 1)
	// dX形状是(样本数，通道数，高，宽)
	// dW形状是(神经元数，通道数，高，宽)
	// db形状是(神经元数，1，1，1)
	for(int n = 0; n < N; n++) {
		for(int c = 0; c < C; c++) {
			// 第n个样本的dX，注意是数值*cube得到cube，要累加所有神经元的梯度
			(*grads[0])[n] += (*d_input)[n](0, 0, c) * (*cache[1])[c];
			// dW，仍然是数值*cube，要累加所有样本的梯度并取平均
			(*grads[1])[c] += (*d_input)[n](0, 0, c) * (*cache[0])[n] / N;
			// db
			(*grads[2])[c] += (*d_input)[n](0, 0, c) / N;
		}
	}

	// 测试
	// 可视化前两个样本传入的梯度
	//vector<cv::Mat_<double>> vec_mat_in0;
	//visualize((*d_input)[0], vec_mat_in0);
	//vector<cv::Mat_<double>> vec_mat_in1;
	//visualize((*d_input)[1], vec_mat_in1);
	// 可视化两个输入的样本
	//vector<cv::Mat_<double>> vec_mat_x0;
	//visualize((*cache[0])[0], vec_mat_x0);
	//vector<cv::Mat_<double>> vec_mat_x1;
	//visualize((*cache[0])[1], vec_mat_x1);
	// 可视化两个神经元(全连接核)
	//vector<cv::Mat_<double>> vec_mat_w0;
	//visualize((*cache[1])[0], vec_mat_w0);
	//vector<cv::Mat_<double>> vec_mat_w1;
	//visualize((*cache[1])[1], vec_mat_w1);
	// 可视化第一个样本的dX
	//vector<cv::Mat_<double>> vec_mat_dx0;
	//visualize((*grads[0])[0], vec_mat_dx0);
	// 可视化第一个神经元的dW
	//vector<cv::Mat_<double>> vec_mat_dw0;
	//visualize((*grads[1])[0], vec_mat_dw0);
	// 可视化第一个神经元的db
	//vector<cv::Mat_<double>> vec_mat_db0;
	//visualize((*grads[2])[0], vec_mat_db0);
}


void DropoutLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	cout << "initiating DropoutLayer" << endl;
}


void DropoutLayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	out_shape.assign(in_shape.begin(), in_shape.end()); // deep copy
}


void DropoutLayer::forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output,
								const LayerParams& curr_params, string mode) {
	//cout << "DropoutLayer::forward_prop" << endl;
	if(output) {
		output.reset();
	}
	double dropout_rate = curr_params.dropout_rate;
	if(!(dropout_rate >= 0 && dropout_rate < 1)) {
		throw "dropout rate is wrong!";
	}

	if(mode == "train") {
		// 获取掩码
		mask.reset(new Blob(input[0]->size(), C_RANDU));
		int N = input[0]->get_N();
		for(int i = 0; i < N; i++) {
			(*mask)[i].transform([dropout_rate](double e) { return e < dropout_rate ? 0 : 1; });
		}
		// dropout，注意需要对剩余的非0数据进行缩放，保证期望不变
		output.reset(new Blob((*input[0]) * (*mask) / (1 - dropout_rate)));
	}
	else if(mode == "test") { // 测试模式中不能使用dropout，输出就是输入
		output.reset(new Blob(*input[0]));
	}
}


void DropoutLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
								 shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	double dropout_rate = curr_params.dropout_rate;
	grads[0].reset(new Blob((*d_input) * (*mask) / (1 - dropout_rate)));
}


// before的形状必须为(1, 1, C)，在每个通道上广播到(H, W, C)
inline void broadcast(cube& after, cube& before, int C) {
	// slice用于从cube中返回matrix
	for(int c = 0; c < C; c++) {
		after.slice(c).fill(as_scalar(before.slice(c)));
	}
}


// 将item在宽和高维度上的元素相加，形状从(H, W, C)变为(1, 1, C)
inline cube de_broadcast(cube& item) {
	return sum(sum(item, 0), 1);
}


void BNLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params)
{
	cout << "initiating BNLayer" << endl;

	int C = in_shape[1];
	int H = in_shape[2];
	int W = in_shape[3];

	// 使用其他层存放W和b的空间存放动态更新的均值和标准差
	if(!curr_blobs[1]) {
		curr_blobs[1].reset(new Blob(1, C, H, W, C_ZEROS));
	}
	if(!curr_blobs[2]) {
		curr_blobs[2].reset(new Blob(1, C, H, W, C_ZEROS));
	}
}


void BNLayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	out_shape.assign(in_shape.begin(), in_shape.end());
}


void BNLayer::forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, 
						   const LayerParams& curr_params, string mode) {
	if(output) {
		output.reset();
	}

	// 输入和输出尺寸相同
	output.reset(new Blob(input[0]->size(), C_ZEROS));

	int N = input[0]->get_N();
	int C = input[0]->get_C();
	int H = input[0]->get_H();
	int W = input[0]->get_W();

	if(mode == "train") {
		// 负均值/方差/标准差
		minus_mean_.reset(new cube(1, 1, C, fill::zeros));
		var_.reset(new cube(1, 1, C, fill::zeros));
		std_.reset(new cube(1, 1, C, fill::zeros));
		
		// 如果是卷积层要先在每个通道上对宽高求均值，使其形状变为(N, C, 1, 1)
		Blob averaged(N, C, 1, 1, C_ZEROS);
		for(int i = 0; i < N; i++) {
			averaged[i] = de_broadcast((*input[0])[i]) / (H * W);
		}

		// 该批次的负均值，
		for(int i = 0; i < N; i++) {
			(*minus_mean_) += averaged[i];
		}
		(*minus_mean_) /= (-N);

		// 该批次的方差
		for(int i = 0; i < N; i++) {
			(*var_) += square(averaged[i] + (*minus_mean_));
		}
		(*var_) /= N;

		// 该批次的标准差
		(*std_) = sqrt((*var_) + epsilon);

		// 广播均值和标准差
		cube minus_mean_bc(H, W, C, fill::zeros);
		cube std_bc(H, W, C, fill::zeros);
		broadcast(minus_mean_bc, *minus_mean_, C);
		broadcast(std_bc, *std_, C);

		// 归一化
		for(int i = 0; i < N; i++) {
			(*output)[i] = ((*input[0])[i] + minus_mean_bc) / std_bc;
		}
			
		// 用第一个批次的均值和标准差初始化动态更新的均值和标准差
		// 这样刚开始几个迭代周期验证的时候有一个相对准确的均值和标准差
		if(!mean_std_initiated) {
			(*input[1])[0] = minus_mean_bc;
			(*input[2])[0] = std_bc;
			mean_std_initiated = true;
		}

		// 动态更新均值和方差
		(*input[1])[0] = eta * (*input[1])[0] + (1 - eta) * minus_mean_bc;
		(*input[2])[0] = eta * (*input[2])[0] + (1 - eta) * std_bc;
	}
	// 测试阶段使用保存的动态均值和标准差进行归一化
	else if(mode == "test") {
		for(int n = 0; n < N; n++) {
			(*output)[n] = ((*input[0])[n] + (*input[1])[0]) / (*input[2])[0];
		}
	}
}


void BNLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, 
							shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	// 梯度尺寸
	grads[0].reset(new Blob(cache[0]->size(), C_ZEROS));
	int N = grads[0]->get_N();
	int C = grads[0]->get_C();
	int H = grads[0]->get_H();
	int W = grads[0]->get_W();

	// 广播均值和标准差
	cube minus_mean_bc(H, W, C, fill::zeros);
	cube std_bc(H, W, C, fill::zeros);
	broadcast(minus_mean_bc, *minus_mean_, C);
	broadcast(std_bc, *std_, C);

	// 使用从计算图得到的公式进行反向传播，item1-item6为中间结果
	cube item1(H, W, C, fill::zeros);
	for(int i = 0; i < N; i++) {
		item1 += (*d_input)[i] % ((*cache[0])[i] + minus_mean_bc);
	}

	cube item2 = -de_broadcast(item1) / (2 * (*var_) % (*std_)) / N;

	cube item3(1, 1, C, fill::zeros);
	vector<cube> grads_cache(N, cube(1, 1, C));

	for(int i = 0; i < N; i++) {
		grads_cache[i] = item2 % (2 * (de_broadcast((*cache[0])[i]) / (H * W) + (*minus_mean_)));
		item3 += grads_cache[i];
	}

	cube item4(H, W, C, fill::zeros);
	for(int i = 0; i < N; i++) {
		item4 += (*d_input)[i] / std_bc;
	}

	cube item5(1, 1, C, fill::zeros);
	item5 = de_broadcast(item4);

	cube item6 = (item3 + item5) / (-N);

	cube grads_pt1(H, W, C, fill::zeros);
	broadcast(grads_pt1, item6, C);
	
	// 每个样本的梯度值，最终梯度由3个部分组成
	for(int n = 0; n < N; n++) {
		cube grads_pt2(H, W, C, fill::zeros);
		cube grads_pt3 = (*d_input)[n] / std_bc;
		broadcast(grads_pt2, grads_cache[n], C);
		(*grads[0])[n] = (grads_pt1 + grads_pt2) / (H * W) + grads_pt3;
	}
}


void ScaleLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	// 通道数
	int C = in_shape[1];

	// 初始化gamma和beta(也就是W和b)
	if(!curr_blobs[1]) {
		curr_blobs[1].reset(new Blob(1, C, 1, 1, C_ONES));
	}
	if(!curr_blobs[2]) {
		curr_blobs[2].reset(new Blob(1, C, 1, 1, C_ONES));
	}
}


void ScaleLayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	out_shape.assign(in_shape.begin(), in_shape.end());
}


void ScaleLayer::forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, 
							  const LayerParams& curr_params, string mode) {
	output.reset(new Blob(input[0]->size(), C_ZEROS));

	int N = input[0]->get_N();
	int C = input[0]->get_C();
	int H = input[0]->get_H();
	int W = input[0]->get_W();

	// 广播gamma和beta
	cube gamma(H, W, C, fill::zeros);
	cube beta(H, W, C, fill::zeros);
	broadcast(gamma, (*input[1])[0], C);
	broadcast(beta, (*input[2])[0], C);

	// 对输入特征做缩放和平移
	for(int n = 0; n < N; n++) {
		(*output)[n] = gamma % (*input[0])[n] + beta;
	}
}


void ScaleLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, 
							   shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	grads[0].reset(new Blob(cache[0]->size(), C_ZEROS)); 
	grads[1].reset(new Blob(cache[1]->size(), C_ZEROS));
	grads[2].reset(new Blob(cache[2]->size(), C_ZEROS));
	int N = grads[0]->get_N();
	int C = grads[0]->get_C();
	int H = grads[0]->get_H();
	int W = grads[0]->get_W();

	// 广播gamma
	cube gamma(H, W, C, fill::zeros);
	broadcast(gamma, (*cache[1])[0], C);

	cube d_gamma(H, W, C, fill::zeros);
	cube d_beta(H, W, C, fill::zeros);
	for(int n = 0; n < N; ++n) {
		(*grads[0])[n] = (*d_input)[n] % gamma; // dX
		d_gamma += (*d_input)[n] % (*cache[0])[n]; 
		d_beta += (*d_input)[n];
	}
	(*grads[1])[0] = de_broadcast(d_gamma) / N; // d_gamma
	(*grads[2])[0] = de_broadcast(d_beta) / N; // d_beta
}


double CrossEntropyLossLayer::softmax_cross_entropy_with_logits(shared_ptr<Blob>& input, shared_ptr<Blob>& Y, shared_ptr<Blob>& d_output) {
	//cout << "LossLayer::softmax_cross_entropy_with_logits" << endl;
	if(d_output) {
		d_output.reset();
	}

	// 获取相关尺寸
	// (样本数N, 类别数量(上一层的神经元个数), 1, 1)
	int N_in = input->get_N();
	int C_in = input->get_C();
	int H_in = input->get_H();
	int W_in = input->get_W();

	if(H_in != 1 || W_in != 1) {
		throw "input H and input W must be 1!";
	}

	d_output.reset(new Blob(N_in, C_in, H_in, W_in));  // 当前层的梯度

	// 计算每个样本的loss
	double total_loss = 0.0;
	for(int n = 0; n < N_in; n++) {
		// 对当前样本进行softmax, 类型是cube / number
		cube prob = arma::exp((*input)[n]) / accu(arma::exp((*input)[n]));
		total_loss += (-accu((*Y)[n] % arma::log(prob)));

		// 当前样本梯度
		(*d_output)[n] = prob - (*Y)[n];
	}

	return total_loss / N_in;
}


double HingeLossLayer::hinge(shared_ptr<Blob>& input, shared_ptr<Blob>& Y, shared_ptr<Blob>& d_output, double delta) {
	if(d_output) {
		d_output.reset();
	}
	// 获取相关尺寸
	// (样本数N, 类别数量(上一层的神经元个数), 1, 1)
	int N_in = input->get_N();
	int C_in = input->get_C();
	int H_in = input->get_H();
	int W_in = input->get_W();

	if(H_in != 1 || W_in != 1) {
		throw "input H and input W must be 1!";
	}

	d_output.reset(new Blob(N_in, C_in, H_in, W_in));  // 当前层的梯度

	double total_loss = 0.0;
	for(int n = 0; n < N_in; n++) {
		// loss
		int correct_idx = (*Y)[n].index_max();  // 当前样本正确类别序号
		double correct_score = (*input)[n](0, 0, correct_idx); // 正确类别预测得分
		// 每个类别(不包括正确类别)的错误得分减去正确得分加上delta
		cube score_cube = (*input)[n] - correct_score + delta;
		score_cube(0, 0, correct_idx) = 0;
		// 计算每个类别的损失
		score_cube.transform([](double e){ return e > 0 ? e : 0; });
		total_loss += arma::accu(score_cube);

		// 梯度，错误类别的梯度就是max操作时的掩码值，正确类别的梯度是错误类别梯度之和乘上-1
		score_cube.transform([](double e){ return e ? 1 : 0; });
		score_cube(0, 0, correct_idx) = -arma::accu(score_cube);
		(*d_output)[n] = score_cube;
	}

	return total_loss / N_in;
}