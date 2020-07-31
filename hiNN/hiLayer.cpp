#include "hiLayer.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace arma;


// ��arma::MatתΪcv::Mat
template<typename T>
void arma2cv(const arma::Mat<T>& arma_mat, cv::Mat_<T>& cv_mat) {
	cv::transpose(cv::Mat_<T>(static_cast<int>(arma_mat.n_cols),
				  static_cast<int>(arma_mat.n_rows),
			      const_cast<T*>(arma_mat.memptr())), cv_mat);
}


// ��cube������MatתΪcv�е�Mat������vector
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
	//  ����˵�N, C, H, W
	int N = curr_params.conv_num;
	int C = in_shape[1];
	int H = curr_params.conv_height;
	int W = curr_params.conv_width;

	// ��ʼ��Blob
	// W: curr_blobs[1]
	if(!curr_blobs[1]) {
		curr_blobs[1].reset(new Blob(N, C, H, W, C_RANDN));
		if(curr_params.conv_init == "Gaussian") {
			// ����׼���Ϊ0.01
			(*curr_blobs[1]) *= 1e-2;
		}
		else if(curr_params.conv_init == "He") { // He��ʼ��
			(*curr_blobs[1]) *= std::sqrt(2 / static_cast<double>(in_shape[1] * in_shape[2] * in_shape[3]));
		}
		else if(curr_params.conv_init == "Xavier") { // Xavier��ʼ��
			(*curr_blobs[1]) *= std::sqrt(1 / static_cast<double>(in_shape[1] * in_shape[2] * in_shape[3]));
		}
		else {
			throw "wrong init method!";
		}
	}
	// b: curr_blobs[2]��ÿ�������ֻ��һ��b
	if(!curr_blobs[2]) {
		curr_blobs[2].reset(new Blob(N, 1, 1, 1, C_RANDN));
		(*curr_blobs[2]) *= 1e-2;
	}
}


void ConvLayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	// ����ߴ�
	int N_in = in_shape[0];
	int C_in = in_shape[1];
	int H_in = in_shape[2];
	int W_in = in_shape[3];
	// ����˳ߴ�
	int c_N = curr_params.conv_num;
	int c_H = curr_params.conv_height;
	int c_W = curr_params.conv_width;
	int c_P = curr_params.conv_pad;
	int c_S = curr_params.conv_stride;
	// ��� 
	out_shape[0] = N_in; // ������
	out_shape[1] = c_N; // ���ͨ����Ϊ����˸���
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
	// ��ȡ��سߴ磨����/�����/�����
	// ����ߴ�(��������/����ͨ����/��/��)
	int N_in = input[0]->get_N();
	int C_in = input[0]->get_C();
	int H_in = input[0]->get_H();
	int W_in = input[0]->get_W();
	// ����˳ߴ磬�����ͨ���� == ����ͨ���������ʡ��
	int N_conv = input[1]->get_N();
	int H_conv = input[1]->get_H();
	int W_conv = input[1]->get_W();
	int stride = curr_params.conv_stride;
	int pad_size = curr_params.conv_pad;
	// ����ߴ�
	int H_out = (H_in + 2 * pad_size - H_conv) / stride + 1;
	int W_out = (W_in + 2 * pad_size - W_conv) / stride + 1;

	// padding
	Blob X_padded = input[0]->pad(pad_size);

	// ���ӻ���һ��ͼƬ
	//arma::mat mat0_arma = X_padded[0].slice(0);
	//cv::Mat_<double> mat0_cv;
	//arma2cv(mat0_arma, mat0_cv);

	// �����ͨ�������Ǿ���˸���
	output.reset(new Blob(N_in, N_conv, H_out, W_out));
	// ������㣬�������������ÿһ��ֵ
	for(int n = 0; n < N_in; n++) {
		for(int c = 0; c < N_conv; c++) {
			for(int h = 0; h < H_out; h++) {
				for(int w = 0; w < W_out; w++) {
					// ��ȡpadding������룬3��span�ֱ�Ϊ��ֹrow/col/slice(channel)���˴����Ҿ�Ϊ�պ�����
					cube slice = X_padded[n](span(h* stride, h* stride + H_conv - 1),
											 span(w* stride, w* stride + W_conv - 1),
											 span::all);
					// ���, sum(slice * conv_core) + b
					// ��Ϊ���ͨ�������Ǿ���˸���������Ҫ�������о���ˣ��������Ϊc
					// b: (c, 1, 1, 1) c=����˸���
					(*output)[n](h, w, c) = accu(slice % (*input[1])[c]) + as_scalar((*input[2])[c]);
				}
			}
		}
	}

	// ����
	// ���ӻ���һ�������
	//vector<cv::Mat_<double>> mat_vec_c_w1;
	//visualize((*input[1])[0], mat_vec_c_w1);
	// ���ӻ���һ��ƫ��b
	//vector<cv::Mat_<double>> mat_vec_c_b1;
	//visualize((*input[2])[0], mat_vec_c_b1);
	// ���ӻ���һ������˵����
	//vector<cv::Mat_<double>> mat_vec_c_o1;
	//visualize((*output)[0], mat_vec_c_o1);
}


void ConvLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
							  shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	//cout << "ConvLayer::backward_prop" << endl;
	// ������ݶȳߴ�
	grads[0].reset(new Blob(cache[0]->size(), C_ZEROS));
	grads[1].reset(new Blob(cache[1]->size(), C_ZEROS));
	grads[2].reset(new Blob(cache[2]->size(), C_ZEROS));

	// �ϲ������ݶȵĳߴ�(��������������˸������ߣ���)
	int N_d = d_input->get_N();
	int C_d = d_input->get_C();
	int H_d = d_input->get_H();
	int W_d = d_input->get_W();

	// ����˲���
	int H_conv = curr_params.conv_height;
	int W_conv = curr_params.conv_width;
	int stride = curr_params.conv_stride;
	int pad_size = curr_params.conv_pad;

	// ����ʵ�ʷ��򴫲������Ӧ����padding�������X
	// dXҪ��X�ߴ�һ�£����dXҲҪ����padding
	Blob X_padded = cache[0]->pad(pad_size);
	Blob dX_padded(X_padded.size(), C_ZEROS);

	// ������һ��ÿ��������ݶ�ֵ(dY)
	for(int n = 0; n < N_d; n++) {
		for(int c = 0; c < C_d; c++) {
			for(int h = 0; h < H_d; h++) {
				for(int w = 0; w < W_d; w++) {
					// ����ǰ�����ʱҲ��ͨ�������ò�����(Y)���м��㣬�������Ƭ�ν�ȡ��ǰ�򴫲���ȫһ��
					// ��ȡҪ������ݶ�dX��Ƭ��
					cube slice = X_padded[n](span(h* stride, h* stride + H_conv - 1),
											 span(w* stride, w* stride + W_conv - 1),
											 span::all);
					// dX��ʹ��ÿһ�������ݶȳ��Զ�Ӧ�ľ���˵õ�
					dX_padded[n](span(h* stride, h* stride + H_conv - 1),
								 span(w* stride, w* stride + W_conv - 1),
								 span::all) += (*d_input)[n](h, w, c) * (*cache[1])[c];
					// ��Ӧ����˵�dW��ʹ����������Ƭ��X���������ݶ�dY�õ�
					(*grads[1])[c] += (*d_input)[n](h, w, c) * slice / N_d;
					// ��Ӧ����˵�db��Ҳ���������ݶ�dY
					(*grads[2])[c](0, 0, 0) += (*d_input)[n](h, w, c) / N_d;
				}
			}
		}
	}

	// ȥ��dX��padding������grads[0]
	*grads[0] = dX_padded.de_pad(pad_size);

	// ������(ʹ�õľ���˸���Ϊ3������˿��Ϊ2��ͨ��Ϊ1)
	// ��һ�������������ݶ�
	//(*d_input)[0].slice(0).print("d_input = ");
	//(*d_input)[0].slice(1).print("d_input = ");
	//(*d_input)[0].slice(2).print("d_input = ");
	// ���������
	//(*cache[1])[0].slice(0).print("W1 = ");
	//(*cache[1])[1].slice(0).print("W2 = ");
	//(*cache[1])[2].slice(0).print("W3 = ");
	// ��һ������������ݶ�
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
	// ReLU: ����0���䣬С��0��0
	int N = output->get_N();
	for(int i = 0; i < N; i++) {
		(*output)[i].transform([](double e) { return e > 0.0 ? e : 0.0; });
		if(clip) {
			(*output)[i].transform([val](double e) { return e > val ? val : e; });
		}
	}
	

	// ���ӻ�ReLU��һ������cube
	//vector<cv::Mat_<double>> mat_vec_r_i1;
	//visualize((*input[0])[0], mat_vec_r_i1);
	// ���ӻ�ReLU���һ�����cube
	//vector<cv::Mat_<double>> mat_vec_r_o1;
	//visualize((*output)[0], mat_vec_r_o1);
}


void ReLULayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
							  shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	//cout << "ReLULayer::backward_prop" << endl;
	// ReLU���ݶ�dX��ֱ�ӳ�ʼ��Ϊ����X�Է������
	grads[0].reset(new Blob(*cache[0]));

	// ReLUʹ�õ�ǰ������X�õ��ݶȣ�����0Ϊ1������Ϊ0
	// ��˴˴�����������������cube����ֱ��д��grads[0]
	// ������������
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

	// ������������ݶ�(dY)��˻�ȡ����ݶ�(dX)
	(*grads[0]) = (*grads[0]) * (*d_input);

	// ����
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

	// ���forward_cache
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

	// ���һ�ε�������Ҫ���forward_cache
	forward_cache.clear();
}


void PoolLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	cout << "initiating PoolLayer" << endl;
}


void PoolLayer::nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) {
	// ����ߴ�
	int N_in = in_shape[0];
	int C_in = in_shape[1];
	int H_in = in_shape[2];
	int W_in = in_shape[3];
	// �ػ��˳ߴ�
	int p_H = curr_params.pool_height;
	int p_W = curr_params.pool_width;
	int p_S = curr_params.pool_stride;
	// ��� 
	out_shape[0] = N_in; // ������
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
	// ��ȡ��سߴ磨����/�ػ���/�����
	// ����ߴ�(��������/�������/��/��)
	int N_in = input[0]->get_N();
	int C_in = input[0]->get_C();
	int H_in = input[0]->get_H();
	int W_in = input[0]->get_W();
	// �ػ��˳ߴ�Ͳ���
	int H_pool = curr_params.pool_height;
	int W_pool = curr_params.pool_width;
	int stride = curr_params.pool_stride;
	// ����ߴ�
	int H_out = (H_in - H_pool) / stride + 1;
	int W_out = (W_in - W_pool) / stride + 1;

	// �ػ���N��C����
	output.reset(new Blob(N_in, C_in, H_out, W_out));
	// ������㣬�������������ÿһ��ֵ
	for(int n = 0; n < N_in; n++) {
		for(int c = 0; c < C_in; c++) {
			for(int h = 0; h < H_out; h++) {
				for(int w = 0; w < W_out; w++) {
					// ��ȡ����Ƭ�Σ�ע��˴�ͨ��Ҫ�������㣨��Ϊ����Ҫ�ϲ�ͨ����
					cube slice = (*input[0])[n](span(h* stride, h* stride + H_pool - 1),
												span(w* stride, w* stride + W_pool - 1),
												span(c, c));
					// ���ػ�
					(*output)[n](h, w, c) = slice.max();
				}
			}
		}
	}
	// ���ӻ���һ���ػ�������
	//vector<cv::Mat_<double>> mat_vec_p_i1;
	//visualize((*input[0])[0], mat_vec_p_i1);
	// ���ӻ���һ���ػ������
	//vector<cv::Mat_<double>> mat_vec_p_o1;
	//visualize((*output)[0], mat_vec_p_o1);
}


void PoolLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
							  shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	//cout << "PoolLayer::backward_prop" << endl;
	// �ػ����ݶȳߴ�dX(�ػ���û��W��b)
	grads[0].reset(new Blob(cache[0]->size(), C_ZEROS));

	// �ϲ������ݶȵĳߴ�(����������ͨ�������ߣ���)
	int N_d = d_input->get_N();
	int C_d = d_input->get_C();
	int H_d = d_input->get_H();
	int W_d = d_input->get_W();

	// �ػ��˲���
	int H_pool = curr_params.pool_height;
	int W_pool = curr_params.pool_width;
	int stride = curr_params.pool_stride;

	// ������һ��ÿ��������ݶ�ֵ(dY)
	for(int n = 0; n < N_d; n++) {
		for(int c = 0; c < C_d; c++) {
			for(int h = 0; h < H_d; h++) {
				for(int w = 0; w < W_d; w++) {
					// ����ǰ�����ʱҲ��ͨ�������ò�����(Y)���м��㣬�����������Ƭ�ν�ȡ��ǰ�򴫲���ȫһ��
					// ��ȡ�ò�ǰ�����ʱ�����Ƭ��(X)����������
					mat X_slice = (*cache[0])[n](span(h* stride, h* stride + H_pool - 1),
												 span(w* stride, w* stride + W_pool - 1),
												 span(c, c));
					double max_value = X_slice.max();
					mat mask = conv_to<mat>::from(max_value == X_slice); // �����������
					// �����ݶȣ��������д��ò��������ݶ�Ƭ��(dX)
					(*grads[0])[n](span(h* stride, h* stride + H_pool - 1),
								   span(w* stride, w* stride + W_pool - 1),
								   span(c, c)) += mask * (*d_input)[n](h, w, c);
				}
			}
		}
	}
	// ����
	//(*d_input)[0].slice(0).print("d_input = ");
	//(*cache[0])[0].slice(0).print("cache = ");
	//(*grads[0])[0].slice(0).print("grads = ");
}


void FCLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	cout << "initiating FCLayer" << endl;
	// ȫ���Ӳ��У���W��bʹ����ά��Blob��ʾ
	// ��ôW����״������Ԫ���������C, H, W����
	// b����״��һά����Ԫ����ʣ��ȫ��1
	int N = curr_params.fc_neuron;
	int C = in_shape[1];
	int H = in_shape[2];
	int W = in_shape[3];

	// ��ʼ��Blob
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
	// ����ߴ�
	int N_out = in_shape[0];
	int C_out = curr_params.fc_neuron; // ��Ԫ����
	int H_out = 1;
	int W_out = 1;
	// ��� 
	out_shape[0] = N_out; // ������
	out_shape[1] = C_out;
	out_shape[2] = H_out;
	out_shape[3] = W_out;
}


void FCLayer::forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output,
						   const LayerParams& curr_params, string mode) {
	// ������
	//vector<shared_ptr<Blob>> input(3, nullptr);
	//input[0].reset(new Blob(2, 2, 2, 2, C_ONES));
	//input[1].reset(new Blob(3, 2, 2, 2, C_RANDU));
	//input[2].reset(new Blob(3, 1, 1, 1, C_RANDU));

	//cout << "FCLayer::forward_prop" << endl;
	if(output) {
		output.reset();
	}
	// ȫ���Ӳ��У���W��bʹ����ά��Blob��ʾ
	// ��ôW����״������Ԫ���������C, H, W���ӣ�������Ԫ��Ϊȫ���Ӻ�
	// �������Ϊ���ǲ���Ϊ0����ߺ�������ȵľ����
	// b����״��һά����Ԫ����ʣ��ȫ��1
	// X*W���ǽ�ÿ�������ÿ��ȫ���Ӻ�������
	// ����ߴ�Ϊ (N, ��Ԫ����(ȫ���Ӻ���), 1, 1)

	// ȷ��X��Wͨ����ͬ
	if(input[0]->get_C() != input[1]->get_C()) {
		throw "input channels don't match W's channels!";
	}

	// ��ȡ��سߴ�
	// ����ߴ�
	int N_in = input[0]->get_N();
	int C_in = input[0]->get_C();
	int H_in = input[0]->get_H();
	int W_in = input[0]->get_W();
	// ȫ���Ӻ������ͳߴ�
	int N_fc = input[1]->get_N();
	int H_fc = input[1]->get_H();
	int W_fc = input[1]->get_W();

	// ȷ��ȫ���Ӻ˿��������һ��
	if(H_in != H_fc || W_in != W_fc) {
		throw "input size don't match W's size!";
	}

	// ����ߴ�
	int H_out = 1;
	int W_out = 1;

	// �����ͨ�������Ǿ���˸���
	output.reset(new Blob(N_in, N_fc, H_out, W_out));

	// ȫ�������㣬�������������ÿһ��ֵ
	// ��͸߶���1������������
	for(int n = 0; n < N_in; n++) {
		for(int c = 0; c < N_fc; c++) {
			// �����ȡ��ֱ�Ӷ�����cube���м���
			cube slice = (*input[0])[n];
			(*output)[n](0, 0, c) = accu(slice % (*input[1])[c]) + as_scalar((*input[2])[c]);
		}
	}

	// ����
	// ���ӻ���һ��ȫ���Ӳ�����
	//vector<cv::Mat_<double>> mat_vec_f_i1;
	//visualize((*input[0])[0], mat_vec_f_i1);
	// ���ӻ���һ��ȫ���Ӻ�
	//vector<cv::Mat_<double>> mat_vec_f_w1;
	//visualize((*input[1])[0], mat_vec_f_w1);
	// ���ӻ���һ��ƫ��b
	//vector<cv::Mat_<double>> mat_vec_f_b1;
	//visualize((*input[2])[0], mat_vec_f_b1);
	// ���ӻ���һ��ȫ���Ӳ����
	//vector<cv::Mat_<double>> mat_vec_f_o1;
	//visualize((*output)[0], mat_vec_f_o1);
}


void FCLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
							shared_ptr<Blob>& d_input, const LayerParams& curr_params) {

	// ������
	//shared_ptr<Blob> d_input(new Blob(2, 2, 1, 1, C_RANDU));
	//vector<shared_ptr<Blob>> cache(3, nullptr);
	//cache[0].reset(new Blob(2, 2, 2, 2, C_ONES));
	//cache[1].reset(new Blob(2, 2, 2, 2, C_RANDU));
	//cache[2].reset(new Blob(2, 1, 1, 1, C_RANDU));

	//cout << "FCLayer::backward_prop" << endl;
	// ��ʼ��grads
	grads[0].reset(new Blob(cache[0]->size(), C_ZEROS));
	grads[1].reset(new Blob(cache[1]->size(), C_ZEROS));
	grads[2].reset(new Blob(cache[2]->size(), C_ZEROS));

	// ������N����Ԫ��C
	int N = grads[0]->get_N();
	int C = grads[1]->get_N();

	// ����ÿ����������Ԫ�������ݶ���״��(����������Ԫ��, 1, 1)
	// dX��״��(��������ͨ�������ߣ���)
	// dW��״��(��Ԫ����ͨ�������ߣ���)
	// db��״��(��Ԫ����1��1��1)
	for(int n = 0; n < N; n++) {
		for(int c = 0; c < C; c++) {
			// ��n��������dX��ע������ֵ*cube�õ�cube��Ҫ�ۼ�������Ԫ���ݶ�
			(*grads[0])[n] += (*d_input)[n](0, 0, c) * (*cache[1])[c];
			// dW����Ȼ����ֵ*cube��Ҫ�ۼ������������ݶȲ�ȡƽ��
			(*grads[1])[c] += (*d_input)[n](0, 0, c) * (*cache[0])[n] / N;
			// db
			(*grads[2])[c] += (*d_input)[n](0, 0, c) / N;
		}
	}

	// ����
	// ���ӻ�ǰ��������������ݶ�
	//vector<cv::Mat_<double>> vec_mat_in0;
	//visualize((*d_input)[0], vec_mat_in0);
	//vector<cv::Mat_<double>> vec_mat_in1;
	//visualize((*d_input)[1], vec_mat_in1);
	// ���ӻ��������������
	//vector<cv::Mat_<double>> vec_mat_x0;
	//visualize((*cache[0])[0], vec_mat_x0);
	//vector<cv::Mat_<double>> vec_mat_x1;
	//visualize((*cache[0])[1], vec_mat_x1);
	// ���ӻ�������Ԫ(ȫ���Ӻ�)
	//vector<cv::Mat_<double>> vec_mat_w0;
	//visualize((*cache[1])[0], vec_mat_w0);
	//vector<cv::Mat_<double>> vec_mat_w1;
	//visualize((*cache[1])[1], vec_mat_w1);
	// ���ӻ���һ��������dX
	//vector<cv::Mat_<double>> vec_mat_dx0;
	//visualize((*grads[0])[0], vec_mat_dx0);
	// ���ӻ���һ����Ԫ��dW
	//vector<cv::Mat_<double>> vec_mat_dw0;
	//visualize((*grads[1])[0], vec_mat_dw0);
	// ���ӻ���һ����Ԫ��db
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
		// ��ȡ����
		mask.reset(new Blob(input[0]->size(), C_RANDU));
		int N = input[0]->get_N();
		for(int i = 0; i < N; i++) {
			(*mask)[i].transform([dropout_rate](double e) { return e < dropout_rate ? 0 : 1; });
		}
		// dropout��ע����Ҫ��ʣ��ķ�0���ݽ������ţ���֤��������
		output.reset(new Blob((*input[0]) * (*mask) / (1 - dropout_rate)));
	}
	else if(mode == "test") { // ����ģʽ�в���ʹ��dropout�������������
		output.reset(new Blob(*input[0]));
	}
}


void DropoutLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads,
								 shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	double dropout_rate = curr_params.dropout_rate;
	grads[0].reset(new Blob((*d_input) * (*mask) / (1 - dropout_rate)));
}


// before����״����Ϊ(1, 1, C)����ÿ��ͨ���Ϲ㲥��(H, W, C)
inline void broadcast(cube& after, cube& before, int C) {
	// slice���ڴ�cube�з���matrix
	for(int c = 0; c < C; c++) {
		after.slice(c).fill(as_scalar(before.slice(c)));
	}
}


// ��item�ڿ�͸�ά���ϵ�Ԫ����ӣ���״��(H, W, C)��Ϊ(1, 1, C)
inline cube de_broadcast(cube& item) {
	return sum(sum(item, 0), 1);
}


void BNLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params)
{
	cout << "initiating BNLayer" << endl;

	int C = in_shape[1];
	int H = in_shape[2];
	int W = in_shape[3];

	// ʹ����������W��b�Ŀռ��Ŷ�̬���µľ�ֵ�ͱ�׼��
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

	// ���������ߴ���ͬ
	output.reset(new Blob(input[0]->size(), C_ZEROS));

	int N = input[0]->get_N();
	int C = input[0]->get_C();
	int H = input[0]->get_H();
	int W = input[0]->get_W();

	if(mode == "train") {
		// ����ֵ/����/��׼��
		minus_mean_.reset(new cube(1, 1, C, fill::zeros));
		var_.reset(new cube(1, 1, C, fill::zeros));
		std_.reset(new cube(1, 1, C, fill::zeros));
		
		// ����Ǿ����Ҫ����ÿ��ͨ���϶Կ�����ֵ��ʹ����״��Ϊ(N, C, 1, 1)
		Blob averaged(N, C, 1, 1, C_ZEROS);
		for(int i = 0; i < N; i++) {
			averaged[i] = de_broadcast((*input[0])[i]) / (H * W);
		}

		// �����εĸ���ֵ��
		for(int i = 0; i < N; i++) {
			(*minus_mean_) += averaged[i];
		}
		(*minus_mean_) /= (-N);

		// �����εķ���
		for(int i = 0; i < N; i++) {
			(*var_) += square(averaged[i] + (*minus_mean_));
		}
		(*var_) /= N;

		// �����εı�׼��
		(*std_) = sqrt((*var_) + epsilon);

		// �㲥��ֵ�ͱ�׼��
		cube minus_mean_bc(H, W, C, fill::zeros);
		cube std_bc(H, W, C, fill::zeros);
		broadcast(minus_mean_bc, *minus_mean_, C);
		broadcast(std_bc, *std_, C);

		// ��һ��
		for(int i = 0; i < N; i++) {
			(*output)[i] = ((*input[0])[i] + minus_mean_bc) / std_bc;
		}
			
		// �õ�һ�����εľ�ֵ�ͱ�׼���ʼ����̬���µľ�ֵ�ͱ�׼��
		// �����տ�ʼ��������������֤��ʱ����һ�����׼ȷ�ľ�ֵ�ͱ�׼��
		if(!mean_std_initiated) {
			(*input[1])[0] = minus_mean_bc;
			(*input[2])[0] = std_bc;
			mean_std_initiated = true;
		}

		// ��̬���¾�ֵ�ͷ���
		(*input[1])[0] = eta * (*input[1])[0] + (1 - eta) * minus_mean_bc;
		(*input[2])[0] = eta * (*input[2])[0] + (1 - eta) * std_bc;
	}
	// ���Խ׶�ʹ�ñ���Ķ�̬��ֵ�ͱ�׼����й�һ��
	else if(mode == "test") {
		for(int n = 0; n < N; n++) {
			(*output)[n] = ((*input[0])[n] + (*input[1])[0]) / (*input[2])[0];
		}
	}
}


void BNLayer::backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, 
							shared_ptr<Blob>& d_input, const LayerParams& curr_params) {
	// �ݶȳߴ�
	grads[0].reset(new Blob(cache[0]->size(), C_ZEROS));
	int N = grads[0]->get_N();
	int C = grads[0]->get_C();
	int H = grads[0]->get_H();
	int W = grads[0]->get_W();

	// �㲥��ֵ�ͱ�׼��
	cube minus_mean_bc(H, W, C, fill::zeros);
	cube std_bc(H, W, C, fill::zeros);
	broadcast(minus_mean_bc, *minus_mean_, C);
	broadcast(std_bc, *std_, C);

	// ʹ�ôӼ���ͼ�õ��Ĺ�ʽ���з��򴫲���item1-item6Ϊ�м���
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
	
	// ÿ���������ݶ�ֵ�������ݶ���3���������
	for(int n = 0; n < N; n++) {
		cube grads_pt2(H, W, C, fill::zeros);
		cube grads_pt3 = (*d_input)[n] / std_bc;
		broadcast(grads_pt2, grads_cache[n], C);
		(*grads[0])[n] = (grads_pt1 + grads_pt2) / (H * W) + grads_pt3;
	}
}


void ScaleLayer::initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) {
	// ͨ����
	int C = in_shape[1];

	// ��ʼ��gamma��beta(Ҳ����W��b)
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

	// �㲥gamma��beta
	cube gamma(H, W, C, fill::zeros);
	cube beta(H, W, C, fill::zeros);
	broadcast(gamma, (*input[1])[0], C);
	broadcast(beta, (*input[2])[0], C);

	// ���������������ź�ƽ��
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

	// �㲥gamma
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

	// ��ȡ��سߴ�
	// (������N, �������(��һ�����Ԫ����), 1, 1)
	int N_in = input->get_N();
	int C_in = input->get_C();
	int H_in = input->get_H();
	int W_in = input->get_W();

	if(H_in != 1 || W_in != 1) {
		throw "input H and input W must be 1!";
	}

	d_output.reset(new Blob(N_in, C_in, H_in, W_in));  // ��ǰ����ݶ�

	// ����ÿ��������loss
	double total_loss = 0.0;
	for(int n = 0; n < N_in; n++) {
		// �Ե�ǰ��������softmax, ������cube / number
		cube prob = arma::exp((*input)[n]) / accu(arma::exp((*input)[n]));
		total_loss += (-accu((*Y)[n] % arma::log(prob)));

		// ��ǰ�����ݶ�
		(*d_output)[n] = prob - (*Y)[n];
	}

	return total_loss / N_in;
}


double HingeLossLayer::hinge(shared_ptr<Blob>& input, shared_ptr<Blob>& Y, shared_ptr<Blob>& d_output, double delta) {
	if(d_output) {
		d_output.reset();
	}
	// ��ȡ��سߴ�
	// (������N, �������(��һ�����Ԫ����), 1, 1)
	int N_in = input->get_N();
	int C_in = input->get_C();
	int H_in = input->get_H();
	int W_in = input->get_W();

	if(H_in != 1 || W_in != 1) {
		throw "input H and input W must be 1!";
	}

	d_output.reset(new Blob(N_in, C_in, H_in, W_in));  // ��ǰ����ݶ�

	double total_loss = 0.0;
	for(int n = 0; n < N_in; n++) {
		// loss
		int correct_idx = (*Y)[n].index_max();  // ��ǰ������ȷ������
		double correct_score = (*input)[n](0, 0, correct_idx); // ��ȷ���Ԥ��÷�
		// ÿ�����(��������ȷ���)�Ĵ���÷ּ�ȥ��ȷ�÷ּ���delta
		cube score_cube = (*input)[n] - correct_score + delta;
		score_cube(0, 0, correct_idx) = 0;
		// ����ÿ��������ʧ
		score_cube.transform([](double e){ return e > 0 ? e : 0; });
		total_loss += arma::accu(score_cube);

		// �ݶȣ����������ݶȾ���max����ʱ������ֵ����ȷ�����ݶ��Ǵ�������ݶ�֮�ͳ���-1
		score_cube.transform([](double e){ return e ? 1 : 0; });
		score_cube(0, 0, correct_idx) = -arma::accu(score_cube);
		(*d_output)[n] = score_cube;
	}

	return total_loss / N_in;
}