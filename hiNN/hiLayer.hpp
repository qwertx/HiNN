#ifndef __HILAYER_HPP__
#define __HILAYER_HPP__
#include <string>
#include <vector>
#include <memory>
#include "hiBlob.hpp"

using std::vector;
using std::shared_ptr;


struct LayerParams {
	int conv_num;
	int conv_width;
	int conv_height;
	int conv_pad;
	int conv_stride;
	string conv_init;

	int pool_width;
	int pool_height;
	int pool_stride;

	int fc_neuron;
	string fc_init;

	double dropout_rate;
};


class Layer {
public:
	Layer() {}
	virtual ~Layer() {}
	// 纯虚函数必须重载
	virtual void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) = 0;
	virtual void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) = 0;
	// 下一层参数用于写入结果
	virtual void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train") = 0;
	// 缓存的当前层W/X/b等参数，存放计算结果的当前层梯度Blob vector，上一层传入的梯度，当前层的超参
	virtual void backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, 
							   shared_ptr<Blob>& d_input, const LayerParams& curr_params) = 0;
};


class ConvLayer : public Layer{
public:
	ConvLayer() {}
	~ConvLayer() {}
	void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params);
	void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params);
	void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train");
	void backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, shared_ptr<Blob>& d_input, const LayerParams& curr_params);
};


class ReLULayer : public Layer {
public:
	ReLULayer(): clip(true), clip_val(6.0) {}
	~ReLULayer() {}
	void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params);
	void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params);
	void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train");
	void backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, shared_ptr<Blob>& d_input, const LayerParams& curr_params);
private:
	double clip_val;
	bool clip;
};


class TanhLayer : public Layer
{
public:
	TanhLayer() {}
	~TanhLayer() {}
	void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params);
	void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params);
	void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train");
	void backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, shared_ptr<Blob>& d_input, const LayerParams& curr_params);
private:
	// 存储前向计算结果用于反向传播
	vector<cube> forward_cache;
};


class PoolLayer : public Layer {
public:
	PoolLayer() {}
	~PoolLayer() {}
	void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params);
	void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params);
	void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train");
	void backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, shared_ptr<Blob>& d_input, const LayerParams& curr_params);
};


class FCLayer : public Layer {
public:
	FCLayer() {}
	~FCLayer() {}
	void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params);
	void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params);
	void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train");
	void backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, shared_ptr<Blob>& d_input, const LayerParams& curr_params);
};


class DropoutLayer : public Layer {
public:
	DropoutLayer() {}
	~DropoutLayer() {}
	void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params);
	void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params);
	void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train");
	void backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, shared_ptr<Blob>& d_input, const LayerParams& curr_params);
private:
	shared_ptr<Blob> mask; // 前向和反向传播都需要使用掩码
};


class BNLayer : public Layer {
public:
	BNLayer(): mean_std_initiated(false), epsilon(1e-8), eta(0.99) {}
	~BNLayer() {}
	void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params);
	void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params);
	void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train");
	void backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, shared_ptr<Blob>& d_input, const LayerParams& curr_params);
private:
	bool mean_std_initiated; // 动态更新的均值和标准差
	double epsilon; // 加在方差后面的常数，使数值稳定
	double eta; // 动态更新均值和方差时的权值
	shared_ptr<cube> minus_mean_;
	shared_ptr<cube> var_;
	shared_ptr<cube> std_;
};


class ScaleLayer : public Layer {
public:
	ScaleLayer() {}
	~ScaleLayer() {}
	void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params);
	void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params);
	void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train");
	void backward_prop(const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& grads, shared_ptr<Blob>& d_input, const LayerParams& curr_params);
};


class CrossEntropyLossLayer {
public:
	// d_output为当前层的梯度
	static double softmax_cross_entropy_with_logits(shared_ptr<Blob>& input, shared_ptr<Blob>& Y, shared_ptr<Blob>& d_output);
};


class HingeLossLayer {
public:
	// d_output为当前层的梯度
	static double hinge(shared_ptr<Blob>& input, shared_ptr<Blob>& Y, shared_ptr<Blob>& d_output, double delta = 0.2);
};

#endif