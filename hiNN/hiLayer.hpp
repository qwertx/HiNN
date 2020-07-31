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
	// ���麯����������
	virtual void initLayer(const vector<int>& in_shape, vector<shared_ptr<Blob>>& curr_blobs, const LayerParams& curr_params) = 0;
	virtual void nextLayerShape(const vector<int>& in_shape, vector<int>& out_shape, const LayerParams& curr_params) = 0;
	// ��һ���������д����
	virtual void forward_prop(const vector<shared_ptr<Blob>>& input, shared_ptr<Blob>& output, const LayerParams& curr_params, string mode = "train") = 0;
	// ����ĵ�ǰ��W/X/b�Ȳ�������ż������ĵ�ǰ���ݶ�Blob vector����һ�㴫����ݶȣ���ǰ��ĳ���
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
	// �洢ǰ����������ڷ��򴫲�
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
	shared_ptr<Blob> mask; // ǰ��ͷ��򴫲�����Ҫʹ������
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
	bool mean_std_initiated; // ��̬���µľ�ֵ�ͱ�׼��
	double epsilon; // ���ڷ������ĳ�����ʹ��ֵ�ȶ�
	double eta; // ��̬���¾�ֵ�ͷ���ʱ��Ȩֵ
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
	// d_outputΪ��ǰ����ݶ�
	static double softmax_cross_entropy_with_logits(shared_ptr<Blob>& input, shared_ptr<Blob>& Y, shared_ptr<Blob>& d_output);
};


class HingeLossLayer {
public:
	// d_outputΪ��ǰ����ݶ�
	static double hinge(shared_ptr<Blob>& input, shared_ptr<Blob>& Y, shared_ptr<Blob>& d_output, double delta = 0.2);
};

#endif