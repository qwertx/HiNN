#ifndef __HINET_HPP__
#define __HINET_HPP__
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "hiLayer.hpp"
#include "hiBlob.hpp"
#include "hiModel.pb.h"

using std::vector;
using std::string;
using std::unordered_map;
using std::shared_ptr;

struct NetParams {
	double lr;
	double lr_decay;
	string optimizer;
	double momentum_rms_value;
	double regularization;
	int epoch_num;
	bool use_batch;
	int batch_size;
	int update_acc_epoches;
	bool lr_update;
	bool snapshot;
	int snapshot_interval;
	bool fine_tune;
	string pre_train_model_path;

	vector<string> layer_name;
	vector<string> layer_type;

	unordered_map<string, LayerParams> layer_params;

	void readNetParams(string file);
};


class Net {
public:
	Net(NetParams& net_params, shared_ptr<Blob> X_train, shared_ptr<Blob> Y_train, 
				 shared_ptr<Blob> X_validate, shared_ptr<Blob> Y_validate);
	void train(NetParams& net_params);
	void train_with_batch(shared_ptr<Blob> X_batch, shared_ptr<Blob> Y_batch, NetParams& net_params, string mode = "train");
	void regularize(NetParams& net_params, int N, string mode = "train");
	void optimize(NetParams& net_params);
	void evaluate(shared_ptr<Blob> Y_batch, NetParams& net_params);
	double accuracy(Blob& predict, Blob& Y);
	void save_model(shared_ptr<hi::Snapshot>& snapshot, NetParams& net_params);
	void load_model(const shared_ptr<hi::Snapshot>& snapshot, NetParams& net_params);

private:
	shared_ptr<Blob> X_train_;
	shared_ptr<Blob> Y_train_;
	shared_ptr<Blob> X_validate_;
	shared_ptr<Blob> Y_validate_;

	double train_loss_;
	double val_loss_;
	double train_acc_;
	double val_acc_;

	// 每一层有3个参数Blob：上一层输入X/W/b
	unordered_map<string, vector<shared_ptr<Blob>>> params_;
	// 同上，每一层有3个梯度Blob，dX/dW/db
	unordered_map<string, vector<shared_ptr<Blob>>> grads_;
	// 在Momentum和RMSProp中存储累加梯度V/S
	unordered_map<string, vector<shared_ptr<Blob>>> op_cache_;
	// loss层没有以下两个参数，即层对象和输出形状
	// 保存层对象
	unordered_map<string, shared_ptr<Layer>> layers_;
	// 每层的输出尺寸
	unordered_map<string, vector<int>> out_shapes_;
};


#endif