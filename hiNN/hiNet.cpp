#include "hiNet.hpp"
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <cassert>
#include <iomanip>

using namespace std;
using namespace arma;


void NetParams::readNetParams(string file) {
	ifstream ifs;
	ifs.open(file);
	if(!ifs.is_open()) {
		throw "cannot open json file!";
	}

	Json::Reader reader;
	Json::Value value;
	if (reader.parse(ifs, value)) {
		if (!value["hyper_params"].isNull()) {
			auto& nparams = value["hyper_params"];
			lr = nparams["lr"].asDouble();
			lr_decay = nparams["lr_decay"].asDouble();
			optimizer = nparams["optimizer"].asString();
			momentum_rms_value = nparams["momentum_rms_value"].asDouble();
			regularization = nparams["regularization"].asDouble();
			epoch_num = nparams["epoch_num"].asInt();
			use_batch = nparams["use_batch"].asBool();
			batch_size = nparams["batch_size"].asInt();
			update_acc_epoches = nparams["update_acc_epoches"].asInt();
			lr_update = nparams["lr_update"].asBool();
			snapshot = nparams["snapshot"].asBool();
			snapshot_interval = nparams["snapshot_interval"].asInt();
			fine_tune = nparams["fine_tune"].asBool();
			pre_train_model_path = nparams["pre_train_model_path"].asString();
		}

		if (!value["net_structure"].isNull()) {
			auto& curr_layer_params = value["net_structure"];
			// ����ÿһ��
			for (auto& layer: curr_layer_params) {
				auto curr_layer_name = layer["name"].asString();
				auto curr_layer_type = layer["type"].asString();
				layer_name.push_back(curr_layer_name);
				layer_type.push_back(curr_layer_type);

				if (curr_layer_type == "Conv") {
					layer_params[curr_layer_name].conv_num = layer["kernel_num"].asInt();
					layer_params[curr_layer_name].conv_width = layer["kernel_width"].asInt();
					layer_params[curr_layer_name].conv_height = layer["kernel_height"].asInt();
					layer_params[curr_layer_name].conv_pad = layer["pad"].asInt();
					layer_params[curr_layer_name].conv_stride = layer["stride"].asInt();
					layer_params[curr_layer_name].conv_init = layer["init_method"].asString();

				}
				else if (curr_layer_type == "Pool") {
					layer_params[curr_layer_name].pool_width = layer["kernel_width"].asInt();
					layer_params[curr_layer_name].pool_height = layer["kernel_height"].asInt();
					layer_params[curr_layer_name].pool_stride = layer["stride"].asInt();
				}
				else if (curr_layer_type == "FC") {
					layer_params[curr_layer_name].fc_neuron = layer["neuron_num"].asInt();
					layer_params[curr_layer_name].fc_init = layer["init_method"].asString();
				}
				else if(curr_layer_type == "Dropout") {
					layer_params[curr_layer_name].dropout_rate = layer["dropout_rate"].asDouble();
				}
			}
		}
	}
}


Net::Net(NetParams& net_params, shared_ptr<Blob> X_train, shared_ptr<Blob> Y_train,
	shared_ptr<Blob> X_validate, shared_ptr<Blob> Y_validate) :
	X_train_(X_train), Y_train_(Y_train), X_validate_(X_validate), Y_validate_(Y_validate) {

	auto layer_num = net_params.layer_name.size();
	// ��һ�������
	vector<int> in_shape{net_params.batch_size, X_train_->get_C(),
							  X_train_->get_H(), X_train_->get_W()};

	
	for(unsigned i = 0; i < layer_num; i++) {
		string layer_name = net_params.layer_name[i];
		string layer_type = net_params.layer_type[i];

		// ��ӡ��ṹ
		cout << "layer_name: " << net_params.layer_name[i] << "; ";
		cout << "layer_type: " << net_params.layer_type[i] << endl;

		// ��ʼ������Blob���ݶ�Blob���Լ�ÿһ������ά��
		params_[layer_name] = vector<shared_ptr<Blob>>(3, nullptr);
		grads_[layer_name] = vector<shared_ptr<Blob>>(3, nullptr);
		op_cache_[layer_name] = vector<shared_ptr<Blob>>(3, nullptr);

		// ����ʼ��������������loss��
		if(i != layer_num - 1) {
			out_shapes_[layer_name] = vector<int>(4, 0);  // (N, C, H, W)

			shared_ptr<Layer> curr_layer;

			cout << "input_shape: " << in_shape[0] << ", " << in_shape[1] << ", " << in_shape[2] << ", " << in_shape[3] << endl;
			if(layer_type == "Conv") {
				curr_layer.reset(new ConvLayer);
			}
			else if(layer_type == "ReLU") {
				curr_layer.reset(new ReLULayer);
			}
			else if(layer_type == "Tanh") {
				curr_layer.reset(new TanhLayer);
			}
			else if(layer_type == "Pool") {
				curr_layer.reset(new PoolLayer);
			}
			else if(layer_type == "FC") {
				curr_layer.reset(new FCLayer);
			}
			else if(layer_type == "Dropout") {
				curr_layer.reset(new DropoutLayer);
			}
			else if(layer_type == "BN") {
				curr_layer.reset(new BNLayer);
			}
			else if(layer_type == "Scale") {
				curr_layer.reset(new ScaleLayer);
			}

			layers_[layer_name] = curr_layer;
			curr_layer->initLayer(in_shape, params_[layer_name], net_params.layer_params[layer_name]);
			curr_layer->nextLayerShape(in_shape, out_shapes_[layer_name], net_params.layer_params[layer_name]);

			// ����in_shape
			in_shape.assign(out_shapes_[layer_name].begin(), out_shapes_[layer_name].end());

			// ��ӡ��ǰ��w��b
			/*if(params_[layer_name][1] != nullptr && params_[layer_name][2] != nullptr) {
				params_[layer_name][1]->print_blob("W: ");
				params_[layer_name][2]->print_blob("b: ");
			}*/
		}
	}

	// ����Ԥѵ��ģ��΢��
	if(net_params.fine_tune) {
		fstream input(net_params.pre_train_model_path, ios::in | ios::binary);
		if(!input) {
			cout << net_params.pre_train_model_path << " not found!" << endl;
			return;
		}

		shared_ptr<hi::Snapshot> snapshot(new hi::Snapshot);
		if(!snapshot->ParseFromIstream(&input)) {
			cout << net_params.pre_train_model_path << " parse failed!" << endl;
			return;
		}
		cout << net_params.pre_train_model_path << " loaded sucessfully." << endl;
		load_model(snapshot, net_params);
	}
}


void Net::train(NetParams& net_params) {
	int N = X_train_->get_N();
	int batch_size = net_params.batch_size;
	cout << "N = " << N << endl;
	// �ܵ���������ÿ��epoch�ĵ������� * epoch����
	int iters = N / batch_size * net_params.epoch_num;

	shared_ptr<Blob> X_batch;
	shared_ptr<Blob> Y_batch;
	for(int iter = 0; iter < iters; iter++) {
	//for(int iter = 0; iter < 40; iter++) {
		// ��ȡminibatch
		int batch_start = (iter * batch_size) % N;
		int batch_end = ((iter + 1) * batch_size) % N;
		X_batch.reset(new Blob(X_train_->splitBlob(batch_start, batch_end)));
		Y_batch.reset(new Blob(Y_train_->splitBlob(batch_start, batch_end)));

		// ǰ��/���򴫲�
		train_with_batch(X_batch, Y_batch, net_params);

		// ����׼ȷ�ʣ�ѵ��������֤����
		if(iter > 0 && iter % net_params.update_acc_epoches == 0) {
			evaluate(Y_batch, net_params);
			cout.setf(ios::fixed);
			cout << "iter = " << (iter + 1) << "/" << iters << "    "
				 << "lr = " << fixed << setprecision(8) << net_params.lr << "    "
				 << "train loss = " << fixed << setprecision(5) << train_loss_ << "    "
				 << "val loss = " << val_loss_ << "    "
				 << "train accuracy = " << fixed << setprecision(1) << train_acc_ * 100 << "%" << "    "
				 << "validate accuracy = " << val_acc_ * 100 << "%"
				 << endl;
		}
		
		// ����ģ��
		if(iter > 0 && net_params.snapshot && (iter % net_params.snapshot_interval == 0)) {
			// ����ļ������40���ַ�
			char out_file[40];
			sprintf(out_file, "./iter%d.saved_model", iter);
			// trunc���Ǵ��ڵ��ļ�
			fstream output(out_file, ios::out | ios::trunc | ios::binary);

			// ������Blobд��protobuf�ж����Snapshot��
			shared_ptr<hi::Snapshot> snapshot(new hi::Snapshot);
			save_model(snapshot, net_params);

			// ��Snapshotд��out_file�ļ�
			if(!snapshot->SerializeToOstream(&output)) {
				// ���д��ʧ��
				cout << "serialize model snapshot failed!" << endl;
			}
		}
	}
}


void Net::train_with_batch(shared_ptr<Blob> X_batch, shared_ptr<Blob> Y_batch, NetParams& net_params, string mode) {
	// ��X_batch��䵽��ʼ��Blob��X
	params_[net_params.layer_name[0]][0] = X_batch;
	int layer_num = static_cast<int>(net_params.layer_name.size());
	double tmp_loss = 0.0;

	// ���ǰ����㣬�����һ�㵥������
	for(int i = 0; i < layer_num - 1; i++) {
		string curr_layer_name = net_params.layer_name[i];
		shared_ptr<Blob> output;
		layers_[curr_layer_name]->forward_prop(params_[curr_layer_name], output, net_params.layer_params[curr_layer_name], mode);
		params_[net_params.layer_name[i + 1]][0] = output;
	}

	// ����loss
	if(net_params.layer_type.back() == "Cross Entropy") {
		tmp_loss = CrossEntropyLossLayer::softmax_cross_entropy_with_logits(params_[net_params.layer_name.back()][0],
																			Y_batch,
																			grads_[net_params.layer_name.back()][0]);
	}
	else if(net_params.layer_type.back() == "Hinge") {
		tmp_loss = HingeLossLayer::hinge(params_[net_params.layer_name.back()][0],
										 Y_batch,
										 grads_[net_params.layer_name.back()][0]);
	}

	// ����ʱ�������val_loss_
	if(mode == "train") {
		train_loss_ = tmp_loss;
		// ��һ�ε�������ʧԼΪ2.3����Ϊû�з��򴫲���ÿ������ƽ������ԼΪ0.1����-log(0.1)ԼΪ2.3
		//cout << "loss = " << loss_ << endl;
	}
	else if(mode == "test") {
		val_loss_ = tmp_loss;
		return;
	}

	// ���򴫲����ӵ����ڶ��㿪ʼ
	if(mode == "train") {
		for(int i = layer_num - 2; i >= 0; i--) {
			string curr_layer_name = net_params.layer_name[i];
			layers_[curr_layer_name]->backward_prop(params_[curr_layer_name],
													grads_[curr_layer_name],
													grads_[net_params.layer_name[i + 1]][0],
													net_params.layer_params[curr_layer_name]);
		}
	}

	// L2����
	if(net_params.regularization != 0) {
		regularize(net_params, X_batch->get_N(), mode);
	}

	// �ݶ��½��������£�ֻ��ѵ��ʱ��Ҫ
	if(mode == "train") {
		optimize(net_params);
	}
}


void Net::regularize(NetParams& net_params, int N, string mode) {
	double reg_item = 0.0;
	for(auto curr_layer : net_params.layer_name) {
		if(grads_[curr_layer][1]) { // ��ǰ����W
			if(mode == "train") { // ֻ��ѵ��ʱ��Ҫ�����ݶ�
				(*grads_[curr_layer][1]) = (*grads_[curr_layer][1]) + net_params.regularization * (*params_[curr_layer][1]) / static_cast<double>(N);
			}
			reg_item += accu(square(*grads_[curr_layer][1]));
		}
	}
	reg_item = reg_item * net_params.regularization / N / 2;
	if(mode == "train") {
		train_loss_ = train_loss_ + reg_item;
	}
	else if(mode == "test") {
		val_loss_ = val_loss_ + reg_item;
	}
}


void Net::optimize(NetParams& net_params) {
	int layer_num = net_params.layer_name.size();
	for(int n = 0; n < layer_num; n++) {
		string curr_layer = net_params.layer_name[n];
		// û��W��b�Ĳ��BN��������µ�ǰ�����
		if(!params_[curr_layer][1] || net_params.layer_type[n] == "BN") {
			continue;
		}
		// 1��W��2��b
		for(int i = 1; i <= 2; i++) {
			// d_tmp = lr * dW/db
			shared_ptr<Blob> d_tmp(new Blob(params_[curr_layer][i]->size(), C_ZEROS));

			if(net_params.optimizer == "sgd") {
				(*d_tmp) = -net_params.lr * (*grads_[curr_layer][i]);
			}
			else if(net_params.optimizer == "momentum") {
				if(!op_cache_[curr_layer][i]) {
					op_cache_[curr_layer][i].reset(new Blob(params_[curr_layer][i]->size(), C_ZEROS));
				}
				(*op_cache_[curr_layer][i]) = net_params.momentum_rms_value * (*op_cache_[curr_layer][i]) + (*grads_[curr_layer][i]);
				(*d_tmp) = -net_params.lr * (*op_cache_[curr_layer][i]);
			}
			else if(net_params.optimizer == "rmsprop") {
				double epsilon = 1e-8;
				double decay = net_params.momentum_rms_value;
				if(!op_cache_[curr_layer][i]) {
					op_cache_[curr_layer][i].reset(new Blob(params_[curr_layer][i]->size(), C_ZEROS));
				}
				(*op_cache_[curr_layer][i]) = decay * (*op_cache_[curr_layer][i]) + (1 - decay) * (*grads_[curr_layer][i]) * (*grads_[curr_layer][i]);
				(*d_tmp) = -net_params.lr * (*grads_[curr_layer][i]) / sqrt(epsilon + (*op_cache_[curr_layer][i]));
			}
			else {
				throw "wrong optimizer!";
			}
			// ����W/b
			(*params_[curr_layer][i]) = (*params_[curr_layer][i]) + (*d_tmp);
		}
	}
	// ����ѧϰ��
	if(net_params.lr_update) {
		net_params.lr *= net_params.lr_decay;
	}
}


void Net::evaluate(shared_ptr<Blob> Y_batch, NetParams& net_params) {
	// ����ѵ����׼ȷ��
	shared_ptr<Blob> X_eval;
	shared_ptr<Blob> Y_eval;
	int N = X_train_->get_N();
	if(N > 1000) {
		X_eval.reset(new Blob(X_train_->splitBlob(0, 1000)));
		Y_eval.reset(new Blob(Y_train_->splitBlob(0, 1000)));
	}
	else {
		X_eval = X_train_;
		Y_eval = Y_train_;
	}
	string last_layer = net_params.layer_name.back();
	train_with_batch(X_eval, Y_eval, net_params, "test");
	train_acc_ = accuracy(*params_[last_layer][0], *Y_eval);

	// ������֤��׼ȷ��
	train_with_batch(X_validate_, Y_validate_, net_params, "test");
	val_acc_ = accuracy(*params_[last_layer][0], *Y_validate_);
}


double Net::accuracy(Blob& predict, Blob& Y) {
	// ȷ��Ԥ��ͱ�ǩBlob��Сһ��
	vector<int> size_p = predict.size();
	vector<int> size_y = Y.size();

	for(int i = 0; i < size_p.size(); i++) {
		if(size_p[i] != size_y[i]) {
			throw "two blobs with different size cannot be multiplied!";
		}
	}

	// ��Ԥ��ֵ�ͱ�ǩ�����ֵλ�ý��бȽϣ���ͳ����ȷԤ�����������
	int N = Y.get_N();
	int correct = 0;
	for(int n = 0; n < N; n++) {
		// index_maxΪ���ֵ������������һ����ֵ����������
		if(Y[n].index_max() == predict[n].index_max())
			correct++;
	}
	return static_cast<double>(correct) / static_cast<double>(N);
}


void Net::save_model(shared_ptr<hi::Snapshot>& snapshot, NetParams& net_params) {
	// �����ã���ӡ����W��b
	//for(auto curr_layer : net_params.layer_name) {
	//	if(!params_[curr_layer][1]) {
	//		continue;
	//	}
	//	cout << curr_layer << endl;
	//	for(int i = 1; i <= 2; i++) {
	//		cout << (i == 1 ? "weight = " : "bias = ") << endl;
	//		params_[curr_layer][i]->print_blob();
	//	}
	//}

	for(auto curr_layer : net_params.layer_name) {
		// û��W��b�����豣��
		if(!params_[curr_layer][1]) {
			continue;
		}
		// ������ǰ���W��b
		for(int i = 1; i <= 2; i++) {
			hi::Snapshot_Block* block = snapshot->add_block();
			int N = params_[curr_layer][i]->get_N();
			int C = params_[curr_layer][i]->get_C();
			int H = params_[curr_layer][i]->get_H();
			int W = params_[curr_layer][i]->get_W();
			block->set_kernel_n(N);
			block->set_kernel_c(C);
			block->set_kernel_h(H);
			block->set_kernel_w(W);
			block->set_layer_name(curr_layer);
			if(i == 1) {
				block->set_param_type("weight"); //д���������
			}
			else {
				block->set_param_type("bias");
			}
			for(int n = 0; n < N; n++) {
				for(int c = 0; c < C; c++) {
					for(int h = 0; h < H; h++) {
						for(int w = 0; w < W; w++) {
							hi::Snapshot_Block_Params* params = block->add_params();
							params->set_value((*params_[curr_layer][i])[n](h, w, c));
						}
					}
				}
			}
		}
	}
}


void Net::load_model(const shared_ptr<hi::Snapshot>& snapshot, NetParams& net_params) {
	// ����snapshot�е�����block(ÿ��һ��)
	for(int i = 0; i < snapshot->block_size(); i++) {
		const hi::Snapshot::Block& block = snapshot->block(i);

		string layer_name = block.layer_name();
		string param_type = block.param_type();
		int N = block.kernel_n();
		int C = block.kernel_c();
		int H = block.kernel_h();
		int W = block.kernel_w();

		// ������ǰblock�Ĳ�������Blob
		int val_idx = 0;
		shared_ptr<Blob> tmp(new Blob(N, C, H, W));

		for(int n = 0; n < N; n++) {
			for(int c = 0; c < C; c++) {
				for(int h = 0; h < H; h++) {
					for(int w = 0; w < W; w++) {
						const hi::Snapshot_Block_Params& value = block.params(val_idx);
						(*tmp)[n](h, w, c) = value.value();
						val_idx++;
					}
				}
			}
		}

		// ��tmp�еĲ�������params_
		if(param_type == "weight") {
			params_[layer_name][1] = tmp;
		}
		else {
			params_[layer_name][2] = tmp;
		}
	}

	// �����ã���ӡ����W��b
	//for(auto curr_layer : net_params.layer_name) {
	//	if(!params_[curr_layer][1]) {
	//		continue;
	//	}
	//	cout << curr_layer << endl;
	//	for(int i = 1; i <= 2; i++) {
	//		cout << (i == 1 ? "weight = " : "bias = ") << endl;
	//		params_[curr_layer][i]->print_blob();
	//	}
	//}
}