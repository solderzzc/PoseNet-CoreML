import mxnet.contrib.onnx as onnx_mxnet
sym, arg_params, aux_params = onnx_mxnet.import_model('model.onnx')
