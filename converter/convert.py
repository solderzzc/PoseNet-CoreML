import tfcoreml as tf_converter
import numpy as np
from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos
from tensorflow.python.tools.freeze_graph import _parse_input_graph_proto
from keras.preprocessing.image import load_img

import tfcoreml
import coremltools
import yaml

import tensorflow as tf

f = open("config.yaml", "r+")
cfg = yaml.load(f)
imageSize = cfg['imageSize']
checkpoints = cfg['checkpoints']
chk = cfg['chk']
chkpoint = checkpoints[chk]
versionName = chkpoint.lstrip('mobilenet_')

# Provide these to run freeze_graph:
# Graph definition file, stored as protobuf TEXT
graph_def_file = './models/model.pbtxt'
# Trained model's checkpoint name
checkpoint_file = './checkpoints/model.ckpt'
# Frozen model's output name
frozen_model_file = './models/frozen_model.pb'
# Output nodes. If there're multiple output ops, use comma separated string, e.g. "out1,out2".
output_node_names = 'heatmap,offset_2,displacement_fwd_2,displacement_bwd_2'
# output_node_names = 'Softmax'

def load_graph_def(input_graph):
  detection_graph = tf.Graph()
  with detection_graph.as_default():
      #graph_def = tf.GraphDef()
      graph_def = _parse_input_graph_proto(input_graph, False)
      tf.import_graph_def(graph_def, name='')
      graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
  return graph_def
my_graph_def = load_graph_def(graph_def_file)
# Call freeze graph
freeze_graph_with_def_protos(my_graph_def,
         input_checkpoint=checkpoint_file,
         input_saver_def=None,
         output_node_names=output_node_names,
         restore_op_name="save/restore_all",
         filename_tensor_name="save/Const:0",
         output_graph=frozen_model_file,
         clear_devices=True,
         initializer_nodes="")

input_tensor_shapes = {"image:0":[1,imageSize, imageSize, 3]}
coreml_model_file = './models/model.mlmodel'
# output_tensor_names = ['output:0']
output_tensor_names = ['heatmap:0','offset_2:0','displacement_fwd_2:0','displacement_bwd_2:0']

coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        image_input_names=['image:0'],
        output_feature_names=output_tensor_names,
        is_bgr=False,
        red_bias = -1,
        green_bias = -1,
        blue_bias = -1,
        image_scale = 2./255)


coreml_model.author = 'Infocom TPO'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Ver.0.0.1'

coreml_model.save('./models/posenet'+ str(imageSize) + '_' + versionName +'.mlmodel')

img = load_img("./images/tennis_in_crowd.jpg", target_size=(imageSize, imageSize))
print(img)
result = coreml_model.predict({'image__0': img})
#out = coreml_model.predict({'image__0': img})['heatmap__0']
out = result['heatmap__0']
offsets = result['offset_2__0']
displacementsFwd = result['displacement_fwd_2__0']
displacementsBwd = result['displacement_bwd_2__0']

print("#output coreml result.")

print(out.shape)
print(np.transpose(out))
print(out)
# print(out[:, 0:1, 0:1])
print(np.mean(out))

print('offsets {}'.format(offsets))
