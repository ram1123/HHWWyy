import os
import argparse
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'

## --- This convertor runs with tensorflow backend ------------------------ ##
## --- if you get error starting with "Using Theano" ---------------------- ##
## --- or AttributeError: 'module' object has no attribute 'get_session' -- ##
## --- Open /home/$USER_NAME/.keras/keras.json and change the line ------- ##
## --- "backend": "theano" to "backend": "tensorflow" --------------------- ##
## --- and then execute the lines like below  ----------------------------- ##


## --- Example -- ##
## python convert_hdf5_2_pb.py --input <input .hdf5 file name with path> --output <output file name>


parser = argparse.ArgumentParser(description='Deploy keras model.')
parser.add_argument('--input', required=True, type=str, help="Input Keras model (hdf5)")
parser.add_argument('--output', required=False, type=str, default=None, help="Output Protocol Buffers file")
args = parser.parse_args()

import tensorflow as tf
from tensorflow.python.framework.graph_io import write_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow import keras

# added this for batch normalization
from tensorflow.keras import backend as KS
# import tensorflow.python.keras.backend as KS
# import tensorflow.python.keras.backend as KS
# import tensorflow.keras import backend as KS
# import tensorflow.keras.backend as KS
# sess = K.get_session()
#from common import LoadModel
from tensorflow.keras.models import load_model

# necessary !!!
tf.compat.v1.disable_eager_execution()

# added below two line for batch normalization
# Reference: https://stackoverflow.com/a/57403976/2302094
KS.clear_session()
KS.set_learning_phase(0)

print( 'args.input: ', args.input)
model = load_model(args.input)
model.summary()

#raise RuntimeError("stop")

#print( ("teste", [node.op.name for node in model.outputs]))
input_nodes = [model.inputs[0].name] #"main_input"]#  [model.inputs[0].name]
output_nodes = [model.outputs[0].name] #"main_output/Softmax"]#["output_node"]
#input_nodes = ["main_input"]#  [model.inputs[0].name]
#output_nodes = ["main_output/Softmax"]#["output_node"]
#node_wrapper = tf.identity(model.outputs[0], name=output_nodes[0])

print("Input node name : {}".format(input_nodes[0]))
print("Output node name: {}".format(output_nodes[0]))
# with KS.get_session() as sess:
with tf.compat.v1.keras.backend.get_session() as sess:

    ops = sess.graph.get_operations()
    const_graph = convert_variables_to_constants(sess, sess.graph.as_graph_def(add_shapes=True), [node.op.name for node in model.outputs])
    #print( ([node.op.name for node in model.outputs]))
    for node in model.outputs:
        print( 'node.op.name: ' , node.op.name)
    final_graph = const_graph

    #for node in model.outputs :
    #    shapes = node.op.attr[model.outputs[0].name]
    #    print( (shapes.list.shape[0].dim[0].size))
    #shape = final_graph.get_tensor_by_name(model.outputs[0].name) #.node(0) #final_graph.node(0).attr().at("shape").shape();
    #print( ("read input layer shape  " , shape) #shape.dim_size() , graphDef.node(0).name()))

if args.output is None:
    input_base = os.path.basename(args.input)
    out_dir = ''
    out_file_name = args.input.split('/')[0]
    print( 'out_file_name: ', out_file_name)
    out_file = out_file_name + ".pb"
else:
    out_dir, out_file = os.path.split(args.output)
write_graph(final_graph, out_dir, out_file, as_text=False)
