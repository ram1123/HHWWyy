MODEL_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_Removed_EBEv2/"

# fetch model, then load it. Then get all input feature names
MODEL_PATH = MODEL_DIR + "model.keras"
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)

model.summary()
# input_feature_names = [layer.name for layer in model.layers if 'input' in layer.name][0]
# print("Input feature names:", input_feature_names)

for layer in model.layers:
    print(f"=> Layer name: {layer.name}, Layer type: {type(layer)}")
