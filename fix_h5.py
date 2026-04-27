import h5py
import json

f = h5py.File('keras_model.h5', 'r+')
model_config = json.loads(f.attrs.get('model_config'))

for layer in model_config['config']['layers']:
    if layer['class_name'] == 'LSTM':
        if 'time_major' in layer['config']:
            del layer['config']['time_major']

f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
f.close()
print("Successfully removed time_major from keras_model.h5")
