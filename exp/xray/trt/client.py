import numpy as np
import sys
import gevent.ssl

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

import SimpleITK as sitk
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from data.dataset import ImageDataset


# 1. 尝试连接服务
url = '10.100.37.100:8700'
verbose = False
model_name = 'xray_classifier'

triton_client = httpclient.InferenceServerClient(url=url, verbose=verbose)

# img = np.random.rand(1,3,512,512)
# img = np.array(img, dtype=np.float32)

infile = '/data/medical/external/xray/CheXpert/CheXpert-v1.0-small/valid/patient64592/study1/view1_frontal.jpg'
image = ImageDataset.aug(infile)
image = image.numpy()
image = np.expand_dims(image, 0)
print(image.shape)

inputs = []
inputs.append(httpclient.InferInput('INPUT__0', [1,3,512,512], "FP32"))
inputs[0].set_data_from_numpy(image, binary_data=False)
outputs = []
outputs.append(httpclient.InferRequestedOutput('OUTPUT__0', binary_data=True))

results = triton_client.infer(model_name,
                                inputs,
                                outputs=outputs)

print(results.as_numpy('OUTPUT__0'))

print('hello world')