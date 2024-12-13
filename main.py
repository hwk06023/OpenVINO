import cv2
import numpy as np
from openvino.inference_engine import IECore

# Load the model
ie = IECore()
net = ie.read_network(model="path/to/your/model.xml", weights="path/to/your/model.bin")
exec_net = ie.load_network(network=net, device_name="CPU")

# Prepare input
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
n, c, h, w = net.input_info[input_blob].input_data.shape

# Load an image
image = cv2.imread("path/to/your/image.jpg")
resized_image = cv2.resize(image, (w, h))
input_image = np.expand_dims(resized_image.transpose(2, 0, 1), axis=0)

# Perform inference
result = exec_net.infer(inputs={input_blob: input_image})

# Process output
output = result[output_blob]
print("Inference result:", output)
