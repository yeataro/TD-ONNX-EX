# This should be able to convert the pre-trained models from lltcggie/waifu2x-caffe to ONNX

# Based on the onnx converter script:
# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/caffe_coreml_onnx.ipynb

import os
import coremltools
import onnxmltools
import logging

models_dir = 'models/'

def convert_model(model_name, model_path):
  # Update your input name and path for your caffe model
  proto_file = os.path.join(model_path, model_name+".prototxt")
  input_caffe_path = os.path.join(model_path, model_name+".json.caffemodel")
  # Update the output name and path for intermediate coreml model, or leave as is
  output_coreml_model = os.path.join(model_path, model_name+".mlmodel")
  # Change this path to the output name and path for the onnx model
  output_onnx_model = os.path.join(model_path, model_name+".onnx")

  # Convert Caffe model to CoreML
  coreml_model = coremltools.converters.caffe.convert((input_caffe_path, proto_file))
  # Save CoreML model
  coreml_model.save(output_coreml_model)
  # Load CoreML model
  coreml_model = coremltools.utils.load_spec(output_coreml_model)
  # Convert CoreML model to ONNX
  onnx_model = onnxmltools.convert_coreml(coreml_model)
  # Save ONNX model
  onnxmltools.utils.save_model(onnx_model, output_onnx_model)


def run():
  logger = logging.getLogger()
  for (path, _, files) in os.walk(models_dir):
    model_files = filter(lambda f: f.endswith('.caffemodel'), files)
    for file in model_files:
      model_name = file[:-16]
      try:
        convert_model(model_name, path)
      except:
        logger.exception("Error when converting {}/{}".format(path, model_name))
        pass


if __name__ == "__main__":
  run()
