#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite
import numpy as np

class ModelTFlite(object):
    def __init__(self, model_file):
        self.interpreter = self._make_interpreter(model_file)
        self.interpreter.allocate_tensors()
        self.input_details, self.output_details = self._input_output_tensor()
    
    def _make_interpreter(self, model_file):
        return tflite.Interpreter(
        model_path=model_file)

    def _get_output(self,):
        """Returns entire output, threshold is applied later."""
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data.copy()

    def _input_output_tensor(self,):
        """Returns dequantized output tensor if quantized before."""
        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter_shape = self.interpreter.get_input_details()[0]['shape']
        return input_details, output_details

    def _set_input(self, input_data):
        """Copies data to input tensor."""
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data[None,...])

    def predict(self, data):
        self._set_input(data)
        self.interpreter.invoke()
        return  self._get_output()
