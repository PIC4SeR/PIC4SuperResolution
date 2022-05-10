from pycoral.utils.edgetpu import make_interpreter
import numpy as np

class ModelCORAL(object):
    def __init__(self, model_file):
        self.interpreter = self._make_interpreter(model_file)
        self.interpreter.allocate_tensors()
        self.input_details, self.output_details = self._input_output_tensor()
    
    def _make_interpreter(self, model_file):
        interpreter = make_interpreter(model_file)
        return interpreter

    def _get_output(self,):
        """Returns entire output, threshold is applied later."""
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        scale, zero_point = self.output_details[0]['quantization']
        output_data = output_data.astype(np.float32)
        output_data = (output_data - zero_point) * scale
        output_data = np.clip(output_data, 0, 255)
        output_data = np.round(output_data)
        return output_data.copy().astype('uint8') 

    def _input_output_tensor(self,):
        """Returns dequantized output tensor if quantized before."""
        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter_shape = self.interpreter.get_input_details()[0]['shape']
        self.input_scale, self.input_zero_point = input_details[0]['quantization']
        return input_details, output_details

    def _set_input(self, input_data):
        """Copies data to input tensor."""
        test_image_int = input_data / self.input_scale + self.input_zero_point
        test_image_int = test_image_int.astype(self.input_details[0]['dtype'])
        self.interpreter.set_tensor(self.input_details[0]['index'], test_image_int[None,...])

    def predict(self, data):
        self._set_input(data)
        self.interpreter.invoke()
        return  self._get_output()
