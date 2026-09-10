"""Quantised exports must remain accurate on non-VNNI x86 as well as Arm."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper

from neural.export import quantise


class QuantizationTests(unittest.TestCase):
    def test_dense_weights_cannot_saturate_pairwise_int16_products(self):
        # Full-range U8S8 weights produce 2 * 255 * 127 per pair here,
        # exceeding int16 on AVX2/non-VNNI AVX512. Check the stored weights
        # and emulate that accumulator so this regression also runs on VNNI.
        weights = np.tile(np.array([[1., -1.]], dtype=np.float32), (8, 1))
        graph = helper.make_graph(
            [helper.make_node('MatMul', ['planes', 'weights'], ['value'])],
            'saturation-regression',
            [helper.make_tensor_value_info('planes', TensorProto.FLOAT, ['batch', 8])],
            [helper.make_tensor_value_info('value', TensorProto.FLOAT, ['batch', 2])],
            [numpy_helper.from_array(weights, name='weights')],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
        with tempfile.TemporaryDirectory() as root:
            source, destination = Path(root) / 'float.onnx', Path(root) / 'int8.onnx'
            onnx.save(model, source)
            quantise(source, destination)
            made = onnx.load(destination)
            onnx.checker.check_model(made)
            initializers = {value.name: numpy_helper.to_array(value) for value in made.graph.initializer}
            node = next(node for node in made.graph.node if node.op_type == 'MatMulInteger')
            quantized = initializers[node.input[1]].astype(np.int32)
            self.assertLessEqual(int(np.abs(quantized).max()), 64)
            products = 255 * quantized
            pairs = products.reshape(4, 2, 2).sum(axis=1)
            np.testing.assert_array_equal(np.clip(pairs, -32768, 32767).sum(axis=0),
                                          products.sum(axis=0))
            session = onnxruntime.InferenceSession(str(destination), providers=['CPUExecutionProvider'])
            values = np.array([[1.] * 8, [0., .25, .5, .75, 1., .75, .5, .25]], dtype=np.float32)
            result, = session.run(['value'], {'planes': values})
            np.testing.assert_allclose(result, values @ weights, rtol=.01, atol=.01)


if __name__ == '__main__':
    unittest.main()
