from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def gelu(features, approximate=False, name=None):
    with ops.name_scope(name, "Gelu", [features]):
        features = ops.convert_to_tensor(features, name="features")
        if approximate:
            coeff = math_ops.cast(0.044715, features.dtype)
            return 0.5 * features * (
                    1.0 + math_ops.tanh(0.7978845608028654 *
                                        (features + coeff * math_ops.pow(
                                            features, 3))))
        else:
            return 0.5 * features * (1.0 + math_ops.erf(
                features / math_ops.cast(1.4142135623730951, features.dtype)))
