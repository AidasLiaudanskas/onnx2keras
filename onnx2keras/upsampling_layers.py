from tensorflow import keras
import numpy as np
import logging


def convert_upsample(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert upsample.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:upsample')
    logger.warning('!!! EXPERIMENTAL SUPPORT (upsample) !!!')

    if len(node.input) == 1:
        scale = np.uint8(params['scales'][-2:])
    elif len(node.input) == 2:
        scale = np.uint8(layers[node.input[1]][-2:])
        logger.warning(
            "Found two inputs, assuming second input is the upsampling scale!")
    else:
        raise AttributeError(
            'Unsupported number of inputs. Expected 1 or 2, got {}. {}'.format(
                len(node.input), node.input))
    interpolation_type = params['mode'].decode('utf-8')
    interpolation_type = 'bilinear' if 'linear' in interpolation_type else interpolation_type

    if interpolation_type not in ['nearest', 'bilinear']:
        logger.warning(
            'Cannot convert non-nearest upsampling. Got {}. Might not work properly'
            .format(interpolation_type))
    # TODO: might need to force nearest upsampling here? Works in my case..

    upsampling = keras.layers.UpSampling2D(size=scale,
                                           name=keras_name,
                                           interpolation=interpolation_type)

    layers[node_name] = upsampling(layers[node.input[0]])
