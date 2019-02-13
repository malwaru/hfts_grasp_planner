import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import os
import math

#
# numpy3d_to_array
# this function was
# taken from pycuda mailing list (striped for C ordering only)
#


def numpy3d_to_array(np_array, allow_surface_bind=False):
    d, h, w = np_array.shape
    # create ArrayDescriptor
    descr = cuda.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = cuda.dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0
    if allow_surface_bind:
        descr.flags = cuda.array3d_flags.SURFACE_LDST
    # create actual array
    device_array = cuda.Array(descr)
    # copy data
    copy = cuda.Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d
    copy()
    return device_array


def array_format_to_dtype(af):
    if af == cuda.array_format.UNSIGNED_INT8:
        return np.uint8
    elif af == cuda.array_format.UNSIGNED_INT16:
        return np.uint16
    elif af == cuda.array_format.UNSIGNED_INT32:
        return np.uint32
    elif af == cuda.array_format.SIGNED_INT8:
        return np.int8
    elif af == cuda.array_format.SIGNED_INT16:
        return np.int16
    elif af == cuda.array_format.SIGNED_INT32:
        return np.int32
    elif af == cuda.array_format.FLOAT:
        return np.float32
    else:
        raise TypeError("cannot convert array_format '%s' to a numpy dtype" % af)


def array_to_numpy3d(cuda_array):
    descriptor = cuda_array.get_descriptor_3d()
    w = descriptor.width
    h = descriptor.height
    d = descriptor.depth
    shape = d, h, w
    dtype = array_format_to_dtype(descriptor.format)
    numpy_array = np.zeros(shape, dtype)
    copy = cuda.Memcpy3D()
    copy.set_src_array(cuda_array)
    copy.set_dst_host(numpy_array)
    itemsize = numpy_array.dtype.itemsize
    copy.width_in_bytes = copy.dst_pitch = w*itemsize
    copy.dst_height = copy.height = h
    copy.depth = d
    copy()
    return numpy_array


class CudaInterpolator(object):
    kernel = None

    @staticmethod
    def get_kernel():
        if CudaInterpolator.kernel is None:
            # first load kernel
            kernel_file_name = os.path.normpath(os.path.dirname(__file__) + '/cuda_interpolation.cu')
            with open(kernel_file_name, 'r') as kernel_file:
                kernel_string = kernel_file.read()
            CudaInterpolator.kernel = SourceModule(kernel_string)
        return CudaInterpolator.kernel

    def __init__(self, data):
        # first get kernel
        kernel = CudaInterpolator.get_kernel()
        # copy data to gpu. need to convert to single precision
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        assert(data.dtype == np.float32)
        self._gpu_data = numpy3d_to_array(data)
        # allocate data to texture memory
        self._tex_field = kernel.get_texref('tex_field')
        self._tex_field.set_array(self._gpu_data)
        self._tex_field.set_filter_mode(cuda.filter_mode.LINEAR)
        # offset = self._gpu_data.bind_to_texref_ext(self._tex_field)
        # print "OFFSET", offset
        self._get_val_fn = kernel.get_function("get_val")
        self._get_val_grad_fn = kernel.get_function("get_val_and_grad")

    def interpolate(self, positions, tf_matrix=np.eye(4), scale=1.0):
        num_pos = positions.shape[0]
        out_values = np.empty(num_pos, dtype=np.float32)
        if positions.dtype != np.float32:
            positions = positions.astype(np.float32)
        if tf_matrix.dtype != np.float32:
            tf_matrix = tf_matrix.astype(np.float32)
        if num_pos == 0:
            return out_values
        grid = (int(math.ceil(num_pos / 1024.0)), 1)  # there can be at most 1024 threads per block
        block = (min(1024, num_pos), 1, 1)
        self._get_val_fn(cuda.In(positions), cuda.Out(out_values), cuda.In(tf_matrix), np.float32(scale),
                         np.int32(num_pos), grid=grid, block=block, texrefs=[self._tex_field])
        return out_values

    def gradient(self, positions, tf_matrix=np.eye(4), scale=1.0, delta=0.01):
        num_pos = positions.shape[0]
        out_values = np.empty(num_pos, dtype=np.float32)
        grad_values = np.empty((num_pos, 3), dtype=np.float32)
        if num_pos == 0:
            return out_values, grad_values
        if positions.dtype != np.float32:
            positions = positions.astype(np.float32)
        if tf_matrix.dtype != np.float32:
            tf_matrix = tf_matrix.astype(np.float32)
        grid = (int(math.ceil(num_pos / 1024.0)), 1)  # there can be at most 1024 threads per block
        block = (min(1024, num_pos), 1, 1)
        self._get_val_grad_fn(cuda.In(positions), cuda.Out(out_values), cuda.Out(grad_values),
                              cuda.In(tf_matrix), np.float32(scale), np.float32(delta), np.int32(num_pos),
                              grid=grid, block=block, texrefs=[self._tex_field])
        return out_values, grad_values
