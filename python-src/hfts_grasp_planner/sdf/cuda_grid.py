import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.compiler as cucompiler
import numpy as np
import os
import math

#
# numpy3d_to_array
# this function was
# taken from pycuda mailing list (striped for C ordering only)
#


def numpy3d_to_array(np_array, layered=False):
    d, h, w = np_array.shape
    # create ArrayDescriptor
    descr = cuda.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = cuda.dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0
    if layered:
        descr.flags = cuda.array3d_flags.ARRAY3D_2DARRAY
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
    """
        This class allows reading values and gradients from a grid using the GPU.
        For frequent queries of the same positions, where only the transformation
        between grid and positions changes, you should use CudaStaticPositionsInterpolator.
    """
    kernel = None

    @staticmethod
    def get_kernel():
        if CudaInterpolator.kernel is None:
            # first load kernel
            kernel_file_name = os.path.normpath(os.path.dirname(__file__) + '/cuda_interpolation.cu')
            with open(kernel_file_name, 'r') as kernel_file:
                kernel_string = kernel_file.read()
            CudaInterpolator.kernel = cucompiler.compile(kernel_string)
        return CudaInterpolator.kernel

    def __init__(self, data, blayered=False):
        """
            Create a new CudaInterpolator.
            ---------
            Arguments
            ---------
            data, np.array of shape (n, m, k) (float) - grid data
        """
        # first get kernel
        kernel = CudaInterpolator.get_kernel()
        self._mod = cuda.module_from_buffer(kernel)
        # copy data to gpu. need to convert to single precision
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        assert(data.dtype == np.float32)
        data = data.transpose().copy()
        self._gpu_data = numpy3d_to_array(data, blayered)
        if blayered:
            # allocate data to texture memory
            self._tex_field = self._mod.get_texref('tex_field_layered')
            self._tex_field.set_array(self._gpu_data)
            self._tex_field.set_filter_mode(cuda.filter_mode.LINEAR)
            # get functions
            self._get_val_fn = self._mod.get_function("get_val_layered")
            self._get_val_fn.prepare(("P", "P", "P", np.float32, np.int32), texrefs=[self._tex_field])
            self._get_val_grad_fn = self._mod.get_function("get_val_and_grad_layered")
            self._get_val_grad_fn.prepare(("P", "P", "P", "P", "P", np.float32, np.float32, np.int32),
                                          texrefs=[self._tex_field])
        else:
            # allocate data to texture memory
            self._tex_field = self._mod.get_texref('tex_field')
            self._tex_field.set_array(self._gpu_data)
            self._tex_field.set_filter_mode(cuda.filter_mode.LINEAR)
            # get functions
            self._get_val_fn = self._mod.get_function("get_val")
            self._get_val_fn.prepare(("P", "P", "P", np.float32, np.int32), texrefs=[self._tex_field])
            self._get_val_grad_fn = self._mod.get_function("get_val_and_grad")
            self._get_val_grad_fn.prepare(("P", "P", "P", "P", "P", np.float32, np.float32, np.int32),
                                          texrefs=[self._tex_field])
        # allocate memory for transformation matrix and gradient directions
        self._gpu_tf_matrix = cuda.mem_alloc(12 * 4)  # 12 float32
        self._gpu_grad_dirs = cuda.mem_alloc(9 * 4)  # 9 float32
        # init variables for dynamic memory
        self._gpu_values = None
        self._gpu_grads = None
        self._cpu_values = None
        self._cpu_grads = None
        self._gpu_positions = None

    def _allocate_gpu_mem(self, num_pos, bgrads):
        """
            Ensure self._gpu_position, self._gpu_values, self._gpu_grads, self._cpu_values, self._cpu_grad
            are sufficiently large for the number of positions queried.
            ---------
            Arguments
            ---------
            num_pos, int - number of positions
            bgrads, bool - whether to allocate gradient memory
        """
        if self._cpu_values is None or self._cpu_values.shape[0] < num_pos:
            self._cpu_values = np.empty(num_pos, dtype=np.float32)
            self._gpu_values = cuda.mem_alloc(self._cpu_values.nbytes)
            self._gpu_positions = cuda.mem_alloc(3 * num_pos * 4)
        if bgrads and (self._cpu_grads is None or self._cpu_grads.shape[0] < num_pos):
            # TODO for layered grids, 3d gradient makes no sense
            self._cpu_grads = np.empty((num_pos, 3), dtype=np.float32)
            self._gpu_grads = cuda.mem_alloc(self._cpu_grads.nbytes)

    def interpolate(self, positions, tf_matrix=np.eye(4), scale=1.0):
        """
            Retrieve grid values for the given positions.
            ---------
            Arguments
            ---------
            positions, np array of shape (n, 3) - query positions
            tf_matrix, np array of shape (4, 4) - transformation matrix from position frame to grid frame
            scale, float - scaling factor from positions to grid indices
            -------
            Returns
            -------
            values, np array of shape (n,) - interpolated values at the given positions
                If a position is outside of the grid, the closest value is returned
                WARNING: the returned array is used for subsequent calls as well. You should copy
                    it, if you intend to keep the values.
        """
        num_pos = positions.shape[0]
        if positions.dtype != np.float32:
            positions = positions.astype(np.float32)
        if tf_matrix.dtype != np.float32:
            tf_matrix = tf_matrix.astype(np.float32)
        tf_matrix = tf_matrix.flatten()[:12]
        if num_pos == 0:
            return np.empty(0)
        self._allocate_gpu_mem(num_pos, False)
        # copy positions to gpu
        cuda.memcpy_htod(self._gpu_positions, positions)
        # copy tf matrix to gpu
        cuda.memcpy_htod(self._gpu_tf_matrix, tf_matrix)
        # compute number of threads
        grid = (int(math.ceil(num_pos / 1024.0)), 1)  # there can be at most 1024 threads per block
        block = (min(1024, num_pos), 1, 1)
        # self._get_val_fn(cuda.In(positions), cuda.Out(out_values), cuda.In(tf_matrix), np.float32(scale),
                        #  np.int32(num_pos), grid=grid, block=block, texrefs=[self._tex_field])
        self._get_val_fn.prepared_call(grid, block,
                                       self._gpu_positions, self._gpu_values, self._gpu_tf_matrix,
                                       np.float32(scale), np.int32(num_pos))
        # copy result
        cuda.memcpy_dtoh(self._cpu_values[:num_pos], self._gpu_values)
        return self._cpu_values[:num_pos]

    def gradient(self, positions, tf_matrix=np.eye(4), scale=1.0, delta=0.01):
        """
            Retrieve grid values and gradients for the given positions.
            ---------
            Arguments
            ---------
            positions, np array of shape (n, 3) - query positions
            tf_matrix, np array of shape (4, 4) - transformation matrix from position frame to grid frame
            scale, float - scaling factor from positions to grid indices
            delta, float - step size for numerical gradient computation
            -------
            Returns
            -------
            values, np array of shape (n,) - interpolated values at the given positions
                If a position is outside of the grid, the closest value is returned
            grad, np array of shape (n, 3) - gradients at the given positions. The gradients are w.r.t
                the x, y, z axis of the frame positions is defined in.
            WARNING: the returned arrays are used for subsequent calls as well. You should copy
                them, if you intend to keep the values.
        """
        num_pos = positions.shape[0]
        if num_pos == 0:
            return np.empty(0), np.empty(0)
        if positions.dtype != np.float32:
            positions = positions.astype(np.float32)
        if tf_matrix.dtype != np.float32:
            tf_matrix = tf_matrix.astype(np.float32)
        gradient_dirs = tf_matrix[:3, :3].transpose().flatten()
        tf_matrix = tf_matrix.flatten()[:12]
        self._allocate_gpu_mem(num_pos, True)
        # copy positions to gpu
        cuda.memcpy_htod(self._gpu_positions, positions)
        # copy tf matrix to gpu
        cuda.memcpy_htod(self._gpu_tf_matrix, tf_matrix)
        # copy gradient directions to gpu
        cuda.memcpy_htod(self._gpu_grad_dirs, gradient_dirs)
        # compute number of threads
        grid = (int(math.ceil(num_pos / 1024.0)), 1)  # there can be at most 1024 threads per block
        block = (min(1024, num_pos), 1, 1)
        self._get_val_grad_fn.prepared_call(grid, block,
                                       self._gpu_positions, self._gpu_values, self._gpu_grads, self._gpu_grad_dirs,
                                        self._gpu_tf_matrix, np.float32(scale), np.float32(delta), np.int32(num_pos))
        # copy result
        cuda.memcpy_dtoh(self._cpu_values[:num_pos], self._gpu_values)
        cuda.memcpy_dtoh(self._cpu_grads[:num_pos], self._gpu_grads)
        return self._cpu_values[:num_pos], self._cpu_grads[:num_pos]


class CudaStaticPositionsInterpolator(object):
    """
        Just like CudaInterpolator, but more efficient if you query the gradient/interpolate function multiple times
        for the same positions and only the transformation matrix changes. This allows us to allocate memory
        on the GPU only for the positions once and then only copy the transformation matrix to GPU when queried.

        Both data and positions need to be of type float. In fact, internally everything is converted to float32,
        as float64 isn't supported by many GPUs.
    """

    def __init__(self, data, positions, base_tf, scale, delta):
        """
            Create a new CudaStaticPositionsInterpolator.
            ---------
            Arguments
            ---------
            data, np.array of shape (n, m, k), float - grid data
            positions, np array of shape (j, 3), float - positions to perform queries for
            base_tf, np array of shape (4, 4), float - base transformation matrix that input tf matrices should be left
                multiplied by (map from world frame to data frame). This also defines the frame the gradients are computed for.
                The gradients will computed along the column vectors of base_tf, i.e. if base_tf maps from frame A to the
                local grid frame, the gradients will be computed w.r.t. frame A's x, y, and z axis.
            scale, float - scaling factor from positions to grid indices
            delta, float - step size for numerical gradient computation
        """
        num_pos = positions.shape[0]
        if num_pos == 0:
            raise ValueError("Can not create interpolator for 0 query positions")
        # first get kernel
        kernel = CudaInterpolator.get_kernel()
        self._mod = cuda.module_from_buffer(kernel)
        # copy data to gpu. need to convert to single precision
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        assert(data.dtype == np.float32)
        data = data.transpose().copy()
        self._gpu_data = numpy3d_to_array(data)
        # allocate data to texture memory
        self._tex_field = self._mod.get_texref('tex_field')
        self._tex_field.set_array(self._gpu_data)
        self._tex_field.set_filter_mode(cuda.filter_mode.LINEAR)
        # allocate memory for positions
        if positions.dtype == np.float64:
            positions = positions.astype(np.float32)
        assert(positions.dtype == np.float32)
        self._gpu_positions = cuda.mem_alloc(positions.nbytes)
        cuda.memcpy_htod(self._gpu_positions, positions)
        # allocate memory for values
        # self._gpu_values = cuda.mem_alloc(num_pos * 4)
        self._gpu_values = gpuarray.GPUArray(num_pos, np.float32)
        self._cpu_values = np.empty(num_pos, dtype=np.float32)
        # allocate memory for gradients
        self._gpu_gradients = cuda.mem_alloc(positions.nbytes)
        self._cpu_gradients = np.empty((num_pos, 3), dtype=np.float32)
        # allocate memory for matrix and gradient dirs
        self._gpu_tf_matrix = cuda.mem_alloc(12 * 4)  # 12 float32
        self._gpu_grad_dirs = cuda.mem_alloc(9 * 4)  # 9 float32
        # copy gradient directions
        float32_base_tf = base_tf.astype(np.float32)
        cuda.memcpy_htod(self._gpu_grad_dirs, float32_base_tf[:3, :3].transpose().flatten())
        # get actual functions and prepare them
        self._get_val_fn = self._mod.get_function("get_val")
        self._get_val_fn.prepare(("P", "P", "P", np.float32, np.int32), texrefs=[self._tex_field])
        self._get_val_grad_fn = self._mod.get_function("get_val_and_grad")
        self._get_val_grad_fn.prepare(("P", "P", "P", "P", "P", np.float32, np.float32, np.int32), texrefs=[self._tex_field])
        self._chomp_smooth_val_grad_fn = self._mod.get_function("chomp_smooth_dist_grad")
        self._chomp_smooth_val_grad_fn.prepare(("P", "P", "P", "P", "P", np.float32, np.float32, np.float32, np.int32),
                                               texrefs=[self._tex_field])
        self._grid = (int(math.ceil(num_pos / 1024.0)), 1)  # there can be at most 1024 threads per block
        self._block = (min(1024, num_pos), 1, 1)
        # save other parameters
        self._num_pos = np.int32(num_pos)
        self._scale = np.float32(scale)
        self._delta = np.float32(delta)
        self._base_tf = base_tf

    def interpolate(self, tf_matrix):
        """
            Retrieve grid values at the set positions given that the transformation matrix is tf_matrix.
            ---------
            Arguments
            ---------
            tf_matrix, np array of shape (4, 4) - transformation matrix from position frame to grid frame
            -------
            Returns
            -------
            values, np array of shape (n,) - interpolated values at the set positions
                If a position is outside of the grid, the closest value is returned
                WARNING: the returned array is used for subsequent calls as well. You should copy
                    it, if you intend to keep the values.
        """
        tf_matrix = np.dot(self._base_tf, tf_matrix)
        if tf_matrix.dtype != np.float32:
            tf_matrix = tf_matrix.astype(np.float32)
        # copy matrix to gpu
        cuda.memcpy_htod(self._gpu_tf_matrix, tf_matrix.flatten()[:12])
        # execute function
        self._get_val_fn.prepared_call(self._grid, self._block, self._gpu_positions,
                                       self._gpu_values.gpudata, self._gpu_tf_matrix, self._scale,
                                       self._num_pos)
        # copy results to CPU
        # cuda.memcpy_dtoh(self._cpu_values, self._gpu_values)
        self._gpu_values.get(self._cpu_values)
        return self._cpu_values

    def gradient(self, tf_matrix):
        """
            Retrieve grid values and gradients for the given positions.
            ---------
            Arguments
            ---------
            tf_matrix, np array of shape (4, 4) - transformation matrix from position frame to grid frame
            -------
            Returns
            -------
            values, np array of shape (n,) - interpolated values at the given positions
                If a position is outside of the grid, the closest value is returned
            grad, np array of shape (n, 3) - gradients at the given positions. The gradients are w.r.t
                the x, y, z axis of the base frame given during construction (world frame).
            WARNING: both values and grad are reused for subsequent calls. You should copy the data, if
                you intend to keep it.
        """
        tf_matrix = np.dot(self._base_tf, tf_matrix)
        if tf_matrix.dtype != np.float32:
            tf_matrix = tf_matrix.astype(np.float32)
        # copy matrix to gpu
        cuda.memcpy_htod(self._gpu_tf_matrix, tf_matrix.flatten()[:12])
        # execute function
        self._get_val_grad_fn.prepared_call(self._grid, self._block, self._gpu_positions,
                                            self._gpu_values.gpudata, self._gpu_gradients, self._gpu_grad_dirs,
                                            self._gpu_tf_matrix, self._scale, self._delta, self._num_pos)
        # copy results to CPU
        self._gpu_values.get(self._cpu_values)
        cuda.memcpy_dtoh(self._cpu_gradients, self._gpu_gradients)
        return self._cpu_values, self._cpu_gradients

    def sum(self, tf_matrix):
        """
            Sum the values lying at the stored positions for the given transformation matrix.
            ---------
            Arguments
            ---------
            tf_matrix, np array of shape (4, 4) - transformation matrix from position frame to grid frame
            -------
            Returns
            -------
            sum, float - the sum of thes evalues
        """
        tf_matrix = np.dot(self._base_tf, tf_matrix)
        if tf_matrix.dtype != np.float32:
            tf_matrix = tf_matrix.astype(np.float32)
        # copy matrix to gpu
        cuda.memcpy_htod(self._gpu_tf_matrix, tf_matrix.flatten()[:12])
        # execute function
        self._get_val_fn.prepared_call(self._grid, self._block, self._gpu_positions,
                                       self._gpu_values.gpudata, self._gpu_tf_matrix, self._scale,
                                       self._num_pos)
        # sum values
        sum_val = gpuarray.sum(self._gpu_values, np.float32)
        # extract single value
        cpu_val = np.zeros(1, dtype=np.float32)
        sum_val.get(cpu_val)
        return cpu_val[0]

    def min(self, tf_matrix):
        """
            Return the minimal value given the transformation matrix.
            ---------
            Arguments
            ---------
            tf_matrix, np array of shape (4, 4) - transformation matrix from position frame to grid frame
            -------
            Returns
            -------
            min, float - minimal value
        """
        tf_matrix = np.dot(self._base_tf, tf_matrix)
        if tf_matrix.dtype != np.float32:
            tf_matrix = tf_matrix.astype(np.float32)
        # copy matrix to gpu
        cuda.memcpy_htod(self._gpu_tf_matrix, tf_matrix.flatten()[:12])
        # get values
        self._get_val_fn.prepared_call(self._grid, self._block, self._gpu_positions,
                                       self._gpu_values.gpudata, self._gpu_tf_matrix, self._scale,
                                       self._num_pos)
        # get min values
        min_val = gpuarray.min(self._gpu_values)
        # extract single value
        cpu_val = np.zeros(1, dtype=np.float32)
        min_val.get(cpu_val)
        return cpu_val[0]

    def chomps_smooth_distance(self, tf_matrix, eps):
        """
            Assuming the underlying data structure stores distances, return Chomp's smooth distance
            for the set positions. The distance function is defined as:
                        -d(x) + eps / 2             if d(x) < 0.0
                ds(x) =  1/(2eps)(d(x) - eps)^2      if 0 <= d(x) <= eps
                        0                           else
            ---------
            Arguments
            ---------
            tf_matrix, np array of shape (4, 4) - transformation matrix from position frame to grid frame
            -------
            Returns
            -------
            values, np array of shape (n,) - interpolated values at the given positions
                If a position is outside of the grid, the closest value is returned
            grad, np array of shape (n, 3) - gradients at the given positions. The gradients are w.r.t
                the x, y, z axis of the base frame given during construction (world frame).
            WARNING: both values and grad are reused for subsequent calls. You should copy the data, if
                you intend to keep it.
        """
        tf_matrix = np.dot(self._base_tf, tf_matrix)
        if tf_matrix.dtype != np.float32:
            tf_matrix = tf_matrix.astype(np.float32)
        # copy matrix to gpu
        cuda.memcpy_htod(self._gpu_tf_matrix, tf_matrix.flatten()[:12])
        # execute function
        self._chomp_smooth_val_grad_fn.prepared_call(self._grid, self._block, self._gpu_positions, self._gpu_values.gpudata,
                                                     self._gpu_gradients, self._gpu_grad_dirs, self._gpu_tf_matrix,
                                                     self._scale, self._delta, np.float32(eps), self._num_pos)
        # copy results to CPU
        # cuda.memcpy_dtoh(self._cpu_values, self._gpu_values)
        self._gpu_values.get(self._cpu_values)
        cuda.memcpy_dtoh(self._cpu_gradients, self._gpu_gradients)
        return self._cpu_values, self._cpu_gradients
