#include <cuda.h>

# define MEM_IDX(X,Y,Z) (X)*data_y*data_z+(Y)*data_z+(Z)
# define OCC_MEM_IDX(X,Y,Z) (X+1)*(data_y+2)*(data_z+2)+(Y+1)*(data_z+2)+Z+1

texture<float, cudaTextureType3D, cudaReadModeElementType> tex_field;
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> tex_field_layered;

__device__ float3 rotate(const float3& pos, const float* const tf_matrix) {
    float3 outpos = make_float3(0.0, 0.0, 0.0);
    outpos.x = tf_matrix[0] * pos.x + tf_matrix[1] * pos.y + tf_matrix[2] * pos.z;
    outpos.y = tf_matrix[4] * pos.x + tf_matrix[5] * pos.y + tf_matrix[6] * pos.z;
    outpos.z = tf_matrix[8] * pos.x + tf_matrix[9] * pos.y + tf_matrix[10] * pos.z;
    return outpos;
}

__device__ float3 transform(const float3& pos, const float* const tf_matrix) {
    float3 outpos = make_float3(0.0, 0.0, 0.0);
    outpos.x = tf_matrix[0] * pos.x + tf_matrix[1] * pos.y + tf_matrix[2] * pos.z + tf_matrix[3];
    outpos.y = tf_matrix[4] * pos.x + tf_matrix[5] * pos.y + tf_matrix[6] * pos.z + tf_matrix[7];
    outpos.z = tf_matrix[8] * pos.x + tf_matrix[9] * pos.y + tf_matrix[10] * pos.z + tf_matrix[11];
    return outpos;
}

__device__ float read_val(const float3& pos, float scale) {
    // transform to cell indices and read texture + 1.5 because of padding cells and center of cell
    // return tex3D(tex_field, scale * pos.x + 1.5, scale * pos.y + 1.5, scale * pos.z + 1.5);
    // return tex3D(tex_field, scale * pos.z + 1.5, scale * pos.y + 1.5, scale * pos.x + 1.5);
    return tex3D(tex_field, scale * pos.x + 1.5, scale * pos.y + 1.5, scale * pos.z + 1.5);
}

__device__ float read_val_layered(const float3& pos, float scale) {
    // transform to cell indices and read texture + 1.5 because of padding cells and center of cell
    int iz = __float2int_rd(pos.z) + 1;
    return tex2DLayered(tex_field_layered, scale * pos.x + 1.5, scale * pos.y + 1.5, iz);
}

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(float val, const float3& vec) {
    return make_float3(val * vec.x, val * vec.y, val * vec.z);
}

__global__ void get_val(const float* positions, float* out_values, const float* const tf_matrix, float scale, int num_positions)
{
    int position_id = blockIdx.x * blockDim.x + threadIdx.x;
    // check whether this thread should do anything at all
    if (position_id < num_positions) {
        float ltf[12];
        ltf[0] = tf_matrix[0]; ltf[1] = tf_matrix[1]; ltf[2] = tf_matrix[2]; ltf[3] = tf_matrix[3];
        ltf[4] = tf_matrix[4]; ltf[5] = tf_matrix[5]; ltf[6] = tf_matrix[6]; ltf[7] = tf_matrix[7];
        ltf[8] = tf_matrix[8]; ltf[9] = tf_matrix[9]; ltf[10] = tf_matrix[10]; ltf[11] = tf_matrix[11];
        // first transform position to local frame
        float3 pos = make_float3(positions[3 * position_id], positions[3 * position_id + 1], positions[3 * position_id + 2]);
        float3 tpos = transform(pos, ltf);
        out_values[position_id] = read_val(tpos, scale);
    }
}

__global__ void get_val_layered(const float* positions, float* out_values, const float* const tf_matrix, float scale, int num_positions)
{
    int position_id = blockIdx.x * blockDim.x + threadIdx.x;
    // check whether this thread should do anything at all
    if (position_id < num_positions) {
        float ltf[12];
        ltf[0] = tf_matrix[0]; ltf[1] = tf_matrix[1]; ltf[2] = tf_matrix[2]; ltf[3] = tf_matrix[3];
        ltf[4] = tf_matrix[4]; ltf[5] = tf_matrix[5]; ltf[6] = tf_matrix[6]; ltf[7] = tf_matrix[7];
        ltf[8] = tf_matrix[8]; ltf[9] = tf_matrix[9]; ltf[10] = tf_matrix[10]; ltf[11] = tf_matrix[11];
        // first transform position to local frame
        float3 pos = make_float3(positions[3 * position_id], positions[3 * position_id + 1], positions[3 * position_id + 2]);
        float3 tpos = transform(pos, ltf);
        out_values[position_id] = read_val_layered(tpos, scale);
    }
}

__global__ void get_val_and_grad(const float* const positions, float* out_values, float* out_grads,
                                 const float* const tf_matrix, float scale,
                                 float delta, int num_positions)
{
    int position_id = blockIdx.x * blockDim.x + threadIdx.x;
    // check whether this thread should do anything at all
    if (position_id < num_positions) {
        float ltf[12];
        ltf[0] = tf_matrix[0]; ltf[1] = tf_matrix[1]; ltf[2] = tf_matrix[2]; ltf[3] = tf_matrix[3];
        ltf[4] = tf_matrix[4]; ltf[5] = tf_matrix[5]; ltf[6] = tf_matrix[6]; ltf[7] = tf_matrix[7];
        ltf[8] = tf_matrix[8]; ltf[9] = tf_matrix[9]; ltf[10] = tf_matrix[10]; ltf[11] = tf_matrix[11];
        // first transform position to local frame
        float3 pos = make_float3(positions[3 * position_id], positions[3 * position_id + 1], positions[3 * position_id + 2]);
        float3 tpos = transform(pos, ltf);
        // transform gradient direction as well
        float3 xgrad = make_float3(ltf[0], ltf[4], ltf[8]);
        float3 ygrad = make_float3(ltf[1], ltf[5], ltf[9]);
        float3 zgrad = make_float3(ltf[2], ltf[6], ltf[10]);
        // read value and compute gradient
        out_values[position_id] = read_val(tpos, scale);
        out_grads[3 * position_id] = (read_val(tpos + delta * xgrad, scale) - read_val(tpos - delta * xgrad, scale)) / (2.0 * delta);
        out_grads[3 * position_id + 1] = (read_val(tpos + delta * ygrad, scale) - read_val(tpos - delta * ygrad, scale)) / (2.0 * delta);
        out_grads[3 * position_id + 2] = (read_val(tpos + delta * zgrad, scale) - read_val(tpos - delta * zgrad, scale)) / (2.0 * delta);
    }
}

__global__ void get_val_and_grad_layered(const float* const positions, float* out_values, float* out_grads,
                                 const float* const tf_matrix, float scale,
                                 float delta, int num_positions)
{
    // TODO implement me
    // int position_id = blockIdx.x * blockDim.x + threadIdx.x;
    // // check whether this thread should do anything at all
    // if (position_id < num_positions) {
    //     float ltf[12];
    //     ltf[0] = tf_matrix[0]; ltf[1] = tf_matrix[1]; ltf[2] = tf_matrix[2]; ltf[3] = tf_matrix[3];
    //     ltf[4] = tf_matrix[4]; ltf[5] = tf_matrix[5]; ltf[6] = tf_matrix[6]; ltf[7] = tf_matrix[7];
    //     ltf[8] = tf_matrix[8]; ltf[9] = tf_matrix[9]; ltf[10] = tf_matrix[10]; ltf[11] = tf_matrix[11];
    //     // first transform position to local frame
    //     float3 pos = make_float3(positions[3 * position_id], positions[3 * position_id + 1], positions[3 * position_id + 2]);
    //     float3 tpos = transform(pos, ltf);
    //     // transform gradient direction as well
    //     float3 xgrad = make_float3(ltf[0], ltf[4], ltf[8]);
    //     float3 ygrad = make_float3(ltf[1], ltf[5], ltf[9]);
    //     float3 zgrad = make_float3(ltf[2], ltf[6], ltf[10]);
    //     // read value and compute gradient
    //     out_values[position_id] = read_val(tpos, scale);
    //     out_grads[3 * position_id] = (read_val(tpos + delta * xgrad, scale) - read_val(tpos - delta * xgrad, scale)) / (2.0 * delta);
    //     out_grads[3 * position_id + 1] = (read_val(tpos + delta * ygrad, scale) - read_val(tpos - delta * ygrad, scale)) / (2.0 * delta);
    //     out_grads[3 * position_id + 2] = (read_val(tpos + delta * zgrad, scale) - read_val(tpos - delta * zgrad, scale)) / (2.0 * delta);
    // }
}

__global__ void chomp_smooth_dist_grad(const float* const positions, float* out_values, float* out_grads,
                                 const float* const tf_matrix, float scale, float delta, float eps, int num_positions)
{
    int position_id = blockIdx.x * blockDim.x + threadIdx.x;
    // check whether this thread should do anything at all
    if (position_id < num_positions) {
        float ltf[12];
        ltf[0] = tf_matrix[0]; ltf[1] = tf_matrix[1]; ltf[2] = tf_matrix[2]; ltf[3] = tf_matrix[3];
        ltf[4] = tf_matrix[4]; ltf[5] = tf_matrix[5]; ltf[6] = tf_matrix[6]; ltf[7] = tf_matrix[7];
        ltf[8] = tf_matrix[8]; ltf[9] = tf_matrix[9]; ltf[10] = tf_matrix[10]; ltf[11] = tf_matrix[11];
        // first transform position to local frame
        float3 pos = make_float3(positions[3 * position_id], positions[3 * position_id + 1], positions[3 * position_id + 2]);
        float3 tpos = transform(pos, ltf);
        // transform gradient direction as well
        float3 xgrad = make_float3(ltf[0], ltf[4], ltf[8]);
        float3 ygrad = make_float3(ltf[1], ltf[5], ltf[9]);
        float3 zgrad = make_float3(ltf[2], ltf[6], ltf[10]);
        // read value and compute gradient
        float val = read_val(tpos, scale);
        float gx = (read_val(tpos + delta * xgrad, scale) - read_val(tpos - delta * xgrad, scale)) / (2.0 * delta);
        float gy = (read_val(tpos + delta * ygrad, scale) - read_val(tpos - delta * ygrad, scale)) / (2.0 * delta);
        float gz = (read_val(tpos + delta * zgrad, scale) - read_val(tpos - delta * zgrad, scale)) / (2.0 * delta);
        if (val < 0.0) {
            out_values[position_id] = -val + eps / 2.0;
            out_grads[3 * position_id] = -gx;
            out_grads[3 * position_id + 1] = -gy;
            out_grads[3 * position_id + 2] = -gz;
        } else if (val <= eps) {
            out_values[position_id] = 1.0 / (2.0 * eps) * (val - eps)* (val - eps);
            out_grads[3 * position_id] = gx * 1.0 / eps * (val - eps);
            out_grads[3 * position_id + 1] = gy * 1.0 / eps * (val - eps);
            out_grads[3 * position_id + 2] = gz * 1.0 / eps * (val - eps);
        } else {
            out_values[position_id] = 0.0;
            out_grads[3 * position_id] = 0.0;
            out_grads[3 * position_id + 1] = 0.0;
            out_grads[3 * position_id + 2] = 0.0;
        }
    }
}