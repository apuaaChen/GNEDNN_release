#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <cub/cub.cuh>

#define BLOCKS(N, T) (N + T - 1)/T


// L2 Normalization Training: Forward pass


template <typename scalar_t, unsigned int blockSize>
__global__ void fusedL2Normv2(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ weight, 
    const scalar_t* __restrict__ bias, 
    scalar_t* y, scalar_t* rv, scalar_t* sm, scalar_t* scale,
    unsigned int HW,  unsigned int numel, float eps, float momentum, unsigned int C
){
    unsigned int c = blockIdx.x;

    typedef cub::BlockReduce<scalar_t, blockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage TempStorage;
    __shared__ scalar_t sms;

    scalar_t smp = 0;
    scalar_t c_ = 0;

    for (unsigned int id = threadIdx.x; id < numel; id += blockSize){
        unsigned int n = id / HW;
        unsigned int hw = id % HW;
        unsigned int vid = (n * C + c) * HW + hw;
        scalar_t data = x[vid];
        scalar_t y = data * data - c_;
        scalar_t t = smp + y;
        c_ = (t - smp) - y;
        smp = t;
    }

    //__syncthreads();
    scalar_t smr = BlockReduce(TempStorage).Sum(smp);
    if (threadIdx.x == 0){
        smr = sqrtf(smr / numel);
        sms = weight[c] / (smr + eps);
        rv[c] = rv[c] * (1 - momentum) + momentum * smr;
        sm[c] = smr;
        scale[c] = sms;
    }
    __syncthreads();

    scalar_t sca = sms;
    scalar_t b = bias[c];

    for (unsigned int id = threadIdx.x; id < numel; id += blockSize){
        unsigned int n = id / HW;
        unsigned int hw = id % HW;
        unsigned int vid = (n * C + c) * HW + hw;
        y[vid] = x[vid] * sca + b;
    }
}



std::vector<torch::Tensor> fused_l2_norm_fv2_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rv,
    float momentum,
    float eps
){
    unsigned int N = x.size(0);
    unsigned int C = x.size(1);
    unsigned int H = x.size(2);
    unsigned int W = x.size(3);
    unsigned int numel = N * H * W;
    // unsigned int total = x.numel();
    // unsigned int n_stride = C * H * W;
    unsigned int HW = H * W;

    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto sm = torch::empty({1, C, 1, 1}, options);
    auto scale = torch::empty_like(sm);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "fusedl2normv2", ([&]{
        fusedL2Normv2<scalar_t, 1024><<<C, 1024>>>(
            x.data<scalar_t>(), weight.data<scalar_t>(),
            bias.data<scalar_t>(), y.data<scalar_t>(),
            rv.data<scalar_t>(), sm.data<scalar_t>(),
            scale.data<scalar_t>(),
            HW, numel, eps, momentum, C
        );
    }));

    return {x, y, sm, scale, rv};
}


// L2 Normalization Training: Backward pass
template <typename scalar_t, unsigned int blockSize>
__global__ void fusedL2Normb(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ weight, const scalar_t* sm,
    const scalar_t* __restrict__ scale, const float eps,
    scalar_t* grad_x, scalar_t* grad_weight, scalar_t* grad_bias,
    unsigned int HW, unsigned int W, unsigned int N, unsigned int numel, unsigned int C
){
    unsigned int c = blockIdx.x;
    scalar_t sca = scale[c];

    typedef cub::BlockReduce<scalar_t, blockSize> BlockReduce;
    typedef cub::BlockReduce<scalar_t, blockSize> BlockReduce2;
    __shared__ typename BlockReduce::TempStorage TempStorage;
    __shared__ typename BlockReduce::TempStorage TempStorage2;
    __shared__ scalar_t gsm;

    scalar_t grad_bias_p = 0;
    scalar_t grad_scale_p = 0;

    scalar_t c_bias = 0;
    scalar_t c_scale = 0;
    
    for (unsigned int id = threadIdx.x; id < numel; id += blockSize){
        unsigned int n = id / HW;
        unsigned int hw = id % HW;
        unsigned int vid = (n * C + c) * HW + hw;
        scalar_t data = grad_y[vid];

        scalar_t y_bias = data - c_bias;
        scalar_t t_bias = grad_bias_p + y_bias;
        c_bias = (t_bias - grad_bias_p) - y_bias;
        grad_bias_p = t_bias;

        scalar_t y_scale = data * x[vid] - c_scale;
        scalar_t t_scale = grad_scale_p + y_scale;
        c_scale = (t_scale - grad_scale_p) - y_scale;
        grad_scale_p = t_scale;
        //grad_bias_p += data;
        //grad_scale_p += data * x[vid];
    }
    scalar_t gscale = BlockReduce(TempStorage2).Sum(grad_scale_p);
    scalar_t gbias = BlockReduce(TempStorage).Sum(grad_bias_p);
    if (threadIdx.x == 0){
        grad_bias[c] = gbias;
        scalar_t m = sm[c];
        grad_weight[c] = gscale / (m + eps);
        gsm = -gscale * sca / (numel * (m + eps) * m);
    }
    __syncthreads();
    
    for (unsigned int id = threadIdx.x; id < numel; id += blockSize){
        unsigned int n = id / HW;
        unsigned int hw = id % HW;
        unsigned int vid = (n * C + c) * HW + hw;
        grad_x[vid] = grad_y[vid] * sca + x[vid] * gsm;
    }
}



std::vector<torch::Tensor> fused_l2_norm_bv2_cuda(
    torch::Tensor x,
    torch::Tensor grad_y,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor sm,
    float eps
){
    unsigned int N = x.size(0);
    unsigned int C = x.size(1);
    unsigned int H = x.size(2);
    unsigned int W = x.size(3);

    auto grad_x = torch::empty_like(x);
    auto grad_weight = torch::empty_like(weight);
    auto grad_bias = torch::empty_like(weight);

    unsigned int HW = H * W;
    unsigned int numel = N * H * W;

    AT_DISPATCH_FLOATING_TYPES(x.type(), 'fusedL2Normb', ([&]{
        fusedL2Normb<scalar_t, 1024><<<C, 1024>>>(
            x.data<scalar_t>(), grad_y.data<scalar_t>(),
            weight.data<scalar_t>(), sm.data<scalar_t>(),
            scale.data<scalar_t>(), eps,
            grad_x.data<scalar_t>(), grad_weight.data<scalar_t>(),
            grad_bias.data<scalar_t>(), 
            HW, W, N, numel, C
        );
    }));

    return {grad_x, grad_weight, grad_bias};
}


// CUDA kernel that gets the second order moment
template <typename scalar_t, unsigned int blockSize>
__global__ void second_moment_reduce(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ x,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> sm_partial,
    unsigned int H, unsigned int W, unsigned int N, unsigned int numel
){
    // The block is 2D, the first block id is the channel
    unsigned int c = blockIdx.x;

    typedef cub::BlockReduce<scalar_t, blockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage TempStorage;

    scalar_t partial_sum = 0;

    for (unsigned int id = blockIdx.y * blockDim.x + threadIdx.x; id < numel; id += gridDim.y * blockDim.x){
        unsigned int n = id / (H * W);
        unsigned int hw = id % (H * W);
        unsigned int h = hw / W;
        unsigned int w = hw % W;
        scalar_t data = x[n][c][h][w];
        partial_sum += data * data;
    }
    __syncthreads();
    scalar_t sum = BlockReduce(TempStorage).Sum(partial_sum);
    if(threadIdx.x == 0){
        sm_partial[c][blockIdx.y] = sum;
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void get_moment(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ sm_partial,
    scalar_t* sm, unsigned int numblock, unsigned int numel
){
    unsigned int c = blockIdx.x;

    typedef cub::BlockReduce<scalar_t, blockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage TempStorage;

    scalar_t partial_sum = 0;

    for (unsigned int id = threadIdx.x; id < numblock; id += blockDim.x){
        partial_sum += sm_partial[c][id];
    }
    __syncthreads();
    scalar_t sum = BlockReduce(TempStorage).Sum(partial_sum);
    if(threadIdx.x == 0){
        sm[c] = sqrtf(sum / numel);
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void get_scale(
    const scalar_t* __restrict__ sm, const scalar_t* __restrict__ weight,
    scalar_t* rv, scalar_t* scale, float eps, unsigned int C, float momentum
){
    for(unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < C; id += gridDim.x * blockDim.x){
        scalar_t s = sm[id];
        scale[id] = weight[id] / (s + eps);
        rv[id] = rv[id] * (1 - momentum) + momentum * s;
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void normalization(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ scale, const scalar_t* __restrict__ bias,
    scalar_t* y, unsigned int total, unsigned int HW, unsigned int C
){
    for(unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < total; id += gridDim.x * blockDim.x){
        unsigned int c = id / HW;
        c = c % C;
        y[id] = x[id] * scale[c] + bias[c];
    }
}


// CUDA kernel that gets the second order moment
template <typename scalar_t, unsigned int blockSize>
__global__ void normalizationv2(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ x,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> y,
    const scalar_t* __restrict__ scale, const scalar_t* __restrict__ bias,
    unsigned int HW, unsigned int W, unsigned int N, unsigned int numel
){
    // The block is 2D, the first block id is the channel
    unsigned int c = blockIdx.x;
    scalar_t sc = scale[c];
    scalar_t b = bias[c];

    for (unsigned int id = blockIdx.y * blockDim.x + threadIdx.x; id < numel; id += gridDim.y * blockDim.x){
        unsigned int n = id / HW;
        unsigned int hw = id % HW;
        unsigned int h = hw / W;
        unsigned int w = hw % W;
        y[n][c][h][w] = x[n][c][h][w] * sc + b;
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void normalizationv3(
    const scalar_t* __restrict__ x, scalar_t* y,
    const scalar_t* __restrict__ scale, const scalar_t* __restrict__ bias,
    unsigned int HW, unsigned int W, unsigned int N, unsigned int numel, unsigned int C
){
    unsigned int c = blockIdx.x;
    scalar_t sc = scale[c];
    scalar_t b = bias[c];
    
    for (unsigned int id = blockIdx.y * blockDim.x + threadIdx.x; id < numel; id += gridDim.y * blockDim.x){
        unsigned int n = id / HW;
        unsigned int hw = id % HW;
        unsigned int vid = (n * C + c) * HW + hw;
        y[vid] = x[vid] * sc + b;
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void normalizationv4(
    const scalar_t* __restrict__ x, scalar_t* y,
    const scalar_t* __restrict__ scale, const scalar_t* __restrict__ bias,
    unsigned int N, unsigned int total, unsigned int n_stride, unsigned int HW
){
    unsigned int c = blockIdx.x;
    scalar_t sc = scale[c];
    scalar_t b = bias[c];

    for (unsigned int n = c*HW; n < total; n += n_stride){
        for (unsigned int hw = threadIdx.x; hw < HW; hw += blockDim.x){
            unsigned int vid = n + hw;
            y[vid] = x[vid] * sc + b;
        }
    }
}



template <typename scalar_t, unsigned int blockSize>
__global__ void fusedL2Norm(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ weight, 
    const scalar_t* __restrict__ bias, 
    scalar_t* y, scalar_t* rv, scalar_t* sm, scalar_t* scale,
    unsigned int N, unsigned int total, unsigned int n_stride, unsigned int HW, 
    unsigned int numel, float eps, float momentum
){
    unsigned int c = blockIdx.x;

    typedef cub::BlockReduce<scalar_t, blockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage TempStorage;
    __shared__ scalar_t sms;

    scalar_t smp = 0;

    for (unsigned int n = c * HW; n < total; n += n_stride){
        for (unsigned int hw = threadIdx.x; hw < HW; hw += blockDim.x){
            unsigned int vid = n + hw;
            scalar_t data = x[vid];
            smp += data * data;
        }
    }
    //__syncthreads();
    scalar_t smr = BlockReduce(TempStorage).Sum(smp);
    if (threadIdx.x == 0){
        smr = sqrtf(smr / numel);
        sms = weight[c] / (smr + eps);
        rv[c] = rv[c] * (1 - momentum) + momentum * smr;
        sm[c] = smr;
        scale[c] = sms;
    }
    __syncthreads();

    scalar_t sca = sms;
    scalar_t b = bias[c];

    for (unsigned int n = c * HW; n < total; n += n_stride){
        for (unsigned int hw = threadIdx.x; hw < HW; hw += blockDim.x){
            unsigned int vid = n + hw;
            y[vid] = x[vid] * sca + b;
        }
    }
}



std::vector<torch::Tensor> fused_l2_norm_f_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rv,
    float momentum,
    float eps
){
    // Step 1: get the input feature size
    unsigned int N = x.size(0);
    unsigned int C = x.size(1);
    unsigned int H = x.size(2);
    unsigned int W = x.size(3);
    unsigned int numel = N * H * W;

    auto y = torch::empty_like(x);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    unsigned int numblock = BLOCKS(numel, 512);

    auto sm_partial = torch::empty({C, numblock}, options);
    auto sm = torch::empty({1, C, 1, 1}, options);
    auto scale = torch::empty_like(sm);

    dim3 gridSize = dim3(C, numblock);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "second_moment_reduce", ([&]{
        second_moment_reduce<scalar_t, 512><<<gridSize, 512>>>(
            x.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            sm_partial.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            H, W, N, numel
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(x.type(), "get_moment", ([&]{
        get_moment<scalar_t, 512><<<C, 512>>>(
            sm_partial.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            sm.data<scalar_t>(),
            numblock, numel
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(x.type(), "get_scale", ([&]{
        get_scale<scalar_t, 512><<<BLOCKS(C, 512), 512>>>(
            sm.data<scalar_t>(), weight.data<scalar_t>(), rv.data<scalar_t>(),
            scale.data<scalar_t>(),
            eps, C, momentum
        );
    }));

    unsigned int total = x.numel();

    dim3 gridSize2 = dim3(C, 1); //BLOCKS(numel, 512 * 16));
    unsigned int n_stride = C * H * W;
    unsigned int HW = H * W;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "normalizationv4", ([&]{
        normalizationv4<scalar_t, 512><<<gridSize2, 512>>>(
            x.data<scalar_t>(), y.data<scalar_t>(),
            scale.data<scalar_t>(), bias.data<scalar_t>(),
            N, total, n_stride, HW
        );
    }));


    /*
    AT_DISPATCH_FLOATING_TYPES(x.type(), "normalizationv3", ([&]{
        normalizationv3<scalar_t, 512><<<gridSize2, 512>>>(
            x.data<scalar_t>(), y.data<scalar_t>(),
            scale.data<scalar_t>(), bias.data<scalar_t>(),
            H*W, W, N, numel, C
        );
    }));

    
    AT_DISPATCH_FLOATING_TYPES(x.type(), "normalizationv2", ([&]{
        normalizationv2<scalar_t, 512><<<gridSize2, 512>>>(
            x.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            y.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            scale.data<scalar_t>(), bias.data<scalar_t>(),
            H*W, W, N, numel
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(x.type(), "normalization", ([&]{
        normalization<scalar_t, 512><<<BLOCKS(total, 512), 512>>>(
            x.data<scalar_t>(), scale.data<scalar_t>(), bias.data<scalar_t>(),
            y.data<scalar_t>(), total, H*W, C
        );
    }));
    */

    return {x, y, sm, scale, rv};
}


template <typename scalar_t, unsigned int blockSize>
__global__ void backward_reduce(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ x,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ grad_y,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits> __restrict__ grad_x,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_scale_partial,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> grad_bias_partial,
    const scalar_t* __restrict__ scale,
    unsigned int H, unsigned int W, unsigned int N, unsigned int numel
){
    // The block is 2D, the first block id is the channel
    unsigned int c = blockIdx.x;
    scalar_t sca = scale[c];

    typedef cub::BlockReduce<scalar_t, blockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage TempStorage;

    scalar_t partial_scale = 0;
    scalar_t partial_bias = 0;

    for (unsigned int id = blockIdx.y * blockDim.x + threadIdx.x; id < numel; id += gridDim.y * blockDim.x){
        unsigned int n = id / (H * W);
        unsigned int hw = id % (H * W);
        unsigned int h = hw / W;
        unsigned int w = hw % W;
        scalar_t data = grad_y[n][c][h][w];
        partial_scale += data * x[n][c][h][w];
        partial_bias += data;
        grad_x[n][c][h][w] = data * sca;
    }
    scalar_t gscale = BlockReduce(TempStorage).Sum(partial_scale);
    scalar_t gbias = BlockReduce(TempStorage).Sum(partial_bias);
    if(threadIdx.x == 0){
        grad_scale_partial[c][blockIdx.y] = gscale;
        grad_bias_partial[c][blockIdx.y] = gbias;
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void get_grad(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ grad_scale_partial,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits> __restrict__ grad_bias_partial,
    scalar_t* grad_scale, scalar_t* grad_bias, unsigned int numblock, unsigned int numel
){
    unsigned int c = blockIdx.x;

    typedef cub::BlockReduce<scalar_t, blockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage TempStorage;

    scalar_t partial_scale = 0;
    scalar_t partial_bias = 0;

    for (unsigned int id = threadIdx.x; id < numblock; id += blockDim.x){
        partial_scale += grad_scale_partial[c][id];
        partial_bias += grad_bias_partial[c][id];
    }
    __syncthreads();
    scalar_t scale = BlockReduce(TempStorage).Sum(partial_scale);
    scalar_t bias = BlockReduce(TempStorage).Sum(partial_bias);
    if(threadIdx.x == 0){
        grad_scale[c] = scale;
        grad_bias[c] = bias;
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void process_grad_scale(
    const scalar_t* __restrict__ grad_scale, const scalar_t* __restrict__ weight,
    const scalar_t* sm, const scalar_t* scale,
    scalar_t* grad_weight, scalar_t* grad_sm, float eps, unsigned int C
){
    for(unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < C; id += gridDim.x * blockDim.x){
        scalar_t gs = grad_scale[id];
        scalar_t m = sm[id];
        grad_weight[id] = gs / (m + eps);
        grad_sm[id] = -gs * scale[id] / (2 * (m + eps) * m);
    }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void moment_grad(
    scalar_t* __restrict__ grad_x, const scalar_t* __restrict__ grad_sm, const scalar_t* __restrict__ x,
    unsigned int total, unsigned int HW, unsigned int C, unsigned int numel
){
    for(unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < total; id += gridDim.x * blockDim.x){
        unsigned int c = id / HW;
        c = c % C;
        grad_x[id] += x[id] * 2 / numel * grad_sm[c];
    }
}


std::vector<torch::Tensor> fused_l2_norm_b_cuda(
    torch::Tensor x,
    torch::Tensor grad_y,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor sm,
    float eps
){
    unsigned int N = x.size(0);
    unsigned int C = x.size(1);
    unsigned int H = x.size(2);
    unsigned int W = x.size(3);
    unsigned int numel = N * H * W;

    auto grad_x = torch::empty_like(x);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    unsigned int numblock = BLOCKS(numel, 512);

    auto grad_scale_partial = torch::empty({C, numblock}, options);
    auto grad_bias_partial = torch::empty({C, numblock}, options);

    auto grad_scale = torch::empty({1, C, 1, 1}, options);
    auto grad_bias = torch::empty({C,}, options);
    auto grad_sm = torch::empty_like(grad_bias);
    auto grad_weight = torch::empty_like(grad_bias);

    dim3 gridSize = dim3(C, numblock);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "backward_reduce", ([&]{
        backward_reduce<scalar_t, 512><<<gridSize, 512>>>(
            x.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_y.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_x.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_scale_partial.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_bias_partial.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            scale.data<scalar_t>(),
            H, W, N, numel
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(x.type(), "get_grad", ([&]{
        get_grad<scalar_t, 512><<<C, 512>>>(
            grad_scale_partial.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_bias_partial.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_scale.data<scalar_t>(), grad_bias.data<scalar_t>(),
            numblock, numel
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(x.type(), "process_grad_scale", ([&]{
        process_grad_scale<scalar_t, 512><<<BLOCKS(C, 512), 512>>>(
            grad_scale.data<scalar_t>(), weight.data<scalar_t>(),
            sm.data<scalar_t>(), scale.data<scalar_t>(), grad_weight.data<scalar_t>(),
            grad_sm.data<scalar_t>(), eps, C
        );
    }));

    unsigned int total = x.numel();

    AT_DISPATCH_FLOATING_TYPES(x.type(), "moment_grad", ([&]{
        moment_grad<scalar_t, 512><<<BLOCKS(total, 512), 512>>>(
            grad_x.data<scalar_t>(), grad_sm.data<scalar_t>(), x.data<scalar_t>(),
            total, H*W, C, numel
        );
    }));

    return {grad_x, grad_weight, grad_bias};

}



template <typename scalar_t, unsigned int blockSize>
__global__ void get_scale_inf(
    const scalar_t* __restrict__ sm, const scalar_t* __restrict__ weight,
    scalar_t* scale, float eps, unsigned int C
){
    for(unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; id < C; id += gridDim.x * blockDim.x){
        scale[id] = weight[id] / (sm[id] + eps);
    }
}



torch::Tensor fused_l2_norm_inf_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor sm,
    torch::Tensor bias,
    float eps
){
    unsigned int C = x.size(1);
    unsigned int H = x.size(2);
    unsigned int W = x.size(3);

    auto y = torch::empty_like(x);
    auto scale = torch::empty_like(sm);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "get_scale", ([&]{
        get_scale_inf<scalar_t, 512><<<BLOCKS(C, 512), 512>>>(
            sm.data<scalar_t>(), weight.data<scalar_t>(),
            scale.data<scalar_t>(), eps, C
        );
    }));

    unsigned int total = x.numel();

    AT_DISPATCH_FLOATING_TYPES(x.type(), "normalization", ([&]{
        normalization<scalar_t, 512><<<BLOCKS(total, 512), 512>>>(
            x.data<scalar_t>(), scale.data<scalar_t>(), bias.data<scalar_t>(),
            y.data<scalar_t>(), total, H*W, C
        );
    }));
    
    return y;
}