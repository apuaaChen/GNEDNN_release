#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_l2_norm_f_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rv,
    float momentum,
    float eps
);

std::vector<torch::Tensor> fused_l2_norm_f(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rv,
    float momentum,
    float eps
){
    return fused_l2_norm_f_cuda(x, weight, bias, rv, momentum, eps);
}


std::vector<torch::Tensor> fused_l2_norm_fv2_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rv,
    float momentum,
    float eps
);

std::vector<torch::Tensor> fused_l2_norm_fv2(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rv,
    float momentum,
    float eps
){
    return fused_l2_norm_fv2_cuda(x, weight, bias, rv, momentum, eps);
}


std::vector<torch::Tensor> fused_l2_norm_b_cuda(
    torch::Tensor x,
    torch::Tensor grad_y,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor sm,
    float eps
);


std::vector<torch::Tensor> fused_l2_norm_b(
    torch::Tensor x,
    torch::Tensor grad_y,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor sm,
    float eps
){
    return fused_l2_norm_b_cuda(x, grad_y, scale, weight, sm, eps);
}


std::vector<torch::Tensor> fused_l2_norm_bv2_cuda(
    torch::Tensor x,
    torch::Tensor grad_y,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor sm,
    float eps
);


std::vector<torch::Tensor> fused_l2_norm_bv2(
    torch::Tensor x,
    torch::Tensor grad_y,
    torch::Tensor scale,
    torch::Tensor weight,
    torch::Tensor sm,
    float eps
){
    return fused_l2_norm_bv2_cuda(x, grad_y, scale, weight, sm, eps);
}



torch::Tensor fused_l2_norm_inf_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor sm,
    torch::Tensor bias,
    float eps
);


torch::Tensor fused_l2_norm_inf(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor sm,
    torch::Tensor bias,
    float eps
){
    return fused_l2_norm_inf_cuda(x, weight, sm, bias, eps);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("fused_l2_norm_f", &fused_l2_norm_f, "fused L2 Norm forward");
    m.def("fused_l2_norm_fv2", &fused_l2_norm_fv2, "fused L2 Norm forward v2");
    m.def("fused_l2_norm_b", &fused_l2_norm_b, "fused L2 Norm backward");
    m.def("fused_l2_norm_bv2", &fused_l2_norm_bv2, "fused L2 Norm backward v2");
    m.def("fused_l2_norm_inf", &fused_l2_norm_inf, "fused L2 Norm inference");
}