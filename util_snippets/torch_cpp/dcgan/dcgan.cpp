
#include <torch/torch.h>
#include <iostream>

/* build command
cmake .. && cmake --build . --config Release
*/

struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M) {
    W = register_parameter("W", torch::randn({N, M}));
    b = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return torch::addmm(b, input, W);
  }
  torch::Tensor W, b;
};

int main(){
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor<< std::endl;
    return 0;
}

// struct Net : torch::nn::Module {
//   Net(int64_t N, int64_t M)
//       : linear(register_module("linear", torch::nn::Linear(N, M))) {
//     another_bias = register_parameter("b", torch::randn(M));
//   }
//   torch::Tensor forward(torch::Tensor input) {
//     return linear(input) + another_bias;
//   }
//   torch::nn::Linear linear;
//   torch::Tensor another_bias;
// };
// int main(){
//     Net net(4, 5);
//     for(const auto& p : net.parameters()) {
//         std::cout << p << std::endl;
//     }
// }

// int main(){
//   auto chkpt = torch::load("/tmp/dummy_model.pt");
//   return 0;
// }
