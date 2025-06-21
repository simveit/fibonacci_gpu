#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <iostream>
#include <chrono>

// 2 x 2 Matrix
// a00, a01, a10, a11
using Matrix = thrust::tuple<int, int, int, int>;

int main() {

  const int N = 32;

  Matrix fibonacci{thrust::make_tuple(1, 1, 1, 0)};
  Matrix identity_matrix{thrust::make_tuple(1, 0, 0, 1)};
  thrust::device_vector<Matrix> in(N, fibonacci);
  thrust::device_vector<Matrix> out(N);

  auto print_lambda = [=]__host__ __device__(Matrix &x) {
    printf("[%d, %d, %d, %d]\n",
      thrust::get<0>(x), thrust::get<1>(x),
      thrust::get<2>(x), thrust::get<3>(x)
      );
  };
  std::cout << "in:" << std::endl;
  thrust::for_each(
    thrust::device,
    in.begin(),
    in.end(),
    print_lambda
  );
  std::cout << std::endl;

  // Mult the current matrix from left to the previous one as scan op.
  auto matmul_lambda = [=]__host__ __device__(const Matrix &x, const Matrix &y) {
    auto a00 = thrust::get<0>(x); auto a01 = thrust::get<1>(x);
    auto a10 = thrust::get<2>(x); auto a11 = thrust::get<3>(x);
    auto b00 = thrust::get<0>(y); auto b01 = thrust::get<1>(y);
    auto b10 = thrust::get<2>(y); auto b11 = thrust::get<3>(y);

    auto c00 = a00 * b00 + a01 * b10;
    auto c01 = a00 * b01 + a01 * b11;
    auto c10 = a10 * b00 + a11 * b10;
    auto c11 = a10 * b01 + a11 * b11;

    return thrust::make_tuple(
      c00,
      c01,
      c10,
      c11
    );
  };

  auto start = std::chrono::steady_clock::now();
  thrust::exclusive_scan(thrust::device,
    in.begin(),
    in.end(),
    out.begin(),
    identity_matrix,
    matmul_lambda
    );
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  std::cout << duration.count() << "ns" << std::endl;


  std::cout << "out:" << std::endl;
  thrust::for_each(
    thrust::device,
    out.begin(),
    out.end(),
    [=]__host__ __device__(Matrix &x) {printf("%d;", thrust::get<1>(x));}
  );
  std::cout << std::endl;
}