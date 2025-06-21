#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <iostream>
#include <chrono>

using Matrix = thrust::tuple<int, int, int, int>;

int main() {

  // https://www.nayuki.io/page/fast-fibonacci-algorithms gives 7 654 000 000 ns for N = 3981072
  // we archieve 914975ns
  // For N = 100,000,000 17 ms
  const int N = 100000000;
  const int MOD = 9837;

  Matrix fibonacci{thrust::make_tuple(1, 1, 1, 0)};
  Matrix identity_matrix{thrust::make_tuple(1, 0, 0, 1)};
  thrust::device_vector<Matrix> in(N, fibonacci);
  thrust::device_vector<Matrix> out(N);

  auto matmul_lambda = [=]__host__ __device__(const Matrix &x, const Matrix &y) {
    int a00 = thrust::get<0>(x); int a01 = thrust::get<1>(x);
    int a10 = thrust::get<2>(x); int a11 = thrust::get<3>(x);
    int b00 = thrust::get<0>(y); int b01 = thrust::get<1>(y);
    int b10 = thrust::get<2>(y); int b11 = thrust::get<3>(y);

    int c00 = (a00 * b00 % MOD + a01 * b10 % MOD) % MOD;
    int c01 = (a00 * b01 % MOD + a01 * b11 % MOD) % MOD;
    int c10 = (a10 * b00 % MOD + a11 * b10 % MOD) % MOD;
    int c11 = (a10 * b01 % MOD + a11 * b11 % MOD) % MOD;

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
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << duration.count() << " milliseconds" << std::endl;


  std::cout << "Last element of out (F(" << N-1 << ") mod " << MOD << "):" << std::endl;
  Matrix last_matrix = out[N - 1];
  printf("%d\n", thrust::get<1>(last_matrix));
}