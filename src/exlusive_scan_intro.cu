#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cstdio>

int main() {

  thrust::device_vector<int> in{1,2,3};
  thrust::device_vector<int>  out(3);

  std::cout << "in:  ";
  thrust::for_each(
  thrust::device,
    in.begin(),
    in.end(),
    [=]__host__ __device__(int &i) {printf("%d;", i);}
  );
  std::cout << std::endl;

  thrust::exclusive_scan(thrust::device,
    in.begin(),
    in.end(),
    out.begin(),
    0
    );

  std::cout << "out: ";
  thrust::for_each(
    thrust::device,
    out.begin(),
    out.end(),
    [=]__host__ __device__(int &i) {printf("%d;", i);}
  );
  std::cout << std::endl;

  thrust::exclusive_scan(thrust::device,
    in.begin(),
    in.end(),
    out.begin(),
    1,
    [=]__host__ __device__(int cur, int prev) { return cur * prev; }
    );

  std::cout << "out: ";
  thrust::for_each(
    thrust::device,
    out.begin(),
    out.end(),
    [=]__host__ __device__(int &i) {printf("%d;", i);}
  );
  std::cout << std::endl;
}