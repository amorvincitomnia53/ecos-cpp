# ecos-cpp

*Ecos-cpp* is a object-oriented modern-C++ wrapper of [ECOS(Embedded Conic Solver)](https://github.com/embotech/ecos), a lightweight Second-Order Cone Programming solver. Ecos-cpp uses [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for its interface and is meant to be more user-friendly than the original C interface.

## Second-order Cone Programming

Ecos-cpp solves the following optimization problem:

$$\begin{align}\arg\min_{x}\quad & c^T x\\
\mathrm{s.t.}\quad& h-Gx\in K\\
& Ax-b = 0\end{align}$$

where $K$ is a direct product of positive orthant $\mathbb{R}_+^p$, quadratic cone $\{(y, z) | \|z\|_2 \le y\}$ and exponential cone $\mathrm{closure}\{(x,y,z) | \exp\frac{x}{z} \le \frac{y}{z}, z>0\}$. 

## Prerequisite

* Eigen 3
* CMake
* A C++ compiler that can compile C++11 (test code uses C++17 features.) 

## Using the library through CMake

Write the following code in your ```CMakeLists.txt```:

```cmake
add_subdirectory(<path-to-this-directory>)

...

target_link_libraries(<your-target> PUBLIC|PRIVATE ecos-cpp)
```

\#include ```<ecos_cpp.hpp>``` in your c++ code.

## Building and running tests

```bash
mkdir build
cd build
cmake .. 
make ecos-cpp-test
./ecos-cpp-test 1 1 1
```

