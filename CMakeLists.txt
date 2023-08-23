cmake_minimum_required(VERSION 3.27)
project(cartpole_autodiff)

# set release mode
set(CMAKE_BUILD_TYPE Release)

# set the path to the autodiff library
set(AUTODIFF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libautodiff)
find_package(autodiff REQUIRED PATHS ${AUTODIFF_DIR}/lib/cmake/autodiff)
find_package(Torch REQUIRED PATHS libtorch/share/cmake/Torch)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# add the executable
add_executable(cartpole_autodiff main.cpp)

# link the autodiff library
target_link_libraries(cartpole_autodiff autodiff::autodiff "${TORCH_LIBRARIES}")
set_property(TARGET cartpole_autodiff PROPERTY CXX_STANDARD 17)
