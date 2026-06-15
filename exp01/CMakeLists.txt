cmake_minimum_required(VERSION 3.16)
project(fhe_dps CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)   # HE is far too slow in Debug
endif()

#find_package(OpenFHE REQUIRED)

set(OPENFHE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../extern/openfhe)
add_subdirectory(${OPENFHE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/openfhe)
add_executable(fhe_dps_test src/cpp/matrix_inference_sample.cpp)

target_include_directories(fhe_dps_test PUBLIC
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${OPENFHE_SOURCE_DIR}/src/pke/include
    ${OPENFHE_SOURCE_DIR}/src/core/include
    ${OPENFHE_SOURCE_DIR}/src/binfhe/include
    ${OPENFHE_SOURCE_DIR}/third-party/cereal/include
    ${CMAKE_BINARY_DIR}/openfhe/src/core       # generated config_core.h
    ${CMAKE_SOURCE_DIR}/../extern/nlohmann/single_include
)

target_link_libraries(fhe_dps_test PUBLIC OPENFHEpke_static OPENFHEcore_static)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fhe_dps_test PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(infer_plaintext src/cpp/infer_plaintext.cpp)
target_include_directories(infer_plaintext PUBLIC
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/../extern/nlohmann/single_include
)
