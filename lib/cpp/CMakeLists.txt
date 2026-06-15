cmake_minimum_required(VERSION 3.16)
project(fhe_dps CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)   # HE is far too slow in Debug
endif()

set(LIB_CPP_DIR        ${CMAKE_CURRENT_SOURCE_DIR})
set(REPO_ROOT          ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(OPENFHE_SOURCE_DIR ${REPO_ROOT}/extern/openfhe)
set(NLOHMANN_INCLUDE   ${REPO_ROOT}/extern/nlohmann/single_include)

add_subdirectory(${OPENFHE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/openfhe)

add_executable(fhe_dps_test ${LIB_CPP_DIR}/matrix_inference_sample.cpp)
target_include_directories(fhe_dps_test PUBLIC
    ${LIB_CPP_DIR}
    ${OPENFHE_SOURCE_DIR}/src/pke/include
    ${OPENFHE_SOURCE_DIR}/src/core/include
    ${OPENFHE_SOURCE_DIR}/src/binfhe/include
    ${OPENFHE_SOURCE_DIR}/third-party/cereal/include
    ${CMAKE_BINARY_DIR}/openfhe/src/core       # generated config_core.h
    ${NLOHMANN_INCLUDE}
)
target_link_libraries(fhe_dps_test PUBLIC OPENFHEpke_static OPENFHEcore_static)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fhe_dps_test PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(infer_plaintext ${LIB_CPP_DIR}/infer_plaintext.cpp)
target_include_directories(infer_plaintext PUBLIC
    ${LIB_CPP_DIR}
    ${NLOHMANN_INCLUDE}
)
