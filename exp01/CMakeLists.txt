cmake_minimum_required(VERSION 3.16)
project(fhe_dps CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)   # HE is far too slow in Debug
endif()

#find_package(OpenFHE REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/../extern/openfhe ${CMAKE_BINARY_DIR}/openfhe)
add_executable(fhe_dps_test fhe_server.cpp)

target_include_directories(fhe_dps_test PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/core
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(fhe_dps_test PRIVATE OPENFHEpke OPENFHEcore)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fhe_dps_test PRIVATE OpenMP::OpenMP_CXX)
endif()
