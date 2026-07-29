# exp04-mnist's CKKS inference binaries.
#
# To be included via add_subdirectory() from lib/cpp/CMakeLists.txt.
#
# Inherited from parent CMakeLists:
# - REPO_ROOT
# - OPENFHE_SOURCE_DIR
# - NLOHMANN_INCLUDE
# - LIB_CPP_DIR

set(EXP04_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)

add_executable(infer_cnn ${EXP04_SRC_CPP}/infer_cnn.cpp)

target_include_directories(infer_cnn PUBLIC
    ${LIB_CPP_DIR}                              # halevi_shoup.h
    ${OPENFHE_SOURCE_DIR}/src/pke/include
    ${OPENFHE_SOURCE_DIR}/src/core/include
    ${OPENFHE_SOURCE_DIR}/src/binfhe/include
    ${OPENFHE_SOURCE_DIR}/third-party/cereal/include
    ${CMAKE_BINARY_DIR}/openfhe/src/core       # generated config_core.h
    ${NLOHMANN_INCLUDE}
)
target_link_libraries(infer_cnn PUBLIC OPENFHEpke_static OPENFHEcore_static)
if(OpenMP_CXX_FOUND)
    target_link_libraries(infer_cnn PUBLIC OpenMP::OpenMP_CXX)
endif()
