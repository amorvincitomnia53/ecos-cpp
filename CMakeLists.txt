cmake_minimum_required(VERSION 3.05)
project(ecos-cpp C CXX)
set(CMAKE_CXX_STANDARD 11)

option(ECOS_USE_LONG "ecos - Whether to use Long or Int for index type" OFF)


FUNCTION(PREPEND var prefix)
   SET(listVar "")
   FOREACH(f ${ARGN})
      LIST(APPEND listVar "${prefix}/${f}")
   ENDFOREACH(f)
   SET(${var} "${listVar}" PARENT_SCOPE)
ENDFUNCTION(PREPEND)



PREPEND(AMD_CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/ecos/external/amd/src
        amd_aat.c amd_1.c amd_2.c amd_dump.c amd_postorder.c amd_post_tree.c
        amd_defaults.c amd_order.c amd_control.c amd_info.c amd_valid.c
        amd_preprocess.c amd_global.c)

PREPEND(LDL_CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/ecos/external/ldl/src
        ldl.c)

PREPEND(ECOS_CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/ecos/src
        ecos.c kkt.c cone.c spla.c ctrlc.c timer.c preproc.c splamm.c equil.c
        expcone.c wright_omega.c)

PREPEND(ECOS_BB_CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/ecos/ecos_bb
        ecos_bb.c ecos_bb_preproc.c)


add_library(ecos-c STATIC ${AMD_CPP_FILES} ${LDL_CPP_FILES} ${ECOS_CPP_FILES}
        #${ECOS_BB_CPP_FILES}
        )

target_include_directories(ecos-c PUBLIC ${CMAKE_CURRENT_LIST_DIR}/ecos/include)
target_include_directories(ecos-c PUBLIC ${CMAKE_CURRENT_LIST_DIR}/ecos/external/SuiteSparse_config
        ${CMAKE_CURRENT_LIST_DIR}/ecos/external/amd/include
        ${CMAKE_CURRENT_LIST_DIR}/ecos/external/ldl/include)

if(ECOS_USE_LONG)
    target_compile_options(ecos-c PUBLIC LDL_LONG DLONG)
endif()

add_library(ecos-cpp INTERFACE)
target_include_directories(ecos-cpp INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(ecos-cpp INTERFACE ecos-c)
if(TARGET eigen)
   target_link_libraries(ecos-cpp INTERFACE eigen)
else()
   message("Target 'eigen' is not defined. Trying to use default Eigen3 package.")
   find_package(Eigen3)
   target_link_libraries(ecos-cpp INTERFACE ${EIGEN3_INCLUDE_DIR})
endif()

add_executable(ecos-cpp-test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_LIST_DIR}/test/test.cpp)
target_link_libraries(ecos-cpp-test ecos-cpp)
target_compile_options(ecos-cpp-test PUBLIC -Wall -Wextra -DEIGEN_NO_DEBUG -DEIGEN_DONT_ALIGN_STATICALLY -DEIGEN_INITIALIZE_MATRICES_BY_NAN)

#CFLAGS += -O2 -Wall -DCTRLC=1 -Wextra -fPIC #-ansi -Werror #-ipo
