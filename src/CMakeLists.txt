cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_program(CLANG_FORMAT_EXE NAMES "clang-format" DOC "Path to clang-format executable")

file(GLOB_RECURSE IBL_HEADERS ${PROJECT_SOURCE_DIR}/src/*.h)
file(GLOB_RECURSE IBL_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE IBL_SHADERS ${PROJECT_SOURCE_DIR}/src/*.glsl)

if(APPLE)
    add_executable(RuntimeIBL MACOSX_BUNDLE ${IBL_HEADERS} ${IBL_SOURCES})
    set(MACOSX_BUNDLE_BUNDLE_NAME "com.dihara.ibl") 
else()
    add_executable(RuntimeIBL ${IBL_HEADERS} ${IBL_SOURCES}) 
endif()

target_link_libraries(RuntimeIBL dwSampleFramework)

if (APPLE)
    add_custom_command(TARGET RuntimeIBL POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shader $<TARGET_FILE_DIR:RuntimeIBL>/RuntimeIBL.app/Contents/Resources/assets/shader)
    add_custom_command(TARGET RuntimeIBL POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/mesh $<TARGET_FILE_DIR:RuntimeIBL>/RuntimeIBL.app/Contents/Resources/mesh)
    add_custom_command(TARGET RuntimeIBL POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/hdr $<TARGET_FILE_DIR:RuntimeIBL>/RuntimeIBL.app/Contents/Resources/hdr)
else()
    add_custom_command(TARGET RuntimeIBL POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/shader $<TARGET_FILE_DIR:RuntimeIBL>/shader)
    add_custom_command(TARGET RuntimeIBL POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/mesh $<TARGET_FILE_DIR:RuntimeIBL>/mesh)
    add_custom_command(TARGET RuntimeIBL POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/hdr $<TARGET_FILE_DIR:RuntimeIBL>/hdr)
endif()

if(CLANG_FORMAT_EXE)
    add_custom_target(clang-format-project-files COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${IBL_HEADERS} ${IBL_SOURCES} ${IBL_SHADERS})
endif()

set_property(TARGET RuntimeIBL PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$(Configuration)")