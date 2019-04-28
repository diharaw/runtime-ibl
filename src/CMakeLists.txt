cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(APPLE)
    add_executable(RuntimeIBL MACOSX_BUNDLE "main.cpp")
    set(MACOSX_BUNDLE_BUNDLE_NAME "com.dihara.ibl") 
else()
    add_executable(RuntimeIBL "main.cpp") 
endif()

target_link_libraries(RuntimeIBL dwSampleFramework)

set_property(TARGET RuntimeIBL PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/$(Configuration)")