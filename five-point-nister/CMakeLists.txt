find_package( OpenCV 2 REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
add_library( five-point-nister SHARED
    five-point.cpp precomp.cpp modelest.cpp )

#target_link_libraries(five-point-nister
#    ${OpenCV_LIBS} )
