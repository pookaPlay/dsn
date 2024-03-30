
add_executable(InterpolateSfrcImage InterpolateSfrcOnImage.cxx )
 
target_link_libraries(InterpolateSfrcImage dadaCore ${VTK_LIBRARIES} ${CORE_LIBS})

add_executable(InterpolateSoutImage InterpolateSoutOnImage.cxx )
 
target_link_libraries(InterpolateSoutImage dadaCore ${VTK_LIBRARIES} ${CORE_LIBS})
