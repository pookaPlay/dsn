
ADD_EXECUTABLE(mamaVectorize mamaVectorizeMain.cpp ) 
TARGET_LINK_LIBRARIES(mamaVectorize dadaSeg dadaCore ${dada_LIBS})
