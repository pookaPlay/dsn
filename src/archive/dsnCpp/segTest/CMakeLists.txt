
ADD_EXECUTABLE(segTest mainSegTest.cpp ) 
TARGET_LINK_LIBRARIES(segTest dadaSeg dadaCore ${dada_LIBS})

ADD_EXECUTABLE(segCC mainSegCC.cpp ) 
TARGET_LINK_LIBRARIES(segCC dadaSeg dadaCore ${dada_LIBS})

ADD_EXECUTABLE(segInt mainSegInt.cpp ) 
TARGET_LINK_LIBRARIES(segInt dadaSeg dadaCore ${dada_LIBS})
