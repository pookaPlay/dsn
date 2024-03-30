
SET (dadaCore_SOURCES  
	DadaDef.cpp DadaDef.h				
	GdalMat.cpp GdalMat.h	
	VizMat.cpp VizMat.h	
	ConnectedComponents.cpp ConnectedComponents.h
)

ADD_LIBRARY(dadaCore STATIC ${dadaCore_SOURCES}) 
TARGET_LINK_LIBRARIES(dadaCore ${dada_LIBS}) 

