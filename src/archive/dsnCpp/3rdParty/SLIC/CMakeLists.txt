
SET(slic_SOURCES
		SLIC.cpp SLIC.h
)

ADD_LIBRARY(slic_Lib ${slic_SOURCES} )

INCLUDE_DIRECTORIES(	
	${PROJECT_SOURCE_DIR}/src/common/rbpl	
	${PROJECT_SOURCE_DIR}/src/common/rbpImg		
	${PROJECT_SOURCE_DIR}/src/mamaWF_Lib
	${PROJECT_SOURCE_DIR}/src/mamaWS_Lib
	${PROJECT_SOURCE_DIR}/src/DadaTools/LBFGS
	)

TARGET_LINK_LIBRARIES(slic_Lib rbpl ${Mama_LIBS})
