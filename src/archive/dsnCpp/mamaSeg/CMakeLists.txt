
SET(mamaWS_Lib_SOURCES
	SegmentWS.cpp SegmentWS.h		
	SetElement.cpp SetElement.h		
	SegmentEval.cpp SegmentEval.h		
	SegmentParameter.cpp SegmentParameter.h			
	SegmentTrain.cpp SegmentTrain.h			
	ThreshSegment_Int.cpp ThreshSegment_Int.h		
	SegmentWSMarker.cpp SegmentWSMarker.h		
)

INCLUDE_DIRECTORIES(	
	${PROJECT_SOURCE_DIR}/src/common/rbpl	
	${PROJECT_SOURCE_DIR}/src/common/rbpImg	
	${PROJECT_SOURCE_DIR}/src/mamaWF_Lib
	)

ADD_LIBRARY(mamaWS_Lib ${mamaWS_Lib_SOURCES} )

TARGET_LINK_LIBRARIES(mamaWS_Lib rbpl rbpImg ${Mama_LIBS})
