
SET(dadaSeg_SOURCES
	SegmentWS.cpp SegmentWS.h		
	SetElement.cpp SetElement.h		
	SegmentEval.cpp SegmentEval.h		
	SegmentParameter.cpp SegmentParameter.h			
	SegmentTrain.cpp SegmentTrain.h				
	SegmentVectorize.cpp SegmentVectorize.h				
	Morphology.cpp Morphology.h
	ThreshSegment_Int.cpp ThreshSegment_Int.h
)

ADD_LIBRARY(dadaSeg ${dadaSeg_SOURCES} )

TARGET_LINK_LIBRARIES(dadaSeg dadaCore ${dada_LIBS})
