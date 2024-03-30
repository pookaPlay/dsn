SET(dadaML_SOURCES
		Discriminant.cpp Discriminant.h	
		FisherDiscriminant.cpp FisherDiscriminant.h	
		SvmDiscriminant.cpp SvmDiscriminant.h	
		NormalDataset.cpp NormalDataset.h	
		SynData.cpp SynData.h	
		KMeanFeatures.cpp KMeanFeatures.h
		Whiten.cpp Whiten.h
		#Hungarian.cpp Hungarian.h
		#OhmHungarian.cpp OhmHungarian.h
		#ErrorEval.cpp ErrorEval.h
		#Stump.cpp Stump.h		
		#MatlabIO.cpp MatlabIO.h
		#ACD.cpp ACD.h		
)

INCLUDE_DIRECTORIES( 
		${PROJECT_SOURCE_DIR}/3rdParty/liblinear-weights-1.91		
		${PROJECT_SOURCE_DIR}/dadaCore
		${PROJECT_SOURCE_DIR}/gtest-1.6.0/include
)

ADD_LIBRARY(dadaML ${i2ml_SOURCES} )
TARGET_LINK_LIBRARIES(i2ml_Lib libLinearWeights)

ADD_EXECUTABLE(i2ml_LibTest i2mlTest.cpp) 
TARGET_LINK_LIBRARIES(i2ml_LibTest rbp-core i2ml_Lib)
