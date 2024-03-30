
SET(dada_Lib_SOURCES
	DadaSLIC.cpp DadaSLIC.h		
	DadaMama.cpp DadaMama.h		
	DadaWS.cpp DadaWS.h		
	DadaWSACD.cpp DadaWSACD.h		
	DadaWSGT.cpp DadaWSGT.h		
	DadaWSTrain.cpp DadaWSTrain.h		
	DadaWSUtil.cpp DadaWSUtil.h		
	DadaFeatures.cpp DadaFeatures.h		
	DadaPooler.cpp DadaPooler.h			
	DadaFeatureGenerator.cpp DadaFeatureGenerator.h		
	DadaFeaturesKMeans.cpp DadaFeaturesKMeans.h		
	DadaFeaturesEdge.cpp DadaFeaturesEdge.h			
	DadaFeaturesEdges.cpp DadaFeaturesEdges.h		
	DadaFeaturesVecEdges.cpp DadaFeaturesVecEdges.h	
	DadaPoolerVecEdges.cpp DadaPoolerVecEdges.h				
	DadaPoolerEdges.cpp DadaPoolerEdges.h			
	DadaPoolerEdge.cpp DadaPoolerEdge.h		
	DadaFeaturesMoments.cpp DadaFeaturesMoments.h			
	DadaPoolerMoments.cpp DadaPoolerMoments.h		
	DadaFeaturesHistogram.cpp DadaFeaturesHistogram.h			
	DadaPoolerHistogram.cpp DadaPoolerHistogram.h		
	DadaParam.cpp DadaParam.h		
	DadaSegmenter.cpp DadaSegmenter.h		
	DadaClassifier.cpp DadaClassifier.h		
	DadaClassifierStump.cpp DadaClassifierStump.h			
	DadaClassifierLinear.cpp DadaClassifierLinear.h		
	DadaClassifierBFGS.cpp DadaClassifierBFGS.h		
	DadaClassifierParam.cpp DadaClassifierParam.h		
	DadaEval.cpp DadaEval.h		
	DadaIID.cpp DadaIID.h		
	DadaBatch.cpp DadaBatch.h		
	KMeanFeatures.cpp KMeanFeatures.h
	Whiten.cpp Whiten.h
	Normalize.cpp Normalize.h	
	DadaSegmenterTree.cpp DadaSegmenterTree.h		
	DadaSegmenterTreeNode.cpp DadaSegmenterTreeNode.h		
	DadaSegmenterTreeSplit.cpp DadaSegmenterTreeSplit.h		
	DadaSegmenterTreeMerge.cpp DadaSegmenterTreeMerge.h		
	DadaSegmenterTreeParam.cpp DadaSegmenterTreeParam.h		
	DadaSegmenterForest.cpp DadaSegmenterForest.h		
	DadaSegmenterForestParam.cpp DadaSegmenterForestParam.h		
	ISegHighGUI.cpp ISegHighGUI.h
)

ADD_LIBRARY(dada_Lib ${dada_Lib_SOURCES} )

INCLUDE_DIRECTORIES(	
	${PROJECT_SOURCE_DIR}/src/common/rbpl	
	${PROJECT_SOURCE_DIR}/src/common/rbpImg		
	${PROJECT_SOURCE_DIR}/src/mamaWF_Lib
	${PROJECT_SOURCE_DIR}/src/mamaWS_Lib
	${PROJECT_SOURCE_DIR}/src/DadaTools/LBFGS
	${PROJECT_SOURCE_DIR}/src/DadaTools/SLIC
	)

TARGET_LINK_LIBRARIES(dada_Lib lbfgs mamaWS_Lib rbpl rbpImg slic_Lib ${Mama_LIBS})
