
SET(libLinearWeights_SOURCES
	linear.cpp linear.h 	## linear.def 
	tron.cpp tron.h
	blas/daxpy.c
	blas/ddot.c
	blas/dnrm2.c
	blas/dscal.c
	blas/blas.h
	blas/blasp.h
)
INCLUDE_DIRECTORIES(	
	blas
)
ADD_LIBRARY(libLinearWeights ${libLinearWeights_SOURCES} )

ADD_EXECUTABLE(train train.c) 
TARGET_LINK_LIBRARIES(train libLinearWeights)

ADD_EXECUTABLE(predict predict.c) 
TARGET_LINK_LIBRARIES(predict libLinearWeights)
