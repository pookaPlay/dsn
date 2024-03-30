
set(GC_DIR /home/rporter/src/nowozin)
set(OSI_DIR ${GC_DIR}/osi-bin)
set(GMM_DIR ${GC_DIR}/gmm-install)

include_directories(
        ${OSI_DIR}/include
        ${GMM_DIR}/include
        ${GC_DIR}/GraphClustering
)


ADD_EXECUTABLE(graphCluster mainGraphCluster.cpp ) 

TARGET_LINK_LIBRARIES(graphCluster
                    dadaCore 
                    ${CORE_LIBS}
                    ${GC_DIR}/GraphClustering/libgclustering.a 
                    ${OSI_DIR}/lib/libOsiClp.a
                    ${OSI_DIR}/lib/libOsi.a 
                    ${OSI_DIR}/lib/libClp.a
                    ${OSI_DIR}/lib/libCoinUtils.a 
                    m z) 
                    #lapack blas)
                    
#                    -lm \
#	-L/usr/lib/sse2 -llapack \
#	-L/usr/lib/atlas/sse2 -lblas
#                    )
