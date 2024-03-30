PROJECT(dsn)

cmake_minimum_required(VERSION 3.1)


IF (WIN32)
  SET (CMAKE_INSTALL_PREFIX "c:/install" CACHE INTERNAL "install prefix")
  SET (Boost_INCLUDE_DIRS ${INSTALL_PREFIX}/include/boost)
  SET (OpenCV_INCLUDE_DIRS ${INSTALL_PREFIX}/include)
  SET (GDAL_INCLUDE_DIRS ${INSTALL_PREFIX}/include)

  SET (log4cplus_INCLUDE_DIRS ${INSTALL_PREFIX}/include)
  SET (log4cplus_LIBRARY log4cplus)

ELSE(WIN32)
  #                    Supporting Debian 10 (Vj)
  #    Add the following line to ~/.bashrc.
  #        export CMAKE_PREFIX_PATH=/usr/include:/usr/lib/x86_64-linux-gnu
  find_package(OpenCV 3.1.1 REQUIRED)
  find_package(Boost 1.65.1 REQUIRED)
  find_package(GDAL REQUIRED)
  
  SET (cfitsio_INCLUDE_DIRS /usr/include)
  SET (log4cplus_INCLUDE_DIRS /usr/include/log4cplus)
  
  SET (CMAKE_INSTALL_PREFIX "../install" CACHE INTERNAL "install prefix")
ENDIF(WIN32)



SET (INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL "install prefix")
SET (EXECUTABLE_OUTPUT_PATH "${CMAKE_INSTALL_PREFIX}/bin" CACHE EXTERNAL "Final bin directory")
SET (LIBRARY_OUTPUT_PATH "${CMAKE_INSTALL_PREFIX}/lib" CACHE EXTERNAL "Final lib directory")


IF (WIN32)
  SET (Boost_LIBRARIES 
    boost_chrono-vc-mt-1_65_1 
    boost_date_time-vc-mt-1_65_1 
    boost_filesystem-vc-mt-1_65_1 
    boost_program_options-vc-mt-1_65_1 
    boost_serialization-vc-mt-1_65_1 
    boost_system-vc-mt-1_65_1 
    boost_thread-vc-mt-1_65_1) 			

  SET (OpenCV_LIBRARIES 
    opencv_features2d331 
    opencv_flann331 
    opencv_highgui331 
    opencv_imgproc331 
    opencv_ml331 
    opencv_core331 
    opencv_imgcodecs331)
  
  SET (GDAL_LIBRARY gdal_i)

ELSE(WIN32)			
  SET (Boost_LIBRARIES boost_chrono boost_date_time boost_filesystem boost_program_options boost_serialization boost_system boost_thread)
  SET (OpenCV_LIBRARIES opencv_features2d opencv_flann opencv_highgui opencv_imgproc opencv_ml opencv_core opencv_imgcodecs)
  SET (GDAL_LIBRARY gdal)
ENDIF(WIN32)




set (dada_LIBS
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}		
  ${GDAL_LIBRARY}		
  log4cplus
  cfitsio        
  )

set(dada_LINK
  ${INSTALL_PREFIX}/lib 
  ${INSTALL_PREFIX}/lib64
  )
IF (WIN32)
  set(dada_INCLUDE ${INSTALL_PREFIX}/include)
ELSE(WIN32)
  set(dada_INCLUDE
        ${OpenCV_INCLUDE_DIRS}
        ${GDAL_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
	${log4cplus_INCLUDE_DIRS}
	${cfitsio_INCLUDE_DIRS}
    )
ENDIF(WIN32)


INCLUDE_DIRECTORIES (${dada_INCLUDE})	

LINK_DIRECTORIES (
  ${dada_LINK}
  )

message ("Libs: ${dada_LIBS}")    
message ("Link Dirs: ${dada_LINK}")    
message ("Include Dirs: ${dada_INCLUDE}")    

include_directories(	
  ${PROJECT_SOURCE_DIR}/dadaCore	
  ${PROJECT_SOURCE_DIR}/dadaSeg	
  )

# -Wdeprecated-declarations
IF (WIN32)
  #/EHsc
  SET (CMAKE_CXX_FLAGS "/EHsc -std=c++14")        
  add_definitions (
    -DBOOST_ALL_DYN_LINK
    -DBOOST_THREAD_PROVIDES_FUTURE
    )
ELSE(WIN32)
  SET (CMAKE_CXX_FLAGS "-O3 -std=c++14 -Wno-deprecated-declarations")        
ENDIF(WIN32)

add_subdirectory (dadaCore)
add_subdirectory (dadaSeg)
add_subdirectory (segTest)
#add_subdirectory (mamaVectorize)
#add_subdirectory (dadaDSN)
#add_subdirectory (paraview)
#add_subdirectory (3rdParty)


