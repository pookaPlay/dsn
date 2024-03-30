
SET(lbfgs_SOURCES
	TerminationCriterion.cpp TerminationCriterion.h
	OWLQN.cpp OWLQN.h
	logreg.cpp logreg.h
)

ADD_LIBRARY(lbfgs ${lbfgs_SOURCES} )

TARGET_LINK_LIBRARIES(lbfgs ${Core_LIBS})
