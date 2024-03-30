// DadaException.h
#ifndef DadaException_h_
#define DadaException_h_

#include <boost/exception/all.hpp>
#include <boost/lexical_cast.hpp>
#include <exception>
#include "opencv2/opencv.hpp"

typedef boost::error_info<struct tag_errmsg, std::string> errmsg_info;

/** 
 * MAMAs derived boost exception class
 **/
class DadaException : public boost::exception, public std::exception 
{
  public:
	  DadaException() {};
	  DadaException(std::string msg) { what_ = msg; };
	  virtual const char *what() const throw() { return( what_.c_str() ); };

	std::string what_;
}; 

/**
* Derived exception class
**/
class Opencv: public DadaException
{
public:
	Opencv() {};
	Opencv(const char *msg) { what_ = std::string(msg); };
};

/**
* Derived exception class
**/
class Unexpected : public DadaException
{
  public:
	  Unexpected() {};
	  Unexpected(std::string msg) { what_ = msg; };
}; 

/**
* Derived exception class
**/
class UnexpectedSize : public DadaException
{
  public:
	  UnexpectedSize() {};
	  UnexpectedSize(std::string msg) { what_ = msg; };
	  UnexpectedSize(int got, int ex) { what_ = "Size "; what_ += boost::lexical_cast<std::string>(got); what_ += " but expected "; what_ += boost::lexical_cast<std::string>(ex); };
}; 

/**
* Derived exception class
**/
class UnsupportedType : public DadaException
{
  public:
	  UnsupportedType() {};
	  UnsupportedType(std::string msg) { what_ = msg; };
	  UnsupportedType(int got, int ex) { what_ = "Unsupported type: "; what_ += boost::lexical_cast<std::string>(got); what_ += " but expected "; what_ += boost::lexical_cast<std::string>(ex); };
}; 

/**
* Derived exception class
**/
class UnexpectedType: public DadaException
{
  public:
	  UnexpectedType() {};
	  UnexpectedType(std::string msg) { what_ = msg; };
	  UnexpectedType(int got, int ex) { what_ = "Unexpected type: "; what_ += boost::lexical_cast<std::string>(got); what_ += " but expected "; what_ += boost::lexical_cast<std::string>(ex); };
}; 

/**
* Derived exception class
**/
class NotImplemented : public DadaException
{
  public:
	  NotImplemented() {};
	  NotImplemented(std::string msg) { what_ = msg; };
}; 

/**
* Derived exception class
**/
class FileIOProblem: public DadaException
{
  public:
	  FileIOProblem() {};
	  FileIOProblem(std::string msg) { what_ = msg; };
}; 

/**
* Derived exception class
**/
class StorageProblem : public DadaException
{
public:
	StorageProblem() {};
	StorageProblem(std::string msg) { what_ = msg; };
};

/**
* Derived exception class
**/
class MemoryProblem : public DadaException
{
  public:
	  MemoryProblem() {};
	  MemoryProblem(std::string msg) { what_ = msg; };
}; 

#endif // Exception_h_
