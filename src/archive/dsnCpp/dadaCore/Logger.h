#ifndef LOGGER_H
#define LOGGER_H

#include <boost/exception/diagnostic_information.hpp>
#include <log4cplus/configurator.h>
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

/**
 * Logger.
 */
typedef log4cplus::Logger Logger;

/**
 * Configures the logging subsystem using basic defaults.
 */
#define LOG_CONFIGURE_BASIC() \
	::log4cplus::BasicConfigurator::doConfigure()

/**
 * Configures the logging subsystem using the specified configuration file.
 *
 * @param  fileName  The name of the configuration file.
 */
#define LOG_CONFIGURE(fileName) \
	::log4cplus::PropertyConfigurator::doConfigure(fileName)

/**
 * Returns the logger with the specified name.
 *
 * @param  name  The name of the logger to return. Typically, this is the name
 *   of a class or function.
 */
#define LOG_GET_LOGGER(name) \
	::log4cplus::Logger::getInstance(name)

///@{
/**
 * Logs the specified message.
 *
 * @param  logger      The logger.
 * @param  expression  The streaming expression that generates the message.
 */
#define LOG_TRACE LOG4CPLUS_TRACE
#define LOG_DEBUG LOG4CPLUS_DEBUG
#define LOG_INFO  LOG4CPLUS_INFO
#define LOG_WARN  LOG4CPLUS_WARN
#define LOG_ERROR LOG4CPLUS_ERROR
#define LOG_FATAL LOG4CPLUS_FATAL
///@}

/**
 * Logs a message on entry to and exit from a method.
 *
 * @param  logger      The logger.
 * @param  expression  The streaming expression that generates the message.
 */
#define LOG_TRACE_METHOD(logger, expression) \
	::log4cplus::tostringstream _log4cplus_trace_logger_stream; \
	_log4cplus_trace_logger_stream << expression; \
	::log4cplus::TraceLogger _log4cplus_trace_logger(logger, \
		_log4cplus_trace_logger_stream.str(), __FILE__, __LINE__);

///@{
/**
 * Logs the specified message with an exception.
 *
 * @param  logger      The logger.
 * @param  expression  The streaming expression that generates the message.
 * @param  ex          The exception.
 */
#define LOG_TRACE_EX(logger, expression, ex) \
	LOG_TRACE(logger, expression \
		<< "\n" << ::boost::diagnostic_information(ex))

#define LOG_DEBUG_EX(logger, expression, ex) \
	LOG_DEBUG(logger, expression \
		<< "\n" << ::boost::diagnostic_information(ex))

#define LOG_INFO_EX(logger, expression, ex) \
	LOG_INFO(logger, expression \
		<< "\n" << ::boost::diagnostic_information(ex))

#define LOG_WARN_EX(logger, expression, ex) \
	LOG_WARN(logger, expression \
		<< "\n" << ::boost::diagnostic_information(ex))

#define LOG_ERROR_EX(logger, expression, ex) \
	LOG_ERROR(logger, expression \
		<< "\n" << ::boost::diagnostic_information(ex))

#define LOG_FATAL_EX(logger, expression, ex) \
	LOG_FATAL(logger, expression \
		<< "\n" << ::boost::diagnostic_information(ex))
///@}

#endif
