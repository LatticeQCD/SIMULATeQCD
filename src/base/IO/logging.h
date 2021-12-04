
#ifndef LOGGER
#define LOGGER

#include <cassert>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stack>
#include <string>

template <typename U>
inline void args_to_stream(std::ostringstream &out, U msg) {
  out << msg;
}

template <typename U, typename... Args>
inline void args_to_stream(std::ostringstream &out, U msg, Args... args) {
  out << msg;
  args_to_stream(out, args...);
}

template <typename... Args> inline std::string sjoin(Args... args) {
  std::ostringstream msg;
  args_to_stream(msg, args...);
  return msg.str();
}

template <typename... Args> std::string sformat(const char *fmt, Args... args) {
  size_t size = snprintf(nullptr, 0, fmt, args...);
  std::string buf;
  buf.reserve(size + 1);
  buf.resize(size);
  snprintf(&buf[0], size + 1, fmt, args...);
  return buf;
}

namespace COLORS {
const std::string red("\033[0;31m");
const std::string redBold("\033[1;31m");
const std::string green("\033[0;32m");
const std::string greenBold("\033[1;32m");
const std::string yellow("\033[0;33m");
const std::string yellowBold("\033[1;33m");
const std::string cyan("\033[0;36m");
const std::string cyanBold("\033[1;36m");
const std::string magenta("\033[0;35m");
const std::string magentaBold("\033[1;35m");
const std::string reset("\033[0m");
} // namespace COLORS

enum LogLevel { ALL, ALLOC, TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF };
static const char *LogLevelStr[] = {"ALL",  "ALLOC", "TRACE", "DEBUG", "INFO",
                                    "WARN", "ERROR", "FATAL", "OFF"};

inline std::string timeStamp() {
  std::ostringstream strStream;
  std::time_t t = std::time(nullptr);
  strStream << "[" << std::put_time(std::localtime(&t), "%F %T") << "] ";
  return strStream.str();
}

class Logger {
private:
  LogLevel log_level;
  bool colorized_output;
  std::stack<LogLevel> verbosity_stack;
  std::ostream &out;
  std::string addPrefix;

public:
  Logger(LogLevel loglevel = ALL, bool colorized_output = true,
         std::ostream &out = std::cout, std::string addPrefix = "")
      : log_level(loglevel), colorized_output(colorized_output), out(out),
        addPrefix(addPrefix) {}

  void setVerbosity(LogLevel v) { log_level = v; }

  LogLevel getVerbosity() { return log_level; }

  inline void push_verbosity(LogLevel v) {
    verbosity_stack.push(log_level);
    log_level = v;
  }

  inline void pop_verbosity() {
    log_level = verbosity_stack.top();
    verbosity_stack.pop();
  }

  inline void set_additional_prefix(std::string add) { addPrefix = add; }
  inline std::string get_additional_prefix() { return addPrefix; }

  template <LogLevel level, typename... Args>
  inline std::string message(Args... args) {
    std::ostringstream prefix, loginfo, postfix;

    if (colorized_output && level == WARN)
      prefix << COLORS::yellow;
    if (colorized_output && level == ERROR)
      prefix << COLORS::red;
    if (colorized_output && level == FATAL)
      prefix << COLORS::redBold;

    loginfo << "# " << timeStamp() << LogLevelStr[level] << ": ";

    std::string msg = sjoin(args...);

    postfix << COLORS::reset
            << std::resetiosflags(
                   std::ios_base::floatfield | std::ios_base::basefield |
                   std::ios_base::adjustfield | std::ios_base::uppercase |
                   std::ios_base::showpos | std::ios_base::showpoint |
                   std::ios_base::showbase | std::ios_base::boolalpha)
            << std::endl;

    if (level > log_level) {
      out << prefix.str() << loginfo.str() << addPrefix << msg << postfix.str();
    }

    return prefix.str() + addPrefix + msg + postfix.str();
  }

  template <typename... Args> inline std::string info(Args... args) {
    return message<INFO>(args...);
  };
  template <typename... Args> inline std::string trace(Args... args) {
    return message<TRACE>(args...);
  };
  template <typename... Args> inline std::string alloc(Args... args) {
    return message<ALLOC>(args...);
  };
  template <typename... Args> inline std::string debug(Args... args) {
    return message<DEBUG>(args...);
  };
  template <typename... Args> inline std::string warn(Args... args) {
    return message<WARN>(args...);
  };

  /*! Use this when something goes wrong but the program can still continue
   * Example: a test gives the wrong results
   */
  template <typename... Args> inline std::string error(Args... args) {
    return message<ERROR>(args...);
  };

  /*! Use this when the program should be terminated (ie throw a runtime
   * exception). Example: there are conflicting input parameters
   */
  template <typename... Args> inline std::string fatal(Args... args) {
    return message<FATAL>(args...);
  };
};

/// This logger prints something on each node. It is created in
/// base/communicationBase_*.cpp and its verbosity should be set at the
/// beginning of main()
extern Logger stdLogger;
/// This logger is only turned on on the root node (in the constructor of
/// CommunicationBase)
extern Logger rootLogger;

#endif
