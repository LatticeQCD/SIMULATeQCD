/*
 * This code is a heavily modified version of einhard.hpp
 *
 * Copyright 2010 Matthias Bach <marix@marix.org>
 *
 * This file is part of Einhard.
 *
 * Einhard is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Einhard is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Einhard.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LOGGER
#define LOGGER

#include <ctime>
#include <iomanip>
#include <iostream>
#include <stack>
#include <stdexcept>
#include <stdio.h>
#include <unistd.h>

// forward declarations related to CommunicationBase
class CommunicationBase;
std::string getLocalInfoStringFromCommBase(CommunicationBase *commBase);

static char const ANSI_ESCAPE = 27;
static char const *const ANSI_COLOR_WARN = "[33m";    // yellow
static char const *const ANSI_COLOR_ERROR = "[31m";   // red
static char const *const ANSI_COLOR_FATAL = "[1;31m"; // bold red
static char const *const ANSI_COLOR_CLEAR = "[0m";

enum LogLevel { ALL, ALLOC, TRACE, DEBUG, INFO, WARN, ERROR, FATAL, OFF };
static const char *LogLevelStr[] = {"ALL",  "ALLOC", "TRACE", "DEBUG", "INFO",
                                    "WARN", "ERROR", "FATAL", "OFF"};

/// This represents a single line of logging output. It starts with a timestamp
/// and the loglevel (which are printed by the constructor) and ends with an
/// end-of-line (by the destructor). Everything in between can be passed with
/// the operator << as long as this operator << exists with std::ostream as left
/// hand side.
struct LogLine {

private:
  std::ostream *const out;
  LogLevel verbosity;
  bool const colorize;

public:
  LogLine(std::ostream *const out = 0, LogLevel verbosity = ALL,
          bool colorize = false, CommunicationBase *commBase = nullptr,
          bool CommBaseIsInitialized = false)
      : out(out), verbosity(verbosity), colorize(colorize) {
    if (out == 0)
      return;

    std::string mpi_info;
    if (CommBaseIsInitialized) {
      mpi_info = getLocalInfoStringFromCommBase(commBase);
    }

    time_t rawtime;
    tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    if (colorize && verbosity == WARN)
      *out << ANSI_ESCAPE << ANSI_COLOR_WARN;
    if (colorize && verbosity == ERROR)
      *out << ANSI_ESCAPE << ANSI_COLOR_ERROR;
    if (colorize && verbosity == FATAL)
      *out << ANSI_ESCAPE << ANSI_COLOR_FATAL;

    *out << "# [" << 1900 + timeinfo->tm_year << '-' << std::setfill('0')
         << std::setw(2) << 1 + timeinfo->tm_mon << '-' << std::setw(2)
         << timeinfo->tm_mday << ' ' << std::setw(2) << timeinfo->tm_hour << ':'
         << std::setw(2) << timeinfo->tm_min << ':' << std::setw(2)
         << timeinfo->tm_sec << ']' << mpi_info << ' ' << LogLevelStr[verbosity]
         << ": " << std::setfill(' ');
  }

  /// Streaming some item will stream it into *out unless out is NULL
  template <typename T> const LogLine &operator<<(const T &msg) const {
    if (out != 0)
      *out << msg;

    return *this;
  }

  /// Same as above but for ibm compiler?!
  const LogLine &operator<<(std::ios_base &(*manip)(std::ios_base &)) const {
    if (out != 0)
      manip(*out);
    return *this;
  }

  /// Streaming of manipulators (like std::endl), which are functions acting on
  /// a std::ostream
  const LogLine &operator<<(std::ostream &(*manip)(std::ostream &)) const {
    if (out != 0)
      manip(*out);
    return *this;
  }

  ~LogLine() {
    if (out == 0)
      return;
    /// Make sure there is no strange formatting set anymore
    *out << std::resetiosflags(
        std::ios_base::floatfield | std::ios_base::basefield |
        std::ios_base::adjustfield | std::ios_base::uppercase |
        std::ios_base::showpos | std::ios_base::showpoint |
        std::ios_base::showbase | std::ios_base::boolalpha);
    if (colorize &&
        (verbosity == WARN || verbosity == ERROR || verbosity == FATAL))
      *out << ANSI_ESCAPE << ANSI_COLOR_CLEAR;
    *out << std::endl;
  }
};

class Logger {
private:
  bool colorize;
  LogLevel verbosity;
  std::ostream &out;
  std::stack<LogLevel> verbosity_stack;
  CommunicationBase *commBase = nullptr;
  bool CommBaseIsInitialized = false;

public:
  Logger(LogLevel verbosity = ALL, std::ostream &out = std::cout)
      : verbosity(verbosity), out(out) {
    colorize = isatty(fileno(stdout));
  }

  const LogLine line(LogLevel lv) const {
    if (lv < verbosity) {
      return LogLine();
    } else {
      return LogLine(&out, lv, colorize, commBase, CommBaseIsInitialized);
    }
  }

  bool shows(LogLevel lv) const { return lv >= verbosity; }

  const LogLine alloc() const { return line(ALLOC); }
  const LogLine trace() const { return line(TRACE); }
  const LogLine debug() const { return line(DEBUG); }
  const LogLine info() const { return line(INFO); }
  const LogLine warn() const { return line(WARN); }

  /*! Use this when something goes wrong but the program can still continue
   * Example: a test gives the wrong results
   */
  const LogLine error() const { return line(ERROR); }

  /*! Use this when the program should be terminated (ie throw a runtime
   * exception). Example: there are conflicting input parameters
   */
  const LogLine fatal() const { return line(FATAL); }

  void setCommunicationBase(CommunicationBase *commBasePtr) {
    this->commBase = commBasePtr;
  }

  void activateLocalProcessInfo() { CommBaseIsInitialized = true; }
  void deactivateLocalProcessInfo() { CommBaseIsInitialized = false; }

  void setVerbosity(LogLevel v) { verbosity = v; }
  LogLevel getVerbosity() { return verbosity; }
  void push_verbosity(LogLevel v) {
    verbosity_stack.push(getVerbosity());
    setVerbosity(v);
  }
  void pop_verbosity() {
    setVerbosity(verbosity_stack.top());
    verbosity_stack.pop();
  }
};

/// This logger prints something on each node. It is created in
/// base/communicationBase_*.cpp and its verbosity should be set at the
/// beginning of main()
extern Logger stdLogger;
/// This logger is only turned on on the root node (in the constructor of
/// CommunicationBase)
extern Logger rootLogger;

#endif
