#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>

#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#if __x86_64__
#define LOGGER_SETW 16
#else
#define LOGGER_SETW 8
#endif

enum OmvSeverity { MAIN = -1, FATAL = 0, ERROR, WARN, INFO, DEBUG };

namespace omv {


class NullBuffer : public std::streambuf {
 public:
  int overflow(int c);
};

class NullStream : public std::ostream {
 public:
  NullStream();
  ~NullStream();

 private:
  NullBuffer m_sb;
};

class CoutStream : public std::ostream {
 public:
  CoutStream();
  ~CoutStream();
};

class Logger : public nvinfer1::ILogger {
 public:
  Logger();
  ~Logger() override = default;

  Logger(Logger const&) = delete;
  Logger& operator=(Logger const&) = delete;

  static Logger& Instance();
  static std::string ts();

  void Init(int severity = 1);
  void Init(OmvSeverity severity = ERROR);

  void log(Severity severity, const char* msg) override;

  inline std::ostream& log_stream(OmvSeverity severity) {
    if (severity <= severity_) {
      Lock();
      /* clang-format off */
        return cout_ << '['
#if defined (OMVFID_DEBUG)
                     << std::right << std::setfill('0') << std::setw(LOGGER_SETW)
                     << std::hex << std::this_thread::get_id() << ']'
                     << '['
#endif
                     << std::left << std::setfill(' ') << std::setw(5)
                     << std::dec << kLevels.at(severity) << ']';
      /* clang-format on */
    } else
      return null_;
  }

  OmvSeverity GetSeverity() const;
  std::string GetLogLevel() const;

  void SetSeverityLevel(OmvSeverity severity);
  void Lock();
  void UnLock();

 private:
  OmvSeverity severity_;
  NullStream null_;
  CoutStream cout_;
  std::mutex mutex_;

  /* clang-format off */
  const std::map<OmvSeverity, std::string> kLevels{{OmvSeverity::DEBUG, "debug"},
                                                   {OmvSeverity::INFO, "info"},
                                                   {OmvSeverity::WARN, "warn"},
                                                   {OmvSeverity::ERROR, "error"},
                                                   {OmvSeverity::FATAL, "fatal"},
                                                   {OmvSeverity::MAIN, "main"}};
  /* clang-format on */
};

class Endl {
 public:
  Endl() = default;
  Endl(OmvSeverity message_severity);

  ~Endl();

  friend std::ostream& operator<<(std::ostream& os, const omv::Endl&);

 private:
  bool enabled_{true};
};

std::ostream& operator<<(std::ostream& os, const omv::Endl&);
}  // namespace omv

inline std::string ComputeMethodName(const std::string& pretty_function) {
  std::string pretty = pretty_function;

  size_t om = pretty.find("omv::");
  if (om != std::string::npos)
    pretty.erase(pretty.begin(), pretty.begin() + static_cast<int>(om + 5));

  size_t brackets = pretty.find_first_of('(');

  if (brackets != std::string::npos)
    pretty.replace(pretty.begin() + static_cast<int>(brackets + 1), pretty.end(), 1, ')');

  size_t space = pretty.find_last_of(' ');

  if (space != std::string::npos)
    pretty.erase(pretty.begin(), pretty.begin() + static_cast<int>(space + 1));

  return pretty;
}

#define VERY_PRETTY_NAME ComputeMethodName(__PRETTY_FUNCTION__)
#define LOG(x) \
  omv::Logger::Instance().log_stream(x) << omv::Endl(x) << '[' << __FUNCTION__ << ':' << __LINE__ << "] "
#define LOGX LOG(omv::MAIN)

#endif  // LOGGER_H
