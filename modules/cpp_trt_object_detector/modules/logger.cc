#include "logger.h"

namespace omv {

int NullBuffer::overflow(int c) { return c; }

NullStream::NullStream() : std::ostream(&m_sb) {}
NullStream::~NullStream() {}
CoutStream::CoutStream() : std::ostream(std::clog.rdbuf()) {}
CoutStream::~CoutStream() {}

std::ostream& operator<<(std::ostream& os, const Endl&) { return os; }

Logger::Logger() {}

Logger& Logger::Instance() {
  static Logger logger;
  return logger;
}

void Logger::Init(int severity) {
  if (severity < 5 && severity > -1)
    severity_ = static_cast<OmvSeverity>(severity);
  else
    severity_ = OmvSeverity::DEBUG;
}

void Logger::Init(OmvSeverity severity) {
  if (severity > OmvSeverity::MAIN && severity <= OmvSeverity::DEBUG)
    severity_ = severity;
  else
    severity_ = OmvSeverity::DEBUG;
}

std::string Logger::ts() {
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);

  char buffer[24];

  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
  return std::string(buffer);
}

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) {
  OmvSeverity s = static_cast<OmvSeverity>(severity);
  log_stream(s) << Endl(s) << "[TensorRT] " << msg;
}

OmvSeverity Logger::GetSeverity() const { return severity_; }
std::string Logger::GetLogLevel() const { return kLevels.at(severity_); }

void Logger::SetSeverityLevel(OmvSeverity severity) { severity_ = severity; }
void Logger::Lock() { mutex_.lock(); }
void Logger::UnLock() { mutex_.unlock(); }

Endl::Endl(OmvSeverity message_severity) { enabled_ = !(message_severity > Logger::Instance().GetSeverity()); }
Endl::~Endl() {
  if (enabled_)
    std::clog << std::endl;

  Logger::Instance().UnLock();
}

}  // namespace omv
