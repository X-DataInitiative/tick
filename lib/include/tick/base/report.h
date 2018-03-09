//
//  report.h
//  TICK
//
//  Created by Philip Deegan
//  Copyright (c) 2015 bacry. All rights reserved.
//
//  Multipurpose Wrapper return type

#ifndef LIB_INCLUDE_TICK_BASE_REPORT_H_
#define LIB_INCLUDE_TICK_BASE_REPORT_H_

#include <sstream>

template <class A = bool, class B = std::string>
class Report {};

class BoolStrReport : public Report<bool, std::string> {
  friend std::ostream& operator<<(std::ostream&, const BoolStrReport&);

 private:
  bool success = false;
  std::string str;

 public:
  BoolStrReport(bool b, const std::string& s) : success(b), str(s) {}
  BoolStrReport(BoolStrReport& b, const std::string& s)
      : success(b.success), str(s) {}
  explicit BoolStrReport(const std::pair<bool, std::string>& p)
      : success(p.first), str(p.second) {}
  explicit BoolStrReport(const std::pair<BoolStrReport, std::string>& p)

      : success(p.first.success), str(p.second) {}
  BoolStrReport(const BoolStrReport& r)
      : success(r.success), str(std::move(r.str)) {}
  BoolStrReport(const BoolStrReport&& r)
      : success(r.success), str(std::move(r.str)) {}
  BoolStrReport& operator=(const BoolStrReport& r) = delete;
  BoolStrReport& operator=(const BoolStrReport&& r) = delete;

  const std::string& why() { return str; }

  explicit operator bool() const { return success; }
};

inline std::ostream& operator<<(std::ostream& stream,
                                const BoolStrReport& report) {
  return stream << report.str;
}

#endif  // LIB_INCLUDE_TICK_BASE_REPORT_H_

