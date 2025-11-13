#ifndef MC_REPORT_HPP
#define MC_REPORT_HPP

#include <ostream>
#include <string>

std::ostream& mc_report();
void init_mc_report(const std::string& path = "report.txt");
void close_mc_report();

class ReportFileGuard {
public:
    explicit ReportFileGuard(const std::string& path = "report.txt");
    ~ReportFileGuard();
    ReportFileGuard(const ReportFileGuard&) = delete;
    ReportFileGuard& operator=(const ReportFileGuard&) = delete;
};

#endif // MC_REPORT_HPP
