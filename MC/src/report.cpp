#include "report.hpp"

#include <fstream>
#include <memory>
#include <stdexcept>

namespace {
std::unique_ptr<std::ofstream> g_report_stream;
std::string g_report_path = "report.txt";
}

std::ostream& mc_report(){
    if (!g_report_stream){
        init_mc_report(g_report_path);
    }
    return *g_report_stream;
}

void init_mc_report(const std::string& path){
    g_report_path = path;
    g_report_stream = std::make_unique<std::ofstream>(g_report_path, std::ios::out | std::ios::trunc);
    if (!g_report_stream->is_open()){
        g_report_stream.reset();
        throw std::runtime_error("Failed to open report file: " + g_report_path);
    }
}

void close_mc_report(){
    if (g_report_stream){
        g_report_stream->close();
        g_report_stream.reset();
    }
}

ReportFileGuard::ReportFileGuard(const std::string& path){
    init_mc_report(path);
}

ReportFileGuard::~ReportFileGuard(){
    close_mc_report();
}
