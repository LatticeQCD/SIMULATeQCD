#pragma once

// Opt-in wall-clock profile of the CPU backend: every host kernel launch (by functor
// type), reduction, copy and MPI halo wait is timed when SIMULATEQCD_CPU_PROFILE=1, and a
// table sorted by time is printed at exit by MPI rank 0 (every rank prints its total and
// MPI halo wait in one line). Without the variable a Scope
// only tests one cached bool.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cpuProfile {

struct Entry {
    size_t calls = 0;
    double seconds = 0.0;
};

inline bool enabled() {
    static const bool on = [] {
        const char *value = std::getenv("SIMULATEQCD_CPU_PROFILE");
        return value && *value && std::string_view(value) != "0";
    }();
    return on;
}

inline int worldRank() {
    for (const char *name : {"OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"}) {
        if (const char *value = std::getenv(name)) return std::atoi(value);
    }
    return 0;
}

inline void report();

inline std::map<std::string, Entry, std::less<>> &table() {
    static std::map<std::string, Entry, std::less<>> entries;
    static const bool registered = [] { return std::atexit(report) == 0; }();
    (void)registered;
    return entries;
}

inline std::mutex &tableMutex() {
    static std::mutex mutex;
    return mutex;
}

inline void add(std::string_view name, double seconds) {
    // type_name<T>() ends in '\n'.
    while (!name.empty() && name.back() == '\n') name.remove_suffix(1);
    std::lock_guard<std::mutex> lock(tableMutex());
    auto &entries = table();
    auto it = entries.find(name);
    if (it == entries.end()) it = entries.emplace(std::string(name), Entry{}).first;
    it->second.calls++;
    it->second.seconds += seconds;
}

inline void report() {
    std::vector<std::pair<std::string, Entry>> rows(table().begin(), table().end());
    std::sort(rows.begin(), rows.end(),
              [](const auto &a, const auto &b) { return a.second.seconds > b.second.seconds; });
    double total = 0.0, wait = 0.0;
    for (const auto &row : rows) {
        total += row.second.seconds;
        if (row.first == "MPI halo wait") wait = row.second.seconds;
    }
    // One line from every rank: a rank's halo wait includes waiting for slower neighbours.
    std::printf("CPU rank profile: rank %d, %.3f s profiled, MPI halo wait %.3f s\n", worldRank(), total, wait);
    if (worldRank() != 0) {
        std::fflush(stdout);
        return;
    }
    std::printf("CPU profile: %d entries, %.3f s profiled in total (rank 0)\n",
                static_cast<int>(rows.size()), total);
    std::printf("CPU profile: %10s %6s %10s %12s  %s\n", "seconds", "%", "calls", "ms/call", "name");
    for (const auto &row : rows) {
        const Entry &e = row.second;
        std::string name = row.first;
        if (name.size() > 150) name = name.substr(0, 147) + "...";
        std::printf("CPU profile: %10.3f %6.2f %10zu %12.4f  %s\n", e.seconds,
                    total > 0 ? 100.0 * e.seconds / total : 0.0, e.calls,
                    e.calls ? 1e3 * e.seconds / e.calls : 0.0, name.c_str());
    }
    std::fflush(stdout);
}

//! Times its own lifetime under name (string literal or type_name<...>()).
class Scope {
    std::string_view _name;
    std::chrono::steady_clock::time_point _start;
    bool _on;
public:
    explicit Scope(std::string_view name) : _name(name), _on(enabled()) {
        if (_on) _start = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (_on) add(_name, std::chrono::duration<double>(std::chrono::steady_clock::now() - _start).count());
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
};

} // namespace cpuProfile
