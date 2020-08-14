
#pragma once

#include <atomic>
#include <memory>

namespace gmmloc {

namespace global {

extern std::atomic_bool pause;
extern std::atomic_bool step;
extern std::atomic_bool stop;
} // namespace global

} // namespace gmmloc