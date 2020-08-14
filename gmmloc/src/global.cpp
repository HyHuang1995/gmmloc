
#include "gmmloc/global.h"

namespace gmmloc {

namespace global {

std::atomic_bool pause(false);
std::atomic_bool step(false);
std::atomic_bool stop(false);

} // namespace global

} // namespace gmmloc