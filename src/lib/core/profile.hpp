#pragma once

#ifdef DACTI_PROFILER_ON
#include "tracy/Tracy.hpp"



#else
#define ZoneScoped
#define ZoneScopedN(...)
#endif