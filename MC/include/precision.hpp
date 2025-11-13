#pragma once

#if defined(PME_USE_FLOAT_PRECISION)
using real = float;
#else
using real = double;
#endif
