/**
 * @brief Collection of preprocessor directives used throughout MeltPoolDG
 */

#pragma once

#include <utility>

/**
 * Creates a lambda that acts as a simple wrapper around the given function, supporting perfect
 * argument forwarding.
 */
#define MPDG_LAMBDA_WRAPPER(func) \
  [&](auto &&...args) -> decltype(auto) { return func(std::forward<decltype(args)>(args)...); }
