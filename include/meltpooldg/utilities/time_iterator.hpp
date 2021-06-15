#pragma once
#include <meltpooldg/utilities/conditional_ostream.hpp>

#include <iomanip>
#include <iostream>

namespace MeltPoolDG
{
  template <typename number = double>
  struct TimeIteratorData
  {
    number       start_time       = 0.0;
    number       end_time         = 1.0;
    number       time_increment   = 0.01;
    unsigned int max_n_time_steps = 1000;
    bool         CFL_condition    = false;
  };
  /*
   *  This class provides a simple time stepping routine.
   */
  template <typename number = double>
  class TimeIterator
  {
  public:
    TimeIterator() = default;

    void
    initialize(const TimeIteratorData<number> &data_in);

    bool
    is_finished() const;

    number
    get_next_time_increment();

    void
    resize_current_time_increment(const number factor);

    void
    reset_max_n_time_steps(const int time_steps_in);

    number
    get_current_time() const;

    number
    get_current_time_increment() const;

    number
    get_current_time_step_number() const;

    void
    reset();

    void
    print_me(const ConditionalOStream &pcout) const;

  private:
    TimeIteratorData<number> time_data;
    number                   old_time;
    number                   current_time;
    number                   current_time_increment;
    number                   n_time_steps;
  };

} // namespace MeltPoolDG
