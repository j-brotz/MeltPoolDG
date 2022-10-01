#pragma once
#include <meltpooldg/utilities/conditional_ostream.hpp>
#include <meltpooldg/utilities/time_stepping_data.hpp>

#include <iomanip>
#include <iostream>

namespace MeltPoolDG
{
  /*
   *  This class provides a simple time stepping routine.
   */
  template <typename number = double>
  class TimeIterator
  {
  public:
    TimeIterator() = default;

    void
    initialize(const TimeSteppingData<number> &data_in);

    bool
    is_finished() const;

    number
    compute_next_time_increment();

    void
    resize_current_time_increment(const number factor);

    void
    set_current_time_increment(const number value);

    void
    reset_max_n_time_steps(const int time_steps_in);

    number
    get_current_time() const;

    number
    get_current_time_increment() const;

    number
    get_current_time_step_number() const;

    number
    get_old_time_increment() const;

    void
    reset();

    void
    print_me(const ConditionalOStream &pcout) const;

    /**
     * Check whether the current time increment is smaller than the @p time_step_limit.
     */
    bool
    check_time_step_limit(const number &time_step_limit);

  private:
    TimeSteppingData<number> time_data;
    number                   old_time;
    number                   current_time;
    number                   current_time_increment;
    number                   old_time_increment;
    number                   n_time_steps;
  };

} // namespace MeltPoolDG
