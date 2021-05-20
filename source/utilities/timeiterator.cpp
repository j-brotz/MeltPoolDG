#include <meltpooldg/utilities/timeiterator.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  TimeIterator<number>::initialize(const TimeIteratorData<number> &data_in)
  {
    time_data              = data_in;
    current_time           = data_in.start_time;
    current_time_increment = data_in.time_increment;
    n_time_steps           = 0;
  }

  template <typename number>
  bool
  TimeIterator<number>::is_finished() const
  {
    // number of maximum steps is reached
    if (n_time_steps >= time_data.max_n_time_steps)
      return true;
    // current_time is larger than end time
    if (current_time > time_data.end_time)
      return true;
    // current_time and end_time are (almost) equal
    else if (std::abs(current_time - time_data.end_time) <= std::numeric_limits<double>::epsilon())
      return true;
    // current_time is smaller than end_time and number of maximum steps is not reached
    else
      return false;
  }

  template <typename number>
  number
  TimeIterator<number>::get_next_time_increment()
  {
    if (current_time + current_time_increment > time_data.end_time)
      current_time_increment = time_data.end_time - current_time;
    current_time += current_time_increment;
    n_time_steps += 1;
    return current_time_increment;
  }

  template <typename number>
  void
  TimeIterator<number>::resize_current_time_increment(const number factor)
  {
    current_time_increment *= factor;
  }

  template <typename number>
  void
  TimeIterator<number>::reset_max_n_time_steps(const int time_steps_in)
  {
    time_data.max_n_time_steps = time_steps_in;
  }

  template <typename number>
  number
  TimeIterator<number>::get_current_time() const
  {
    return current_time;
  }

  template <typename number>
  number
  TimeIterator<number>::get_current_time_increment() const
  {
    return current_time_increment;
  }

  template <typename number>
  number
  TimeIterator<number>::get_current_time_step_number() const
  {
    return n_time_steps;
  }

  template <typename number>
  void
  TimeIterator<number>::reset()
  {
    n_time_steps           = 0;
    current_time           = time_data.start_time;
    current_time_increment = time_data.time_increment;
  }

  template <typename number>
  void
  TimeIterator<number>::print_me(std::ostream &pcout) const
  {
    pcout << "      | Time step " << n_time_steps << " at t=" << std::fixed << std::setprecision(5)
          << current_time << std::endl;
  }

  template class TimeIterator<>;
} // namespace MeltPoolDG
