#pragma once

#include <deal.II/base/exceptions.h>

#include <boost/algorithm/string/join.hpp>

#include <string>
#include <vector>


namespace MeltPoolDG
{
  class ScopedName
  {
  public:
    ScopedName(const std::string name)
      : name(name)
    {
      path.push_back(name);
    }

    ~ScopedName()
    {
      AssertThrow(path.back() == name, dealii::ExcInternalError());
      path.pop_back();
    }

    operator std::string() const
    {
      return boost::algorithm::join(path, "::");
    }

  private:
    const std::string                      name;
    inline static std::vector<std::string> path;
  };
} // namespace MeltPoolDG
