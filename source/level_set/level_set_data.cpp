#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/level_set/level_set_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  void
  LevelSetData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("level set");
    {
      prm.add_parameter(
        "n subdivisions",
        n_subdivisions,
        "Set the number of subdivisions for the finite element of the level set operation.");
      prm.add_parameter("do localized heaviside",
                        do_localized_heaviside,
                        "Determine if the heaviside representation of the level set should be "
                        "calculated as a localized function, being exactly 0 and 1 outside of "
                        "the interface region.");

      nearest_point.add_parameters(prm);
      advec_diff.add_parameters(prm);
      normal_vec.add_parameters(prm);
      curv.add_parameters(prm);
      reinit.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  LevelSetData<number>::post()
  {
    advec_diff.post();
    normal_vec.post();
    curv.post();
    reinit.post();
  }

  template <typename number>
  void
  LevelSetData<number>::check_input_parameters(const unsigned int base_degree) const
  {
    // The level set problem for simplices can only be solved when no subdivision of the
    // finite element is undertaken.
    AssertThrow((n_subdivisions == 1 || base_degree == 1),
                ExcMessage("If you use n_subdivisions for the level set, degree must be 1."));

    advec_diff.check_input_parameters();
    normal_vec.check_input_parameters(reinit.interface_thickness_parameter.type);
    curv.check_input_parameters(reinit.interface_thickness_parameter.type);
    reinit.check_input_parameters(normal_vec.linear_solver.do_matrix_free);
  }


  template struct LevelSetData<double>;
} // namespace MeltPoolDG::LevelSet
