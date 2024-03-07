#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include "meltpooldg/reinitialization/reinitialization_data.hpp"
#include <meltpooldg/curvature/curvature_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  CurvatureData<number>::CurvatureData()
  {
    linear_solver.solver_type         = LinearSolverType::CG;
    linear_solver.preconditioner_type = PreconditionerType::Diagonal;
  }

  template <typename number>
  void
  CurvatureData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("curvature");
    {
      prm.add_parameter(
        "enable",
        enable,
        "Set this parameter to true if curvature should be computed. This is required in case of "
        "surface tension forces.");
      prm.add_parameter(
        "do curvature correction",
        do_curvature_correction,
        "Set this parameter to true if the curvature value at the discrete interface "
        "i.e. where the level set is 0, should be extended to the interface region.");
      prm.add_parameter("filter parameter",
                        filter_parameter,
                        "curvature computation: damping = (cell size)² * filter parameter");
      prm.add_parameter("implementation",
                        implementation,
                        "Choose the corresponding implementation of the curvature operation.",
                        Patterns::Selection("meltpooldg|adaflo"));
      prm.add_parameter(
        "verbosity level",
        verbosity_level,
        "Sets the maximum verbosity level of the console output. The maximum level with respect to the "
        "base value is decisive.");

      prm.enter_subsection("narrow band");
      {
        prm.add_parameter(
          "enable",
          narrow_band.enable,
          "Set this parameter to true to compute the normal vector only in the interfacial region.");
        prm.add_parameter(
          "level set threshold",
          narrow_band.level_set_threshold,
          "If narrow band is enabled to true this parameter determines the level set "
          "treshold for the narrow band.");
      }
      prm.leave_subsection();

      predictor.add_parameters(prm);
      linear_solver.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  CurvatureData<number>::check_input_parameters(const InterfaceThicknessParameterType &type) const
  {
    AssertThrow(type == InterfaceThicknessParameterType::proportional_to_cell_size ||
                  implementation == "meltpooldg",
                ExcMessage("For the adaflo implementation, a variable thickness parameter epsilon "
                           "is mandatory."));

    AssertThrow(!narrow_band.enable || narrow_band.level_set_threshold > 0.0,
                ExcMessage(
                  "The level set threshold for narrow band width must be positive! Abort..."));
    AssertThrow(linear_solver.do_matrix_free || implementation == "meltpooldg",
                ExcNotImplemented());
    AssertThrow(
      !narrow_band.enable || linear_solver.do_matrix_free,
      ExcMessage(
        "The computation of the curvature in a narrow band is only implemented matrix-free."));
  }

  template <typename number>
  void
  CurvatureData<number>::post()
  {
    predictor.post();
  }

  template struct CurvatureData<double>;
} // namespace MeltPoolDG::LevelSet
