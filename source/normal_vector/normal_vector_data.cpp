#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <meltpooldg/normal_vector/normal_vector_data.hpp>

namespace MeltPoolDG::LevelSet
{
  template <typename number>
  NormalVectorData<number>::NormalVectorData()
  {
    linear_solver.solver_type         = LinearSolverType::CG;
    linear_solver.preconditioner_type = PreconditionerType::Diagonal;
  }

  template <typename number>
  void
  NormalVectorData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    /*
     *   normal vector
     */
    prm.enter_subsection("normal vector");
    {
      prm.add_parameter("filter parameter",
                        filter_parameter,
                        "normal vector computation: damping = (cell size)²  * filter parameter");
      prm.add_parameter("implementation",
                        implementation,
                        "Choose the corresponding implementation of the normal vector operation.",
                        Patterns::Selection("meltpooldg|adaflo"));
      prm.add_parameter(
        "verbosity level",
        verbosity_level,
        "Sets the maximum verbosity level of the console output. The maximum level with respect to the "
        " base value is decisive.");
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
  NormalVectorData<number>::check_input_parameters(
    const InterfaceThicknessParameterType &type) const
  {
    AssertThrow(!narrow_band.enable || narrow_band.level_set_threshold > 0.0,
                ExcMessage(
                  "The level set threshold for narrow band width must be positive! Abort..."));
    AssertThrow(linear_solver.do_matrix_free || implementation == "meltpooldg",
                ExcNotImplemented());

    AssertThrow(
      !narrow_band.enable || linear_solver.do_matrix_free,
      ExcMessage(
        "The computation of the normal vector in a narrow band is only implemented matrix-free."));
    AssertThrow(type == InterfaceThicknessParameterType::proportional_to_cell_size ||
                  implementation == "meltpooldg",
                ExcMessage("For the adaflo implementation, a variable thickness parameter epsilon "
                           "is mandatory."));
  }

  template <typename number>
  void
  NormalVectorData<number>::post()
  {
    predictor.post();
  }
  template struct NormalVectorData<double>;
} // namespace MeltPoolDG::LevelSet
