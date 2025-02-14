#include <meltpooldg/cut/cut_data.hpp>

namespace MeltPoolDG
{
  template <typename number>
  void
  GhostPenaltyParam<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("ghost-penalty");
    {
      prm.add_parameter("gamma M degree 0",
                        gamma_M_degree_0,
                        "Mass matrix ghost-penalty parameter for degree 0.");
      prm.add_parameter("gamma M degree 1",
                        gamma_M_degree_1,
                        "Mass matrix ghost-penalty parameter for degree 1.");
      prm.add_parameter("gamma M degree 2",
                        gamma_M_degree_2,
                        "Mass matrix ghost-penalty parameter for degree 2.");
      prm.add_parameter("gamma A degree 0",
                        gamma_A_degree_0,
                        "Stiffness matrix ghost-penalty parameter for degree 0.");
      prm.add_parameter("gamma A degree 1",
                        gamma_A_degree_1,
                        "Stiffness matrix ghost-penalty parameter for degree 1.");
      prm.add_parameter("gamma A degree 2",
                        gamma_A_degree_2,
                        "Stiffness matrix ghost-penalty parameter for degree 2.");
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  CutParam<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("stabilization");
    {
      prm.add_parameter("nitsche parameter", nitsche_parameter, "Nitsche stabilization parameter.");
      ghost_penalty.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template struct GhostPenaltyParam<double>;
  template struct CutParam<double>;
} // namespace MeltPoolDG
