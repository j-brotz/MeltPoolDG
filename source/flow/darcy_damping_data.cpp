#include <meltpooldg/flow/darcy_damping_data.hpp>

namespace MeltPoolDG::Flow
{
  template <typename number>
  void
  DarcyDampingData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("darcy damping");
    {
      prm.add_parameter("mushy zone morphology",
                        mushy_zone_morphology,
                        "Mushy zone morphology for Darcy damping");
      prm.add_parameter("avoid div zero constant",
                        avoid_div_zero_constant,
                        "This parameter exists to avoid division by zero in the "
                        "Kozeny–Carman equation for the Darcy damping force.");
      prm.add_parameter("formulation",
                        formulation,
                        "Set the formulation of the Darcy damping force.");
    }
    prm.leave_subsection();
  }

  template struct DarcyDampingData<double>;
} // namespace MeltPoolDG::Flow