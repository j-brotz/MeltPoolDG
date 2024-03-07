#include <meltpooldg/flow/flow_data.hpp>

namespace MeltPoolDG::Flow
{

  template <typename number>
  void
  FlowData<number>::add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("flow");
    {
      prm.add_parameter("gravity", gravity, "Set the value for the gravity");

      surface_tension.add_parameters(prm);
      darcy_damping.add_parameters(prm);
    }
    prm.leave_subsection();
  }

  template <typename number>
  void
  FlowData<number>::post(const MaterialData<number> &material)
  {
    surface_tension.post(material);
  }

  template <typename number>
  void
  FlowData<number>::check_input_parameters(const bool curv_enable) const
  {
    surface_tension.check_input_parameters(curv_enable);
  }

  template struct FlowData<double>;
} // namespace MeltPoolDG::Flow