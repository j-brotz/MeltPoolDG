#define MELTPOOLDG_REGISTER_CASE(CaseClass, ConcreteCaseClass, case_name, dim)             \
  static bool case_name_is_registered_##dim =                                              \
    SimulationCaseFactory<CaseClass<dim>>::register_simulation(                            \
      case_name, [](const std::string parameter_file, const MPI_Comm mpi_communicator) {   \
        return std::make_unique<ConcreteCaseClass<dim>>(parameter_file, mpi_communicator); \
      });
