#pragma once
#include <deal.II/base/mpi.h>

#include <meltpooldg/core/case_factory.hpp>

/**
 * @brief Registers a simulation case in the SimulationCaseFactory.
 *
 * This macro defines a static boolean variable that ensures the registration
 * of a specific simulation case class in the factory. The registration allows
 * the creation of instances of the given simulation case based on the provided
 * parameters.
 *
 * @tparam CaseClass The abstract simulation case class template of the respective application.
 * @tparam ConcreteCaseClass The concrete implementation of the simulation case.
 * @param case_name A unique name for the simulation case.
 * @param dim The dimensionality of the simulation case.
 * @param number The used floating point number format.
 *
 * The macro invokes `SimulationCaseFactory<CaseClass<dim, number>>::register_simulation`,
 * registering a lambda function that creates an instance of `ConcreteCaseClass<dim, number>`
 * when given a parameter file and an MPI communicator.
 */
#define MELTPOOLDG_REGISTER_CASE(CaseClass, ConcreteCaseClass, case_name, dim, number)             \
  static bool case_name_is_registered_##dim =                                                      \
    SimulationCaseFactory<CaseClass<dim, number>>::register_simulation(                            \
      case_name, [](const std::string parameter_file, const MPI_Comm mpi_communicator) {           \
        return std::make_unique<ConcreteCaseClass<dim, number>>(parameter_file, mpi_communicator); \
      });


/**
 * @brief Similar to the above, but allows an additional template parameter in the
 *        concrete simulation case that refers to the base simulation case class.
 *
 * This additional parameter enables the instantiation of the same simulation case
 * across multiple applications, allowing for more flexible use of the base case class.
 */
#define MELTPOOLDG_REGISTER_MULTI_APP_CASE(CaseClass, ConcreteCaseClass, case_name, dim, number) \
  static bool case_name_is_registered_##dim =                                                    \
    SimulationCaseFactory<CaseClass<dim, number>>::register_simulation(                          \
      case_name, [](const std::string parameter_file, const MPI_Comm mpi_communicator) {         \
        return std::make_unique<ConcreteCaseClass<dim, number, CaseClass<dim, number>>>(         \
          parameter_file, mpi_communicator);                                                     \
      });
