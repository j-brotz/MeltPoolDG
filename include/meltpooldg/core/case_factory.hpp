#pragma once
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <map>
#include <memory>
#include <string>

/**
 * @brief Factory class for registering and creating simulation cases.
 *
 * This class provides a mechanism for dynamically registering and creating
 * instances of simulation case classes based on a string identifier.
 *
 * @tparam CaseType The base class type for simulation cases.
 */
template <typename CaseType>
class SimulationCaseFactory
{
public:
  /**
   * @brief Type definition for a function that creates a simulation case.
   *
   * The function takes a parameter file and an MPI communicator as input
   * and returns a unique pointer to an instance of the simulation case.
   */
  using SimulationCreator =
    std::function<std::unique_ptr<CaseType>(const std::string, const MPI_Comm)>;

  /**
   * @brief Registers a simulation case with a unique name.
   *
   * @param name The unique identifier for the simulation case.
   * @param creator A function that creates an instance of the simulation case.
   * @return True if the registration was successful.
   * @throws Exception if a case with the same name is already registered.
   *
   * This function ensures that each simulation case is registered only once.
   */
  static bool
  register_simulation(const std::string name, SimulationCreator creator)
  {
    AssertThrow(not creators.contains(name),
                dealii::ExcMessage("Requested simulation case already registered: " + name));
    creators[name] = creator;
    return true;
  }

  /**
   * @brief Creates an instance of a registered simulation case.
   *
   * @param name The unique identifier of the simulation case to create.
   * @param parameter_file The parameter file used for initialization.
   * @param mpi_communicator The MPI communicator for parallel execution.
   * @return A unique pointer to the created simulation case instance.
   * @throws Exception if the requested case is not registered.
   *
   * This function retrieves the corresponding creator function from the factory
   * and uses it to instantiate the requested simulation case.
   */
  static std::unique_ptr<CaseType>
  create_simulation(const std::string name,
                    const std::string parameter_file,
                    const MPI_Comm    mpi_communicator)
  {
    auto it = creators.find(name);
    AssertThrow(it != creators.end(),
                dealii::ExcMessage(
                  "Requested simulation case not registered: " + name +
                  " Did you forget to create a *.cpp file for your simulation case, " +
                  "where you explicitly instantiate your case?"));
    return it->second(parameter_file, mpi_communicator);
  }

private:
  /**
   * @brief Stores registered simulation case creators.
   *
   * A static map that associates simulation case names with their corresponding
   * creator functions.
   */
  static inline std::map<std::string, SimulationCreator> creators;
};
