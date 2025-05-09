#pragma once

#include <deal.II/base/parameter_handler.h>

#include <concepts>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace MeltPoolDG
{
  /**
   * @brief Adds parameters from a file to a ParameterHandler object.
   *
   * This free function reads user-defined parameters from a file (either JSON or PRM format),
   * parses them, and applies them to the provided `ParameterHandler` object.
   *
   * @param parameter_file Path to the parameter file.
   * @param add_parameters Lambda function to define how problem-specific parameters are added.
   * @param pcout ConditionalOStream to print the text.
   * @param enable_print If true, parameters are printed to the console.
   */
  void
  add_and_parse_parameters(const std::string                                     &parameter_file,
                           const std::function<void(dealii::ParameterHandler &)> &add_parameters,
                           const bool enable_print   = false,
                           const bool print_details  = false,
                           const bool skip_undefined = true);

  /**
   * @brief Abstract base class for managing parameter files.
   *
   * Provides an interface for processing, validating, and printing parameter files
   * using `ParameterHandler`. Derived classes must implement methods for adding
   * parameters and handling post-processing logic.
   */
  struct ParametersBase
  {
  public:
    /**
     * @brief Process a parameter file and populate the ParameterHandler.
     *
     * Reads a parameter file in `.json` or `.prm` format, validates its contents,
     * and updates the associated `ParameterHandler` object.
     *
     * @param prm The `ParameterHandler` object to populate with parameters.
     * @param parameter_filename The name of the parameter file to process.
     *
     * @throws dealii::ExcMessage If parameters have already been read or the file format is
     * unsupported.
     */
    virtual void
    process_parameters_file(dealii::ParameterHandler &prm, const std::string &parameter_filename);

    /**
     * @brief Print parameters to an output stream.
     *
     * Prints the parameters in the `ParameterHandler` to a specified stream, either
     * in detailed or compact format.
     *
     * @param prm The `ParameterHandler` object containing the parameters to print.
     * @param pcout The output stream where parameters will be printed.
     * @param print_details If `true`, prints parameters in detailed format.
     */
    void
    print_parameters(dealii::ParameterHandler &prm, std::ostream &pcout, const bool print_details);

  protected:
    /**
     * @brief Add parameters to the ParameterHandler.
     *
     * Derived classes must implement this method to define the parameters
     * managed by the `ParameterHandler`.
     *
     * @param prm The `ParameterHandler` object where parameters will be added.
     */
    virtual void
    add_parameters(dealii::ParameterHandler &prm) = 0;

    /**
     * @brief Post-processing after reading the parameter file.
     *
     * This method is called after the parameter file has been parsed. Derived
     * classes can override it to perform additional setup or validation based
     * on the file contents.
     *
     * @param parameter_filename The name of the parameter file that was processed.
     */
    virtual void
    post(const std::string &parameter_filename) = 0;

    bool parameters_read = false; /**< Tracks whether parameters have been read. */

  private:
    /**
     * @brief Check if a parameter file exists and is accessible.
     *
     * Verifies that the specified file exists and can be opened for reading.
     *
     * @param parameter_filename The name of the file to check.
     *
     * @throws dealii::ExcMessage If the file cannot be found or opened.
     */
    void
    check_for_file(const std::string &parameter_filename) const;
  };

  /**
   * @brief Concept to check if a type is derived from ParametersBase.
   *
   * Ensures that a given type `T` inherits from `ParametersBase`.
   *
   * @tparam T The type to check.
   */
  template <typename T>
  concept ParameterObject = std::is_base_of<ParametersBase, T>::value;

} // namespace MeltPoolDG
