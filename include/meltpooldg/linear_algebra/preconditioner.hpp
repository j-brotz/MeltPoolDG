/**
 * @brief Interface for a general preconditioner.
 */

#pragma once

#include <meltpooldg/core/scratch_data.hpp>

#include <any>
#include <memory>
#include <utility>

namespace MeltPoolDG
{
  /**
   * A concept defining the requirements on a specific preconditioner implementation.
   */
  template <typename PreconditionerType, unsigned int dim, typename VectorType>
  concept PreconditionerTypeConcept =
    requires(PreconditionerType preconditioner, VectorType &vec, const VectorType &const_vec) {
      /**
       * A function that updates the preconditioner. The usage of the function argument idepends on
       * the specific preconditioner implementation.
       */
      preconditioner.update(std::declval<std::any>());

      /**
       * Apply the preconditioner to the second function argument and return the result in the first
       * function argument.
       */
      preconditioner.vmult(vec, const_vec);

      /**
       * Initilaize the preconditioner.
       */
      preconditioner.reinit(std::declval<ScratchData<dim>>(), std::declval<unsigned int>());
    };


  template <unsigned int dim, typename VectorType>
  class Preconditioner
  {
  public:
    Preconditioner() = default;

    /**
     * Constructor, stores the passed specifc preconditioner object internally.
     *
     * @param preconditioner Preconditioner object for a specific type of preconditioner.
     */
    template <typename PreconditionerType>
    explicit Preconditioner(PreconditionerType &&preconditioner)
      : preconditioner_pimpl(new PreconditionerModel<PreconditionerType>(std::move(preconditioner)))
    {}

    /**
     * Apply the preconditioner to the given @p src vector and store the result in the @p dst vector.
     *
     * @param dst Vector in which the result is stored.
     * @param src Source vector to which the preconditioner is applied.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      preconditioner_pimpl->vmult(dst, src);
    }

    /**
     * Update the preconditioner. For details see the specific preconditioner classes.
     */
    void
    update(const std::any &external_setup = std::any())
    {
      if (flag_update_preconditioner == always_update or flag_update_preconditioner == update_once)
        preconditioner_pimpl->update(external_setup);
      if (flag_update_preconditioner == update_once)
        flag_update_preconditioner = do_not_update;
    }

    /**
     * Initiliaze the preconditioner and its internal data structures. For details see the specific
     * preconditioner classes.
     */
    void
    reinit(const ScratchData<dim> &scratch_data, const unsigned int dof_idx)
    {
      preconditioner_pimpl->reinit(scratch_data, dof_idx);
    }

    /**
     * Check if the preconditioner is initialized, i.e. that the current object holds a valid
     * pointer to any preconditioner object.
     *
     * @return True if object holds a valid preconditoner object.
     */
    bool
    is_initialized() const
    {
      return (preconditioner_pimpl != nullptr);
    }

    /**
     * Delete the preconditioner object stored in this class.
     */
    void
    clear()
    {
      preconditioner_pimpl->reset(nullptr);
    }

    /**
     * Sets a flag to determine whether the preconditioner should be updated during the next
     * call to update().
     * If this function is called with @p do_update = 'true', the internal flag is set to request an
     * update, and this state persists until update() is called. Once update() is executed, the flag
     * resets automatically.
     *
     * The optional parameter @p do_overwrite allows overriding the flag's current state. If set to
     * 'true', the flag is explicitly set according to @p do_update, even if it was previously set
     * to trigger an update.
     *
     * This function can be used when multiple conditions need to be checked before deciding whether
     * to update the preconditioner.
     *
     * @param do_update If 'true', the preconditioner is marked for an update during the next call
     * to update(). If 'false', it is not updated unless previously marked.
     * @param do_overwrite If 'true', forces the flag to be set based on @p do_update, ignoring
     * previous states.
     *
     * @note By default, the preconditioner is set to update on every update() call, i.e.
     * if this function is never called, the preconditioner will always be updated during
     * update().
     */
    void
    set_do_update_preconditioner(const bool do_update, const bool do_overwrite = false)
    {
      if ((do_update || flag_update_preconditioner == update_once) && not do_overwrite)
        flag_update_preconditioner = update_once;
      else
        flag_update_preconditioner = do_not_update;
    }

  private:
    struct PreconditionerConcept
    {
      virtual ~PreconditionerConcept() = default;

      virtual void
      vmult(VectorType &dst, const VectorType &src) const = 0;

      virtual void
      update(const std::any &external_setup) = 0;

      virtual void
      reinit(const ScratchData<dim> &scratch_data, const unsigned int dof_idx) = 0;
    };

    template <PreconditionerTypeConcept<dim, VectorType> PreconditionerType>
    struct PreconditionerModel final : public PreconditionerConcept
    {
    public:
      explicit PreconditionerModel(PreconditionerType &&preconditioner_in)
        : preconditioner(std::forward<PreconditionerType>(preconditioner_in))
      {}

      void
      vmult(VectorType &dst, const VectorType &src) const override
      {
        preconditioner.vmult(dst, src);
      }

      void
      update(const std::any &external_setup) override
      {
        preconditioner.update(external_setup);
      }

      void
      reinit(const ScratchData<dim> &scratch_data, const unsigned int dof_idx) override
      {
        preconditioner.reinit(scratch_data, dof_idx);
      }

    private:
      PreconditionerType preconditioner;
    };

    //! Pointer to the actual preconditioner object to which the function calls are forwarded.
    std::unique_ptr<PreconditionerConcept> preconditioner_pimpl;

    //! Enum indicating whether the preconditioner shall be updated the next time update() is
    //! called. Defaults to alway_update until changed by set_do_update_preconditioner().
    enum
    {
      //! always update the preconditioner when calling update()
      always_update,
      //! update the precondioner only when calling update() for the next time
      update_once,
      //! do nothing when calling update()
      do_not_update
    } flag_update_preconditioner{always_update};
  };
} // namespace MeltPoolDG