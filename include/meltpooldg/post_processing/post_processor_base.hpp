/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2022
 *
 * ---------------------------------------------------------------------*/

namespace MeltPoolDG
{
  namespace PostProcessing
  {
    using namespace dealii;

    class PostProcessorBase
    {
    public:
      virtual void
      process(const unsigned int n_time_step) = 0;
    };
  } // namespace PostProcessing
} // namespace MeltPoolDG
