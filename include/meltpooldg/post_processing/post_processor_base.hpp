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

    template <int dim>
    class PostProcessorBase
    {
      public:
        virtual void process()=0;
    };
  }
}

