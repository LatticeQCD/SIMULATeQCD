/*
 * gaugefield.cu
 *
 * L. Mazur
 *
 */


#include "gaugefield.h"

template<class floatT>
struct fill_with_rand
{
    uint4* _rand_state;
    explicit fill_with_rand(uint4* rand_state) : _rand_state(rand_state){}

    GSU3<floatT> my_mat;

    __host__ __device__ void initialize(__attribute__((unused)) gSite site){
    }

    __device__ __host__ GSU3<floatT> operator()(gSite site, __attribute__((unused)) size_t mu){
        my_mat.random(&_rand_state[site.isite]);
        return my_mat;
    }
};

template<class floatT>
struct fill_with_gauss {
    uint4* _rand_state;
    explicit fill_with_gauss(uint4* rand_state) : _rand_state(rand_state){}

    GSU3<floatT> my_mat;

    __host__ __device__ void initialize(__attribute__((unused)) gSite site) {
    }

    __device__ __host__ GSU3<floatT> operator()(gSite site, __attribute__((unused)) size_t mu) {
        my_mat.gauss(&_rand_state[site.isite]);
        return my_mat;
    }
};

/// Kernel to unitarize all links.
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
struct UnitKernel{

    gaugeAccessor<floatT,comp> gaugeAccessor;
    explicit UnitKernel(Gaugefield<floatT,onDevice,HaloDepth,comp>& gauge) : gaugeAccessor(gauge.getAccessor()){}
    __device__ __host__ GSU3<floatT> operator()(gSiteMu siteMu){
        typedef GIndexer<All,HaloDepth> GInd;
        GSU3<double> temp;
        temp=gaugeAccessor.template getLink<double>(siteMu);
        temp.su3unitarize();
        return temp;
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::random(uint4* rand_state) {
    iterateOverBulkLoopMu(fill_with_rand<floatT>(rand_state));
    this->updateAll();
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::gauss(uint4* rand_state) {
    if (comp != R18) {
        rootLogger.error("Gaussian matrices are only possible in R18 fields!");
    } else {
        iterateOverBulkLoopMu(fill_with_gauss<floatT>(rand_state));
    }
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::one() {
    iterateWithConst(gsu3_one<floatT>());
}

/// Unitarize all the SU3 links on the lattice.
template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
void Gaugefield<floatT,onDevice,HaloDepth,comp>::su3latunitarize() {
    iterateOverFullAllMu(UnitKernel<floatT,onDevice,HaloDepth,comp>(*this));
}

//project to su3 using su3unitarize_hits
////warning on usage, convergence to su3 matrix can fail in edge cases, when input far away from proper su3 matrix
//such that convergence never happens, and algorithm gives up
template<class floatT,size_t HaloDepth,CompressionType comp>
__host__ __device__ GSU3<floatT> inline su3unitarize_project(gaugeAccessor<floatT,comp> gAcc, gaugeAccessor<floatT,comp> gAcc_base, gSiteMu siteMu) {

  typedef GIndexer<All,HaloDepth> GInd;

  GSU3<floatT> temp = gsu3_zero<floatT>();
  GSU3<floatT> temp_guess = gsu3_zero<floatT>();

  int mu = siteMu.mu;
  gSite origin = GInd::getSite(siteMu.isite);
  temp += gAcc.getLink(GInd::getSiteMu(origin,mu));
  temp_guess += gAcc_base.getLink(GInd::getSiteMu(origin,mu));

  if(std::isnan(real(temp(0,0)))) return temp;
  int status = su3unitarize_hits(&temp_guess, &temp, 9, 0.);
  assert(!status);

  return temp_guess;
}


/// Explicit instantiation double and float
#define _GLATTICE_CLASS_INIT(floatT, onDevice, HaloDepth,COMP) \
template class Gaugefield<floatT,onDevice,HaloDepth, COMP>; \


#define INIT(floatT, HALO,COMP)           \
_GLATTICE_CLASS_INIT(floatT, 0,HALO,COMP) \
_GLATTICE_CLASS_INIT(floatT, 1,HALO,COMP)
INIT_PHC(INIT)


/// Explicit instantiation half
#define _GLATTICE_CLASS_INIT_HALF(floatT, onDevice, HaloDepth) \
template class Gaugefield<floatT,onDevice,HaloDepth, R18>; \
template class Gaugefield<floatT,onDevice,HaloDepth, U3R14>; \

#define INIT_HALF(floatT, HALO)                  \
_GLATTICE_CLASS_INIT_HALF(floatT, 1,HALO)

INIT_H_HALF(INIT_HALF)

