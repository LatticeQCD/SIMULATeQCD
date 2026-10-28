#include "dslash.h"
#include "cpuHisqDslash.h"
#ifdef CPU_HISQ_DSLASH_SIMD
#include <limits>
#include <mutex>
#include <type_traits>
#endif

#ifdef USE_HIP_AMD
#define BLOCKSIZE 64
#else
#define BLOCKSIZE 32
#endif



#ifdef CPU_HISQ_DSLASH_SIMD
//! Index table of the Dslash acting on a spinor of layout LatLayoutRHS (output on the other
//! parity), built with exactly the indexer calls of HisqDslashFunctor::operator(). Cached per
//! lattice geometry; nullptr if an index does not fit into 32 bits.
template<Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
const cpuSimd::DslashTable *cpuHisqDslashTable(const size_t nSites, const int width) {
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;
    static cpuSimd::DslashTable tables[2];  // width 8 (double) and 16 (float)
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    cpuSimd::DslashTable &t = tables[width == 8 ? 0 : 1];
    const auto lat = GInd::getLatData();
    if (t.width == width && t.nSites == nSites && t.lx == lat.lxFull && t.ly == lat.lyFull
        && t.lz == lat.lzFull && t.lt == lat.ltFull) {
        return &t;
    }
    t = cpuSimd::DslashTable();
    const size_t nBlocks = (nSites + width - 1) / width;
    std::vector<int32_t> out(nBlocks * width), spinor(nBlocks * 16 * width), link(nBlocks * 16 * width);
    std::vector<uint8_t> stride(nBlocks * 33);
    bool fits = true;
    constexpr size_t maxIndex = std::numeric_limits<int32_t>::max() / 2;
#pragma omp parallel for schedule(static) reduction(&&:fits)
    for (size_t i = 0; i < nBlocks * width; ++i) {
        const size_t block = i / width, lane = i % width;
        const gSiteStack site = GInd::getSiteStack(i < nSites ? i : nSites - 1, 0);
        size_t index[33];
        index[32] = site.isiteStackFull;
        for (int mu = 0; mu < 4; mu++) {
            index[4 * mu + 0] = GInd::site_up(site, mu).isiteStackFull;
            index[4 * mu + 1] = GInd::site_dn(site, mu).isiteStackFull;
            index[4 * mu + 2] = GInd::site_up_up_up(site, mu, mu, mu).isiteStackFull;
            index[4 * mu + 3] = GInd::site_dn_dn_dn(site, mu, mu, mu).isiteStackFull;
            index[16 + 4 * mu + 0] = GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)).indexMuFull;
            index[16 + 4 * mu + 1] = GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)).indexMuFull;
            index[16 + 4 * mu + 2] = GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)).indexMuFull;
            index[16 + 4 * mu + 3] = GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)).indexMuFull;
        }
        for (size_t value : index) fits = fits && value <= maxIndex;
        out[i] = static_cast<int32_t>(2 * index[32]);
        for (int k = 0; k < 16; ++k) {
            spinor[(block * 16 + k) * width + lane] = static_cast<int32_t>(2 * index[k]);
            link[(block * 16 + k) * width + lane] = static_cast<int32_t>(2 * index[16 + k]);
        }
    }
    if (!fits) return nullptr;
    // The padded last block repeats one site, so its strides are irregular: it is gathered
    // and scattered with a lane mask.
#pragma omp parallel for schedule(static)
    for (size_t block = 0; block < nBlocks; ++block) {
        for (int k = 0; k < 16; ++k) {
            stride[block * 33 + k] = cpuSimd::strideOf(&spinor[(block * 16 + k) * width], width);
            stride[block * 33 + 16 + k] = cpuSimd::strideOf(&link[(block * 16 + k) * width], width);
        }
        // Only contiguous output is written with plain stores.
        stride[block * 33 + 32] = cpuSimd::strideOf(&out[block * width], width) == 1 ? 1 : 0;
    }
    // A block is interior if all its spinor operands are bulk sites of the input.
    typedef GIndexer<LatLayoutRHS, HaloDepthSpin> GIndIn;
    const auto latIn = GIndIn::getLatData();
    std::vector<char> isBulk(LatLayoutRHS == All ? latIn.vol4Full : latIn.sizehFull, 0);
    const size_t nBulkIn = LatLayoutRHS == All ? latIn.vol4 : latIn.sizeh;
    for (size_t i = 0; i < nBulkIn; ++i) isBulk[GIndIn::getSiteStack(i, size_t(0)).isiteStackFull] = 1;
    std::vector<uint32_t> interior, boundary;
    for (size_t block = 0; block < nBlocks; ++block) {
        bool inner = true;
        for (size_t k = 0; k < 16 * size_t(width) && inner; ++k) inner = isBulk[spinor[block * 16 * width + k] / 2];
        (inner ? interior : boundary).push_back(static_cast<uint32_t>(block));
    }
    size_t count[3] = {0, 0, 0};
    for (size_t block = 0; block < nBlocks; ++block)
        for (int k = 0; k < 32; ++k) count[stride[block * 33 + k]]++;
    const double operands = 32.0 * nBlocks;
    rootLogger.info("CPU SIMD Dslash table: ", nSites, " sites, width ", width, ", operands with stride 1 / 2 / gathered: ",
                    sformat("%.1f", 100 * count[1] / operands), "% / ", sformat("%.1f", 100 * count[2] / operands),
                    "% / ", sformat("%.1f", 100 * count[0] / operands), "%; interior blocks ",
                    sformat("%.1f", 100.0 * interior.size() / nBlocks), "%");
    t.lx = lat.lxFull; t.ly = lat.lyFull; t.lz = lat.lzFull; t.lt = lat.ltFull;
    t.nSites = nSites;
    t.nBlocks = nBlocks;
    t.width = width;
    t.out = std::move(out);
    t.spinor = std::move(spinor);
    t.link = std::move(link);
    t.stride = std::move(stride);
    t.interior = std::move(interior);
    t.boundary = std::move(boundary);
    return &t;
}

//! Which sites cpuHisqDslash works on: all, or only the blocks that read only bulk sites of rhs
//! (interior) or the rest (boundary).
enum class CpuDslashPart { all, interior, boundary };

//! Lets MPI progress a pending halo exchange (called by one thread during the interior Dslash).
inline void cpuMpiProgress() {
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
}

//! lhs = alpha * Dslash(rhs) + beta * aux with the AVX-512 kernel, for every stack. Returns
//! false (and does nothing) where the kernel does not apply: SIMULATEQCD_CPU_SIMD=0.
template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, bool onDevice, size_t NStacks>
bool cpuHisqDslash(Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> &lhs,
                   const Spinorfield<floatT, onDevice, LatLayoutRHS, HaloDepthSpin, NStacks> &rhs,
                   Gaugefield<floatT, onDevice, HaloDepthGauge, R18> &gaugeSmeared,
                   Gaugefield<floatT, onDevice, HaloDepthGauge, U3R14> &gaugeNaik, const floatT c3000,
                   const floatT alpha,
                   const Spinorfield<floatT, onDevice, LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin, NStacks> *aux,
                   const floatT beta, const CpuDslashPart part = CpuDslashPart::all) {
    if constexpr (!(std::is_same_v<floatT, double> || std::is_same_v<floatT, float>)) {
        return false;
    } else {
        if (!cpuSimd::enabled()) return false;
        using S = std::conditional_t<std::is_same_v<floatT, double>, cpuSimd::VecD, cpuSimd::VecF>;
        const cpuSimd::DslashTable *table =
            cpuHisqDslashTable<LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>(lhs.getNumberLatticePoints(), S::W);
        if (!table) return false;
        constexpr bool isDouble = std::is_same_v<floatT, double>;
        const char *name[2][3] = {{"cpuSimd::hisqDslash<float>", "cpuSimd::hisqDslash<float> interior",
                                   "cpuSimd::hisqDslash<float> boundary"},
                                  {"cpuSimd::hisqDslash<double>", "cpuSimd::hisqDslash<double> interior",
                                   "cpuSimd::hisqDslash<double> boundary"}};
        const char *nameStacked[2][3] = {{"cpuSimd::hisqDslash<float, stacked>", "cpuSimd::hisqDslash<float, stacked> interior",
                                          "cpuSimd::hisqDslash<float, stacked> boundary"},
                                         {"cpuSimd::hisqDslash<double, stacked>", "cpuSimd::hisqDslash<double, stacked> interior",
                                          "cpuSimd::hisqDslash<double, stacked> boundary"}};
        cpuProfile::Scope profile((NStacks == 1 ? name : nameStacked)[isDouble][static_cast<int>(part)]);
        // Distance (in reals) from stack s to stack s+1 in every entry array.
        typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;
        const size_t stackStride = 2 * (GInd::getSiteStack(size_t(0), size_t(1)).isiteStackFull - GInd::getSiteStack(size_t(0), size_t(0)).isiteStackFull);
        const auto lhsAcc = lhs.getAccessor();
        const auto rhsAcc = rhs.getAccessor();
        const auto fatAcc = gaugeSmeared.getAccessor();
        const auto naikAcc = gaugeNaik.getAccessor();
        floatT *out[3];
        const floatT *in[3], *auxPtr[3], *fat[9], *naik[7];
        for (int c = 0; c < 3; ++c) {
            out[c] = reinterpret_cast<floatT *>(lhsAcc.getEntryPointer(c));
            in[c] = reinterpret_cast<const floatT *>(rhsAcc.getEntryPointer(c));
            auxPtr[c] = aux ? reinterpret_cast<const floatT *>(aux->getAccessor().getEntryPointer(c)) : nullptr;
        }
        for (int e = 0; e < 9; ++e) fat[e] = reinterpret_cast<const floatT *>(fatAcc.getEntryPointer(e));
        for (int e = 0; e < 7; ++e) naik[e] = reinterpret_cast<const floatT *>(naikAcc.getEntryPointer(e));
        const std::vector<uint32_t> *blocks = part == CpuDslashPart::interior ? &table->interior
                                              : part == CpuDslashPart::boundary ? &table->boundary : nullptr;
        cpuSimd::hisqDslash<S, NStacks>(*table, out, in, fat, naik, static_cast<floatT>(C_1000), c3000, alpha,
                                        aux ? auxPtr : nullptr, beta, stackStride,
                                        blocks ? blocks->data() : nullptr, blocks ? blocks->size() : 0,
                                        part == CpuDslashPart::interior ? cpuMpiProgress : nullptr);
        return true;
    }
}
#endif

//! HisqDslash
template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto HisqDslashFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site) const{
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    Vect3<floatT> Stmp(0.0);

    #ifdef USE_HIP
    static_for<0,4>::apply([&](auto mu) {

        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                            * _spinorIn.getElement(GInd::site_up(site, mu));

        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                            * _spinorIn.getElement(GInd::site_dn(site, mu));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                             * _spinorIn.getElement(GInd::site_up_up_up(site, mu, mu, mu));

        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                             * _spinorIn.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));
    });
    #else
    #pragma unroll
    for (int mu = 0; mu < 4; mu++) {
        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                            * _spinorIn.getElement(GInd::site_up(site, mu));

        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                            * _spinorIn.getElement(GInd::site_dn(site, mu));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                             * _spinorIn.getElement(GInd::site_up_up_up(site, mu, mu, mu));

        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                             * _spinorIn.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));
    }
    #endif
    return Stmp;
}


template<bool onDevice, class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
__host__ __device__ void HisqDslashStackedFunctor<onDevice, floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::operator()(gSiteStack site) {
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;
    size_t stack_offset = GInd::getStack(site);
        SimpleArray<Vect3<floatT>, NStacks> Stmp((floatT)0.0);
        
        #ifdef USE_HIP
        constexpr size_t Ntiles = NStacks/NStacks_blockdim;
        static_for<0,4>::apply([&](auto mu) {
            size_t stack = stack_offset;
            static_for<0,Ntiles>::apply([&](auto i) {

                Stmp[i] += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                                    * _spinorIn.getElement(GInd::site_up(GInd::getSiteStack(site,stack), mu));

                Stmp[i] -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_dn(GInd::getSiteStack(site,stack), mu));

                Stmp[i] += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_up_up_up(GInd::getSiteStack(site,stack), mu, mu, mu));

                Stmp[i] -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_dn_dn_dn(GInd::getSiteStack(site,stack), mu, mu, mu));

                stack += NStacks_blockdim;
            });
        });
        #else
 
        #pragma unroll
        for (int mu = 0; mu < 4; mu++) {
            size_t i = 0;
            #pragma unroll
            for (size_t stack = stack_offset; stack-stack_offset < NStacks; stack+=NStacks_blockdim, i++) {

                Stmp[i] += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                                    * _spinorIn.getElement(GInd::site_up(GInd::getSiteStack(site,stack), mu));

                Stmp[i] -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_dn(GInd::getSiteStack(site,stack), mu));

                Stmp[i] += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_up_up_up(GInd::getSiteStack(site,stack), mu, mu, mu));

                Stmp[i] -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                                    * _spinorIn.getElement(GInd::site_dn_dn_dn(GInd::getSiteStack(site,stack), mu, mu, mu));
            }
       
        }
        #endif
    

    // size_t i = 0;
    // #pragma unroll NStacks
    // for (size_t stack = stack_offset; i < Ntiles; stack+=NStacks_blockdim, i++) {
 
    #ifdef USE_HIP
    size_t stack = stack_offset;
    static_for<0,Ntiles>::apply([&](auto i) {
        const gSiteStack writeSite = GInd::getSiteStack(site,stack);
        _spinorOut.setElement(writeSite,Stmp[i]);
        stack += NStacks_blockdim;
    });
    #else
    size_t i = 0;
    #pragma unroll NStacks
    for (size_t stack = stack_offset; stack-stack_offset < NStacks; stack+=NStacks_blockdim, i++) {
        const gSiteStack writeSite = GInd::getSiteStack(site,stack);
        _spinorOut.setElement(writeSite,Stmp[i]);
    }   
    #endif
}


template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto HisqMdaggMFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site){
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    Vect3<floatT> Stmp(0.0);
    for (int mu = 0; mu < 4; mu++) {

        Stmp += static_cast<floatT>(C_1000) * _gAcc_smeared.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu)))
                                            * _spinorTmp.getElement(GInd::site_up(site, mu));

        Stmp -= static_cast<floatT>(C_1000) * _gAcc_smeared.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu)))
                                            * _spinorTmp.getElement(GInd::site_dn(site, mu));

        Stmp += static_cast<floatT>(_c_3000) * _gAcc_Naik.getLink(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_up(site, mu), mu)))
                                             * _spinorTmp.getElement(GInd::site_up_up_up(site, mu, mu, mu));

        Stmp -= static_cast<floatT>(_c_3000) * _gAcc_Naik.getLinkDagger(GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn_dn(site, mu, mu), mu)))
                                             * _spinorTmp.getElement(GInd::site_dn_dn_dn(site, mu, mu, mu));
    }
    return _spinorIn.getElement(site)*_mass2 - Stmp;
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::Dslash(SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
    if (NStacks * NStacks_blockdim == 1) {
        Dslash_nostack(lhs,rhs,update);
    }
    else {
        if (NStacks % NStacks_blockdim != 0) {
            throw std::runtime_error(stdLogger.fatal("Error in Dslash call: Nstacks not divisible by NStacks_blockdim!"));
        }
        Dslash_stacked(lhs,rhs,update);
    }
}


template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::Dslash_nostack(SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
    // The getFunctor calls the DSlash functor. This is to clean up the DSlash functor call.
#ifdef CPU_HISQ_DSLASH_SIMD
    if (!cpuHisqDslash<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>(
            lhs, rhs, _gauge_smeared, _gauge_Naik, _c_3000, floatT(1),
            static_cast<const SpinorLHS_t *>(nullptr), floatT(0)))
#endif
    lhs.template iterateOverBulk<BLOCKSIZE>(getFunctor(rhs));
    if(update){
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::Dslash_stacked(SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
   
#ifdef CPU_HISQ_DSLASH_SIMD
    if (!cpuHisqDslash<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>(
            lhs, rhs, _gauge_smeared, _gauge_Naik, _c_3000, floatT(1),
            static_cast<const SpinorLHS_t *>(nullptr), floatT(0)))
#endif
    {
    HisqDslashStackedFunctor<onDevice, floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim> dslash_func(lhs, rhs,_gauge_smeared,_gauge_Naik,_c_3000);
   
    CalcGSiteStack<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> calcGSite;
   
    iterateFunctorNoReturn<onDevice,BLOCKSIZE>(dslash_func,calcGSite,lhs.getNumberLatticePoints(),NStacks_blockdim);
    }

    if(update) {
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::applyMdaggM_nostack(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){

#ifdef CPU_HISQ_DSLASH_SIMD
    // Several ranks: each Dslash computes the sites that need no halo of its input while the
    // halo exchange of that input runs, then the sites next to the halos. For the first one
    // only if the caller started the exchange of spinorIn (updateInputHalos).
    SpinorRHS_t &in = const_cast<SpinorRHS_t &>(spinorIn);  // only its halos are written
    const bool inPending = in.haloExchangePending();
    if (cpuSimd::overlapEnabled() && _tmpSpin.getComm().getNumberProcesses() > 1
        && cpuHisqDslash<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>(
               _tmpSpin, spinorIn, _gauge_smeared, _gauge_Naik, _c_3000, floatT(1),
               static_cast<const SpinorLHS_t *>(nullptr), floatT(0),
               inPending ? CpuDslashPart::interior : CpuDslashPart::all)) {
        if (inPending) {
            in.updateAll(COMM_FINISH | Hyperplane);
            cpuHisqDslash<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>(
                _tmpSpin, spinorIn, _gauge_smeared, _gauge_Naik, _c_3000, floatT(1),
                static_cast<const SpinorLHS_t *>(nullptr), floatT(0), CpuDslashPart::boundary);
        }
        _tmpSpin.updateAll(COMM_START | Hyperplane);
        const floatT alpha = _mass != 0.0 ? floatT(-1) : floatT(1);
        const SpinorRHS_t *aux = _mass != 0.0 ? &spinorIn : nullptr;
        const floatT beta = _mass != 0.0 ? floatT(_mass2) : floatT(0);
        cpuHisqDslash<floatT, LayoutSwitcher<LatLayoutRHS>(), HaloDepthGauge, HaloDepthSpin>(
            spinorOut, _tmpSpin, _gauge_smeared, _gauge_Naik, _c_3000, alpha, aux, beta, CpuDslashPart::interior);
        _tmpSpin.updateAll(COMM_FINISH | Hyperplane);
        cpuHisqDslash<floatT, LayoutSwitcher<LatLayoutRHS>(), HaloDepthGauge, HaloDepthSpin>(
            spinorOut, _tmpSpin, _gauge_smeared, _gauge_Naik, _c_3000, alpha, aux, beta, CpuDslashPart::boundary);
        if (update) spinorOut.updateAll();
        return;
    }
    if (inPending) in.updateAll(COMM_FINISH | Hyperplane);
#endif

    Dslash(_tmpSpin, spinorIn, true);

#ifdef CPU_HISQ_DSLASH_SIMD
    // spinorOut = mass^2 * spinorIn - Dslash(_tmpSpin), or Dslash(_tmpSpin) for zero mass.
    if (_mass != 0.0
        ? cpuHisqDslash<floatT, LayoutSwitcher<LatLayoutRHS>(), HaloDepthGauge, HaloDepthSpin>(
              spinorOut, _tmpSpin, _gauge_smeared, _gauge_Naik, _c_3000, floatT(-1), &spinorIn, _mass2)
        : cpuHisqDslash<floatT, LayoutSwitcher<LatLayoutRHS>(), HaloDepthGauge, HaloDepthSpin>(
              spinorOut, _tmpSpin, _gauge_smeared, _gauge_Naik, _c_3000, floatT(1),
              static_cast<const SpinorRHS_t *>(nullptr), floatT(0))) {
    } else
#endif
    // Optimization: We might get a speed up if we put this in a custom operator
    if(_mass != 0.0) {
        spinorOut.template iterateOverBulk<BLOCKSIZE>(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    } else {
        spinorOut.template iterateOverBulk<BLOCKSIZE>(getFunctor(_tmpSpin));
    }

    if(update)
        spinorOut.updateAll();
}


template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::applyMdaggM_stacked(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){
    Dslash(_tmpSpin, spinorIn, true);

#ifdef CPU_HISQ_DSLASH_SIMD
    // spinorOut = mass^2 * spinorIn - Dslash(_tmpSpin), or Dslash(_tmpSpin) for zero mass.
    if (_mass != 0.0
        ? cpuHisqDslash<floatT, LayoutSwitcher<LatLayoutRHS>(), HaloDepthGauge, HaloDepthSpin>(
              spinorOut, _tmpSpin, _gauge_smeared, _gauge_Naik, _c_3000, floatT(-1), &spinorIn, _mass2)
        : cpuHisqDslash<floatT, LayoutSwitcher<LatLayoutRHS>(), HaloDepthGauge, HaloDepthSpin>(
              spinorOut, _tmpSpin, _gauge_smeared, _gauge_Naik, _c_3000, floatT(1),
              static_cast<const SpinorRHS_t *>(nullptr), floatT(0))) {
    } else
#endif
    {
    HisqDslashStackedFunctor<onDevice, floatT, LayoutSwitcher<LatLayoutRHS>(), HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim> dslash_2nd(spinorOut,_tmpSpin, _gauge_smeared, _gauge_Naik, _c_3000);
    CalcGSiteStack<LatLayoutRHS, HaloDepthSpin> calcGSite;

    iterateFunctorNoReturn<onDevice,BLOCKSIZE>(dslash_2nd,calcGSite,spinorOut.getNumberLatticePoints(),NStacks_blockdim);
    if (_mass != 0.0) {
        spinorOut = (spinorIn * _mass2) - spinorOut;
    }
    }

    if(update) {
        spinorOut.updateAll();
    }

}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
void HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::applyMdaggM(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){
    if (NStacks * NStacks_blockdim == 1) {
        applyMdaggM_nostack(spinorOut,spinorIn,update);
    }
    else {
        if (NStacks % NStacks_blockdim != 0) {
            throw std::runtime_error(stdLogger.fatal("Error in Dslash call: Nstacks not divisible by NStacks_blockdim!"));
        }
        applyMdaggM_stacked(spinorOut, spinorIn, update);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks, size_t NStacks_blockdim>
template<Layout LatLayout>
HisqDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin> HisqDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks, NStacks_blockdim>::getFunctor(const Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks>& rhs){
    return HisqDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>(rhs, _gauge_smeared, _gauge_Naik, _c_3000);
}


//! stdStagDslash
template<class floatT, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin>
__host__ __device__ auto stdStagDslashFunctor<floatT, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin>::operator()(gSiteStack site) const{
    typedef GIndexer<LayoutSwitcher<LatLayoutRHS>(), HaloDepthSpin> GInd;

    Vect3<floatT> Stmp(0.0);
    floatT phase =1.0;
    floatT up_bound=1.0;
    floatT down_bound=1.0;

    for (int mu = 0; mu < 4; mu++) {

        int rest=site.coord.x%2;
        if (rest == 1 && mu == 1) phase= -1.0;

        rest = (site.coord.x + site.coord.y) % 2;
        if (rest == 1 && mu == 2) phase= -1.0;

        rest = (site.coord.x + site.coord.y + site.coord.z) % 2;
        if (rest == 1 && mu == 3) phase= -1.0;

        sitexyzt localCoord = site.coord;
        sitexyzt globalCoord = GInd::getLatData().globalPos(localCoord);

        if(mu==3 && (globalCoord.t == (int) GInd::getLatData().globLT - 1)) up_bound = -1.0;
        if(mu==3 && (globalCoord.t == 0)) down_bound = -1.0;

        Stmp += static_cast<floatT>(C_1000) * phase * up_bound *_gAcc.getLink(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(site, mu))) *
            _spinorIn.getElement(GInd::site_up(site, mu));
        Stmp -= static_cast<floatT>(C_1000) * phase * down_bound * _gAcc.getLinkDagger(
                GInd::template convertSite<All, HaloDepthGauge>(GInd::getSiteMu(GInd::site_dn(site, mu), mu))) *
            _spinorIn.getElement(GInd::site_dn(site, mu));
    }
    return Stmp;
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::Dslash(
        SpinorLHS_t& lhs, const SpinorRHS_t& rhs, bool update){
    lhs.template iterateOverBulk(getFunctor(rhs));
    if(update){
        lhs.updateAll(COMM_BOTH | Hyperplane);
    }
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
void stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::applyMdaggM(SpinorRHS_t& spinorOut, const SpinorRHS_t& spinorIn, bool update){

    Dslash(_tmpSpin, spinorIn, true);
    // This is not wrong! One Dslash is hiden in the line below in the getFunctor() call!
    // Optimization: We might get a speed up if we put this in a custom operator
    spinorOut.template iterateOverBulk(general_subtract(spinorIn * _mass2, getFunctor(_tmpSpin)));
    if(update)
        spinorOut.updateAll();
}

template<typename floatT, bool onDevice, Layout LatLayoutRHS, size_t HaloDepthGauge, size_t HaloDepthSpin, size_t NStacks>
template<Layout LatLayout>
stdStagDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin> stdStagDSlash<floatT, onDevice, LatLayoutRHS, HaloDepthGauge, HaloDepthSpin, NStacks>::getFunctor(const Spinorfield<floatT, onDevice, LatLayout, HaloDepthSpin, NStacks>& rhs){
    return stdStagDslashFunctor<floatT, LatLayout, HaloDepthGauge, HaloDepthSpin>(rhs, _gauge);
}



//! explicit template instantiations
#define DSLASH_INIT(floatT, LO, HaloDepth, HaloDepthSpin, NStacks) \
  template class HisqDSlash<floatT,false,LO,HaloDepth,HaloDepthSpin,NStacks>;\
  template class stdStagDSlash<floatT,false,LO,HaloDepth,HaloDepthSpin,NStacks>;\
  template class stdStagDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks>;
INIT_PLHHSN(DSLASH_INIT)

#define DSLASH_INIT2(floatT, LO, HaloDepth, HaloDepthSpin, NStacks, NStacks_blockdim) \
  template class HisqDSlash<floatT,true,LO,HaloDepth,HaloDepthSpin,NStacks, NStacks_blockdim>;
INIT_PLHHSNNB_HALF(DSLASH_INIT2)

