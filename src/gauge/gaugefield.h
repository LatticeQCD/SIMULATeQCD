/*
 * gaugefield.h
 *
 * L. Mazur
 *
 */

#pragma once

#include "../define.h"
#include "../base/math/operators.h"
#include "../base/math/su3array.h"
#include "../base/gutils.h"
#include "../base/IO/misc.h"
#include "../base/communication/siteComm.h"

template<class floatT_source, class floatT_target, bool onDevice, size_t HaloDepth, CompressionType comp>
struct convert_prec;

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp = R18>
class Gaugefield : public siteComm<floatT, onDevice, SU3Accessor<floatT, comp>, SU3<floatT>,EntryCount<comp>::count, 4, All, HaloDepth>
{
protected:
    SU3array<floatT, onDevice, comp> _lattice;
private:

    Gaugefield(const Gaugefield<floatT, onDevice, HaloDepth> &glat) = delete;

public:
    typedef GIndexer<All, HaloDepth> GInd;

    explicit Gaugefield(CommunicationBase &comm, std::string gaugefieldName="Gaugefield")
            : siteComm<floatT, onDevice, SU3Accessor<floatT,comp>, SU3<floatT>,EntryCount<comp>::count, 4, All, HaloDepth>(comm),
              _lattice(GInd::getLatData().vol4Full * 4, gaugefieldName) {
    }

    /// Assignment operator
    template<bool onDevice2>
    Gaugefield<floatT, onDevice, HaloDepth,comp> &
    operator=(const Gaugefield<floatT, onDevice2, HaloDepth,comp> &gaugeRHS) {
        _lattice.copyFrom(gaugeRHS.get_lattice_pointer());
        return *this;
    }
    Gaugefield<floatT, onDevice, HaloDepth,comp> &
    operator=(const Gaugefield<floatT, onDevice, HaloDepth,comp> &gaugeRHS) {
        _lattice.copyFrom(gaugeRHS.get_lattice_pointer());
        return *this;
    }


    const SU3array<floatT, onDevice,comp> &get_lattice_pointer() const { return _lattice; }

    void readconf_nersc(const std::string &fname);
    void readconf_nersc_host(SU3Accessor<floatT,comp> SU3Accessor, const std::string &fname);

    void readconf_ildg(const std::string &fname);
    void readconf_ildg_host(SU3Accessor<floatT,comp> SU3Accessor, const std::string &fname);

    void readconf_milc(const std::string &fname);
    void readconf_milc_host(SU3Accessor<floatT,comp> SU3Accessor, const std::string &fname);

    void readconf_openqcd(const std::string &fname);
    void readconf_openqcd_host(SU3Accessor<floatT,comp> SU3Accessor, const std::string &fname);

    void writeconf_nersc(const std::string &fname, int rows = 2, int diskprec = 1, Endianness e = ENDIAN_BIG);
    void writeconf_nersc_host(SU3Accessor<floatT, comp> SU3Accessor, const std::string &fname, int rows = 2,
                              int diskprec = 1, Endianness e = ENDIAN_BIG);

    void writeconf_ildg(const std::string &fname, LatticeParameters param);
    void writeconf_ildg_host(SU3Accessor<floatT, comp> SU3Accessor, const std::string &fname, LatticeParameters param);


    /// init lattice
    void one();                        /// set all links to one
    void random(uint4* rand_state);    /// set all links randomly
    void gauss(uint4* rand_state);     /// set all links to contain gaussian algebra elements
    void gauss_test(uint4* rand_state);

//TODO: put that into the cpp file and fix explicit instantiation macros to reduce compile time
    template<class floatT_source>
    void convert_precision(Gaugefield<floatT_source, onDevice, HaloDepth, comp> &gaugeIn) {

        iterateOverFullAllMu(convert_prec<floatT_source,floatT,onDevice, HaloDepth, comp>(gaugeIn));
    }


    void swap_memory(Gaugefield<floatT, onDevice, HaloDepth,comp> &gauge){
        _lattice.swap(gauge._lattice);
    }

    SU3Accessor<floatT, comp> getAccessor() const;

    template<unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverFullAllMu(Functor op);

    template<unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    deviceStream<onDevice> iterateOverBulkAllMu(Functor op, bool useStream = false);

    template<unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverFullLoopMu(Functor op);

    template<unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverBulkLoopMu(Functor op);

    template<uint8_t mu, unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverFullAtMu(Functor op);

    template<uint8_t mu, unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void iterateOverBulkAtMu(Functor op);

    template<typename Functor>
    Gaugefield &operator=(Functor op);

    template<unsigned BlockSize = DEFAULT_NBLOCKS, typename Object>
    void iterateWithConst(Object ob);

    /// THIS IS EXPERIMENTAL!!
    template<unsigned BlockSize = DEFAULT_NBLOCKS, typename Functor>
    void constructWithHaloUpdateAllMu(Functor op);

    void su3latunitarize();
};

template<class floatT_source, class floatT_target, bool onDevice, size_t HaloDepth, CompressionType comp>
struct convert_prec {
    SU3Accessor<floatT_source,comp> gAcc_source;

    convert_prec(Gaugefield<floatT_source, onDevice, HaloDepth, comp> &gaugeIn) : gAcc_source(gaugeIn.getAccessor()) {}

    __device__ __host__ SU3<floatT_target> operator()(gSiteMu site) {
        return gAcc_source.template getLink<floatT_target>(site);
    }
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
inline SU3Accessor<floatT, comp> Gaugefield<floatT, onDevice, HaloDepth, comp>::getAccessor() const {
    return (_lattice.getAccessor());
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
template<unsigned BlockSize, typename Object>
void Gaugefield<floatT, onDevice, HaloDepth,comp>::iterateWithConst(Object ob){
        CalcGSiteAllMuFull<All, HaloDepth> calcGSiteAllMuFull;
        WriteAtReadMu writeAtReadMu;
        this->template iterateWithConstObject<BlockSize>(ob, calcGSiteAllMuFull, writeAtReadMu, GInd::getLatData().vol4Full, 4);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
template<unsigned BlockSize, typename Functor>
void Gaugefield<floatT, onDevice, HaloDepth,comp>::iterateOverFullAllMu(Functor op) {
    CalcGSiteAllMuFull<All, HaloDepth> calcGSiteAllMuFull;
    WriteAtReadMu writeAtReadMu;
    this->template iterateFunctor<BlockSize>(op, calcGSiteAllMuFull, writeAtReadMu, GInd::getLatData().vol4Full, 4);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
template<unsigned BlockSize, typename Functor>
deviceStream<onDevice> Gaugefield<floatT, onDevice, HaloDepth, comp>::iterateOverBulkAllMu(Functor op, bool useStream) {
    CalcGSiteAllMu<All, HaloDepth> calcGSiteAllMu;
    deviceStream<onDevice> stream(useStream);
    WriteAtReadMu writeAtReadMu;
    this->template iterateFunctor<BlockSize>(op, calcGSiteAllMu, writeAtReadMu, GInd::getLatData().vol4, 4, 1, stream._stream);
    return stream;
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
template<unsigned BlockSize, typename Functor>
void Gaugefield<floatT, onDevice, HaloDepth,comp>::iterateOverFullLoopMu(Functor op) {
    CalcGSiteLoopMuFull<All, HaloDepth> calcGSiteLoopMuFull;
    WriteAtLoopMu<All, HaloDepth> writeAtLoopMu;
    this->template iterateFunctorLoop<4, BlockSize>(op, calcGSiteLoopMuFull, writeAtLoopMu, GInd::getLatData().vol4Full);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
template<unsigned BlockSize, typename Functor>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::iterateOverBulkLoopMu(Functor op) {
    CalcGSiteLoopMu<All, HaloDepth> calcGSiteLoopMu;
    WriteAtLoopMu<All, HaloDepth> writeAtLoopMu;
    this->template iterateFunctorLoop<4, BlockSize>(op, calcGSiteLoopMu, writeAtLoopMu, GInd::getLatData().vol4);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
template<uint8_t mu, unsigned BlockSize, typename Functor>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::iterateOverFullAtMu(Functor op) {
    CalcGSiteAtMuFull<mu, All, HaloDepth> calcGSiteAtMuFull;
    WriteAtReadMu writeAtReadMu;
    this->template iterateFunctor<BlockSize>(op, calcGSiteAtMuFull, writeAtReadMu, GInd::getLatData().vol4Full);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
template<uint8_t mu, unsigned BlockSize, typename Functor>
void Gaugefield<floatT, onDevice, HaloDepth, comp>::iterateOverBulkAtMu(Functor op) {
    CalcGSiteAtMu<mu, All, HaloDepth> calcGSiteAtMu;
    WriteAtReadMu writeAtReadMu;
    this->template iterateFunctor<BlockSize>(op, calcGSiteAtMu, writeAtReadMu, GInd::getLatData().vol4);
}


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
template<typename Functor>
Gaugefield<floatT, onDevice, HaloDepth, comp> &
Gaugefield<floatT, onDevice, HaloDepth, comp>::operator=(Functor op) {
    iterateOverFullAllMu(op);
    return *this;
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, typename T>
auto operator*(Gaugefield<floatT, onDevice, HaloDepth, comp> &lhs, T rhs)
{
    return general_mult(lhs, rhs);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, typename T>
auto operator*(T lhs, Gaugefield<floatT, onDevice, HaloDepth, comp> &rhs)
{
    return general_mult(lhs, rhs);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
auto operator*(Gaugefield<floatT, onDevice, HaloDepth, comp> &lhs,
               Gaugefield<floatT, onDevice, HaloDepth, comp> &rhs)
{
    return general_mult(lhs, rhs);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, typename T>
auto operator+(Gaugefield<floatT, onDevice, HaloDepth, comp> &lhs, T rhs)
{
    return general_add(lhs, rhs);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, typename T>
auto operator+(T lhs, Gaugefield<floatT, onDevice, HaloDepth, comp> &rhs)
{
    return general_add(lhs, rhs);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
auto operator+(Gaugefield<floatT, onDevice, HaloDepth, comp> &lhs,
               Gaugefield<floatT, onDevice, HaloDepth, comp> &rhs)
{
    return general_add(lhs, rhs);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, typename T>
auto operator-(Gaugefield<floatT, onDevice, HaloDepth, comp> &lhs, T rhs)
{
    return general_subtract(lhs, rhs);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, typename T>
auto operator-(T lhs, Gaugefield<floatT, onDevice, HaloDepth, comp> &rhs)
{
    return general_subtract(lhs, rhs);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp>
auto operator-(Gaugefield<floatT, onDevice, HaloDepth, comp> &lhs,
               Gaugefield<floatT, onDevice, HaloDepth, comp> &rhs)
{
    return general_subtract(lhs, rhs);
}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, typename T>
auto operator/(Gaugefield<floatT, onDevice, HaloDepth, comp> &lhs, T rhs)
{
    return general_divide(lhs, rhs);
}

