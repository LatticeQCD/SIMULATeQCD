//
// Created by Hai-Tao Shu on 07.06.2019
//

#ifndef SUBLATMEAS_H
#define SUBLATMEAS_H

#include "../observables/EnergyMomentumTensor.h"
#include "../observables/PolyakovLoop.h"
#include "../../gauge/gaugefield.h"
#include "../../gauge/GaugeAction.h"

template<class floatT, bool onDevice, size_t HaloDepth>
class SubLatMeas {
protected:
    LatticeContainer<onDevice, Matrix4x4Sym<floatT> > _redBaseU;
    LatticeContainer<onDevice, floatT> _redBaseE;
    LatticeContainer<onDevice, GCOMPLEX(floatT)> _redBaseCE;
    Gaugefield<floatT, onDevice, HaloDepth> &_gauge;
    int _sub_lt;

private:
    typedef GIndexer<All, HaloDepth> GInd;
    const size_t _elems1 = GInd::getLatData().vol3;
    const size_t _spatialvol = GInd::getLatData().globvol3;
    const int _Nt = (int)GInd::getLatData().globLT;
    const int _Nx = (int)GInd::getLatData().lx;
    const int _Ny = (int)GInd::getLatData().ly;
    const int _Nz = (int)GInd::getLatData().lz;
    const size_t _elems2 = _elems1 - (_Nx-2)*(_Ny-2)*(_Nz-2);
public:
    SubLatMeas(Gaugefield<floatT, onDevice, HaloDepth> &gaugefield, int sub_lt) :
            _redBaseU(gaugefield.getComm()), _redBaseE(gaugefield.getComm()), _redBaseCE(gaugefield.getComm()),
            _gauge(gaugefield), _sub_lt(sub_lt) {

	_redBaseU.adjustSize(_elems1);
        _redBaseE.adjustSize(_elems1);
        _redBaseCE.adjustSize(_elems1);

    }

    void updateSubEMT(int pos_t, int count, MemoryAccessor &sub_E_gpu, MemoryAccessor &sub_U_gpu, std::vector<floatT> &SubBulk_Nt_p0,
         std::vector<Matrix4x4Sym<floatT>> &SubShear_Nt_p0, std::vector<floatT> &SubBulk_Nt, std::vector<Matrix4x4Sym<floatT>> &SubShear_Nt,
         int dist, int pz, int count_i, floatT displacement, int flag_real_imag);
    void updateSubPolyCorr(int pos_t, int count, MemoryAccessor &sub_poly_Nt, MemoryAccessor &sub1_cec_Nt, MemoryAccessor &sub2_cec_Nt);

    void updateSubNorm(int pos_t, std::vector<floatT> &SubTbarbp00, std::vector<floatT> &SubSbp, std::vector<GCOMPLEX(floatT)> &SubTbarbc00_SubSbc);

    GCOMPLEX(floatT) contraction_poly(MemoryAccessor &sub_poly_Nt, int min_dist);
    std::vector<GCOMPLEX(floatT)> contraction_cec(MemoryAccessor &sub1_cec_Nt, MemoryAccessor &sub2_cec_Nt, int min_dist);
};

template<class floatT>
class Contraction_cpu {
protected:
    int _sub_lt;
    int _Nt;

public:
    Contraction_cpu(int sub_lt, int Nt) : _sub_lt(sub_lt), _Nt(Nt) { }
    void ImproveNormalizeBulk(std::vector<floatT> &SubBulk_Nt_real, std::vector<floatT> &SubBulk_Nt_p0, int count);
    void ImproveNormalizeShear(std::vector<Matrix4x4Sym<floatT>> &SubShear_Nt_real, std::vector<Matrix4x4Sym<floatT>> &SubShear_Nt_p0, int count);
    void ImproveContractionBulk(std::vector<floatT> &SubBulk_Nt_real, std::vector<floatT> &SubBulk_Nt_imag, int min_dist, size_t global_spatial_vol, int pz, std::vector<floatT> &Improve_BulkResult);
    void ImproveContractionShear(std::vector<Matrix4x4Sym<floatT> > &SubShear_Nt_real, std::vector<Matrix4x4Sym<floatT> > &SubShear_Nt_imag, int min_dist, size_t global_spatial_vol, int pz, std::vector<floatT> &Improve_ShearResult);
};

#endif //SUBLATMEAS_H
