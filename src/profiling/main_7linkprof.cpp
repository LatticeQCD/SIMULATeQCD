#include "../simulateqcd.h"
#include "../gauge/constructs/hisqForceConstructs.h"


template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int part, size_t term = 0>
class contribution_7link {
private:
    SU3Accessor<floatT, comp> _SU3Accessor;
    SU3Accessor<floatT> _forceAccessor;
    floatT _c7 = 1/48./8.;
public:
    contribution_7link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceInm);
    __host__ __device__ SU3<floatT> operator() (gSiteMu siteMu);
};

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int Part, size_t term>
contribution_7link<floatT, onDevice, HaloDepth, comp, Part, term>::contribution_7link(Gaugefield<floatT, onDevice, HaloDepth, comp> &GaugeIn, Gaugefield<floatT, onDevice, HaloDepth> &ForceIn) : _SU3Accessor(GaugeIn.getAccessor()), _forceAccessor(ForceIn.getAccessor()) {}

template<class floatT, bool onDevice, size_t HaloDepth, CompressionType comp, int Part, size_t term>
__host__ __device__ SU3<floatT> contribution_7link<floatT, onDevice, HaloDepth, comp, Part, term>::operator() (gSiteMu siteMu) {
    typedef GIndexer<All, HaloDepth> GInd;
    gSite site = GInd::getSite(siteMu.isite);
    switch (Part) {
    case 1:
        return sevenLinkContribution_1<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 2:
        return sevenLinkContribution_2<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 3:
        return sevenLinkContribution_3<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 4:
        return sevenLinkContribution_4<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 5:
        return sevenLinkContribution_5<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 6:
        return sevenLinkContribution_6<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    case 7:
        return sevenLinkContribution_7<floatT, HaloDepth, comp, term>(_SU3Accessor, _forceAccessor, site, siteMu.mu, _c7);
    default:
        return su3_zero<floatT>();
    }

}



#define PREC float

int main(int argc, char *argv[]) {
    stdLogger.setVerbosity(INFO);

    CommunicationBase commBase(&argc, &argv);
    LatticeParameters rhmc_param;

    rhmc_param.readfile(commBase, "../parameter/profiling/mrhsDSlashProf.param", argc, argv);

    commBase.init(rhmc_param.nodeDim());

    const size_t HaloDepth = 2;
    initIndexer(HaloDepth, rhmc_param,commBase);
    Gaugefield<PREC, true, HaloDepth, R18> gauge(commBase);
    Gaugefield<PREC, true, HaloDepth> force(commBase);
    Gaugefield<PREC, true, HaloDepth> forceOut(commBase);

    grnd_state<true> rand;

    initialize_rng(1337, rand);
    gauge.random(rand.state);
    force.random(rand.state);
    gauge.updateAll();
    force.updateAll();

    contribution_7link<PREC, true, HaloDepth, R18, 1, 0> F1_7link_part_10(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 1, 1> F1_7link_part_11(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 1, 2> F1_7link_part_12(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 1, 3> F1_7link_part_13(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 1, 4> F1_7link_part_14(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 1, 5> F1_7link_part_15(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 1, 6> F1_7link_part_16(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 1, 7> F1_7link_part_17(gauge,force);
    
    contribution_7link<PREC, true, HaloDepth, R18, 2, 0> F1_7link_part_20(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 2, 1> F1_7link_part_21(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 2, 2> F1_7link_part_22(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 2, 3> F1_7link_part_23(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 2, 4> F1_7link_part_24(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 2, 5> F1_7link_part_25(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 2, 6> F1_7link_part_26(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 2, 7> F1_7link_part_27(gauge,force);



    contribution_7link<PREC, true, HaloDepth, R18, 3, 0> F1_7link_part_30(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 3, 1> F1_7link_part_31(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 3, 2> F1_7link_part_32(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 3, 3> F1_7link_part_33(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 3, 4> F1_7link_part_34(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 3, 5> F1_7link_part_35(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 3, 6> F1_7link_part_36(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 3, 7> F1_7link_part_37(gauge,force);



    contribution_7link<PREC, true, HaloDepth, R18, 4, 0> F1_7link_part_40(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 4, 1> F1_7link_part_41(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 4, 2> F1_7link_part_42(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 4, 3> F1_7link_part_43(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 4, 4> F1_7link_part_44(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 4, 5> F1_7link_part_45(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 4, 6> F1_7link_part_46(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 4, 7> F1_7link_part_47(gauge,force);


    // contribution_7link<PREC, true, HaloDepth, R18, 2> F1_7link_part_2(gauge,force);
    // contribution_7link<PREC, true, HaloDepth, R18, 3> F1_7link_part_3(gauge,force);
    //contribution_7link<PREC, true, HaloDepth, R18, 4> F1_7link_part_4(gauge,force);
    // contribution_7link<PREC, true, HaloDepth, R18, 5> F1_7link_part_5(gauge,force);


    contribution_7link<PREC, true, HaloDepth, R18, 5, 0> F1_7link_part_50(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 5, 1> F1_7link_part_51(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 5, 2> F1_7link_part_52(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 5, 3> F1_7link_part_53(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 5, 4> F1_7link_part_54(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 5, 5> F1_7link_part_55(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 5, 6> F1_7link_part_56(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 5, 7> F1_7link_part_57(gauge,force);

    
    contribution_7link<PREC, true, HaloDepth, R18, 6, 0> F1_7link_part_60(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 6, 1> F1_7link_part_61(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 6, 2> F1_7link_part_62(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 6, 3> F1_7link_part_63(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 6, 4> F1_7link_part_64(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 6, 5> F1_7link_part_65(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 6, 6> F1_7link_part_66(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 6, 7> F1_7link_part_67(gauge,force);


    contribution_7link<PREC, true, HaloDepth, R18, 7, 0> F1_7link_part_70(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 7, 1> F1_7link_part_71(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 7, 2> F1_7link_part_72(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 7, 3> F1_7link_part_73(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 7, 4> F1_7link_part_74(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 7, 5> F1_7link_part_75(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 7, 6> F1_7link_part_76(gauge,force);
    contribution_7link<PREC, true, HaloDepth, R18, 7, 7> F1_7link_part_77(gauge,force);

//    contribution_7link<PREC, true, HaloDepth, R18, 6> F1_7link_part_6(gauge,force);
    // contribution_7link<PREC, true, HaloDepth, R18, 7> F1_7link_part_7(gauge,force);
    StopWatch<true> timer;

    timer.start();
    forceOut.iterateOverBulkAllMu(F1_7link_part_10);
    forceOut.iterateOverBulkAllMu(F1_7link_part_11);
    forceOut.iterateOverBulkAllMu(F1_7link_part_12);
    forceOut.iterateOverBulkAllMu(F1_7link_part_13);
    forceOut.iterateOverBulkAllMu(F1_7link_part_14);
    forceOut.iterateOverBulkAllMu(F1_7link_part_15);
    forceOut.iterateOverBulkAllMu(F1_7link_part_16);
    forceOut.iterateOverBulkAllMu(F1_7link_part_17);
    
    
    // forceOut.iterateOverBulkAllMu(F1_7link_part_2);
    forceOut.iterateOverBulkAllMu(F1_7link_part_20);
    forceOut.iterateOverBulkAllMu(F1_7link_part_21);
    forceOut.iterateOverBulkAllMu(F1_7link_part_22);
    forceOut.iterateOverBulkAllMu(F1_7link_part_23);
    forceOut.iterateOverBulkAllMu(F1_7link_part_24);
    forceOut.iterateOverBulkAllMu(F1_7link_part_25);
    forceOut.iterateOverBulkAllMu(F1_7link_part_26);
    forceOut.iterateOverBulkAllMu(F1_7link_part_27);

    
    forceOut.iterateOverBulkAllMu(F1_7link_part_30);
    forceOut.iterateOverBulkAllMu(F1_7link_part_31);
    forceOut.iterateOverBulkAllMu(F1_7link_part_32);
    forceOut.iterateOverBulkAllMu(F1_7link_part_33);
    forceOut.iterateOverBulkAllMu(F1_7link_part_34);
    forceOut.iterateOverBulkAllMu(F1_7link_part_35);
    forceOut.iterateOverBulkAllMu(F1_7link_part_36);
    forceOut.iterateOverBulkAllMu(F1_7link_part_37);
    // forceOut.iterateOverBulkAllMu(F1_7link_part_3);
    
    forceOut.iterateOverBulkAllMu(F1_7link_part_40);
    forceOut.iterateOverBulkAllMu(F1_7link_part_41);
    forceOut.iterateOverBulkAllMu(F1_7link_part_42);
    forceOut.iterateOverBulkAllMu(F1_7link_part_43);
    forceOut.iterateOverBulkAllMu(F1_7link_part_44);
    forceOut.iterateOverBulkAllMu(F1_7link_part_45);
    forceOut.iterateOverBulkAllMu(F1_7link_part_46);
    forceOut.iterateOverBulkAllMu(F1_7link_part_47);

    // forceOut.iterateOverBulkAllMu(F1_7link_part_4);
   

    forceOut.iterateOverBulkAllMu(F1_7link_part_50);
    forceOut.iterateOverBulkAllMu(F1_7link_part_51);
    forceOut.iterateOverBulkAllMu(F1_7link_part_52);
    forceOut.iterateOverBulkAllMu(F1_7link_part_53);
    forceOut.iterateOverBulkAllMu(F1_7link_part_54);
    forceOut.iterateOverBulkAllMu(F1_7link_part_55);
    forceOut.iterateOverBulkAllMu(F1_7link_part_56);
    forceOut.iterateOverBulkAllMu(F1_7link_part_57);

    // forceOut.iterateOverBulkAllMu(F1_7link_part_5);
    
    forceOut.iterateOverBulkAllMu(F1_7link_part_60);
    forceOut.iterateOverBulkAllMu(F1_7link_part_61);
    forceOut.iterateOverBulkAllMu(F1_7link_part_62);
    forceOut.iterateOverBulkAllMu(F1_7link_part_63);
    forceOut.iterateOverBulkAllMu(F1_7link_part_64);
    forceOut.iterateOverBulkAllMu(F1_7link_part_65);
    forceOut.iterateOverBulkAllMu(F1_7link_part_66);
    forceOut.iterateOverBulkAllMu(F1_7link_part_67);

    // forceOut.iterateOverBulkAllMu(F1_7link_part_6);
    
    
    forceOut.iterateOverBulkAllMu(F1_7link_part_70);
    forceOut.iterateOverBulkAllMu(F1_7link_part_71);
    forceOut.iterateOverBulkAllMu(F1_7link_part_72);
    forceOut.iterateOverBulkAllMu(F1_7link_part_73);
    forceOut.iterateOverBulkAllMu(F1_7link_part_74);
    forceOut.iterateOverBulkAllMu(F1_7link_part_75);
    forceOut.iterateOverBulkAllMu(F1_7link_part_76);
    forceOut.iterateOverBulkAllMu(F1_7link_part_77);

 
    // forceOut.iterateOverBulkAllMu(F1_7link_part_7);
    timer.stop();
    rootLogger.info("Time for 7link contribution: ", timer);
    
    return 0;

}