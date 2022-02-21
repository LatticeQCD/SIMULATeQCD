/* 
 * main_correlatorTest.cu
 *
 * D. Clarke
 *
 * Test and playground for correlator class.
 *
 */

#include "../SIMULATeQCD.h"
#include "../modules/gaugeFixing/PolyakovLoopCorrelator.h"

#define PREC double 
#define HAVECONFIG true

/// INTENT:  IN--CPUcorr, CPUnorm, testName, r2max, referenceValue; OUT--lerror
void compareWithOne(Correlator<false,PREC> &CPUcorr, Correlator<false,PREC> &CPUnorm, std::string testName,
                    int r2max, bool &lerror)
{
    PREC norm, corrResult;
    LatticeContainerAccessor _CPUcorr(CPUcorr.getAccessor());
    LatticeContainerAccessor _CPUnorm(CPUnorm.getAccessor());
    for(int ir2=0; ir2<r2max+1; ir2++) {
        _CPUnorm.getValue<PREC>(ir2,norm);
        if(norm > 0) {
            _CPUcorr.getValue<PREC>(ir2,corrResult);
            if(!isApproximatelyEqual(corrResult,1.0,1e-15)) {
                rootLogger.error(testName + " ir2, corr = " ,  ir2 ,  ", " ,  corrResult);
                lerror=true;
            }
        }
    }
}

/// TEMPLATES: <the type of the correlator>
/// INTENT:  IN--CPUcorr, CPUnorm, testName, r2max, referenceValue; OUT--lerror
void compareWithId3(Correlator<false,GCOMPLEX(PREC)> &CPUcorr, Correlator<false,PREC> &CPUnorm, std::string testName,
                    int r2max, bool &lerror)
{
    GCOMPLEX(PREC) corrResult;
    PREC norm;
    LatticeContainerAccessor _CPUcorr(CPUcorr.getAccessor());
    LatticeContainerAccessor _CPUnorm(CPUnorm.getAccessor());
    for(int ir2=0; ir2<r2max+1; ir2++) {
        _CPUnorm.getValue<PREC>(ir2,norm);
        if(norm > 0) {
            _CPUcorr.getValue<GCOMPLEX(PREC)>(ir2,corrResult);
            if(!isApproximatelyEqual(real(corrResult),3.0,1e-15)) {
                rootLogger.error(testName + " ir2, corr = " ,  ir2 ,  ", " ,  corrResult);
                lerror=true;
            }
        }
    }
}


int main(int argc, char *argv[]) { /// ---------------------------------------------------------------------- BEGIN MAIN

    stdLogger.setVerbosity(INFO);
    const size_t HaloDepth  = 0;

    /// Read in parameters and initialize communication base.
    rootLogger.info("Initialization");
    LatticeParameters param;
    CommunicationBase commBase(&argc, &argv);
    param.readfile(commBase, "../parameter/tests/correlatorTest.param", argc, argv);
    commBase.init(param.nodeDim());
    initIndexer(HaloDepth,param,commBase);
    typedef GIndexer<All,HaloDepth> GInd;

    /// More initialization.
    StopWatch<true> timer;
    Gaugefield<PREC,false,HaloDepth>     gauge(commBase);
    Gaugefield<PREC,true,HaloDepth>      gaugeDev(commBase);
    CorrelatorTools<PREC,true,HaloDepth> corrTools;

    /// Read the configuration. Remember a halo exchange is needed every time the gauge field changes.
    rootLogger.info("Read configuration");
    if (HAVECONFIG) gauge.readconf_nersc(param.GaugefileName());
    gauge.updateAll();
    gaugeDev=gauge;

    /// Error flag
    bool lerror = false;

    /// ---------------------------------------------------------------------------------------------- TEST INITIALIZERS

    int corrInt;
    PREC corrScalar;
    GCOMPLEX(PREC) corrComplex, corrComplexControl;
    GSU3<PREC> corrSU3, field1, field2;

    initCorrToZero<PREC>(corrInt);
    initCorrToZero<PREC>(corrScalar);
    initCorrToZero<PREC>(corrComplex);
    initCorrToZero<PREC>(corrSU3);

    if(corrInt!=0) lerror=true;
    if(corrScalar!=0.) lerror=true;
    if(!compareGCOMPLEX(corrComplex, (GCOMPLEX(PREC))0., 1e-13)) lerror=true;
    if(!compareGSU3(corrSU3, gsu3_zero<PREC>(), 1e-13)) lerror=true;

    initCorrToOne<PREC>(corrInt);
    initCorrToOne<PREC>(corrScalar);
    initCorrToOne<PREC>(corrComplex);
    initCorrToOne<PREC>(corrSU3);

    if(corrInt!=1) lerror=true;
    if(corrScalar!=1.) lerror=true;
    if(!compareGCOMPLEX(corrComplex, (GCOMPLEX(PREC))1., 1e-13)) lerror=true;
    if(!compareGSU3(corrSU3, gsu3_one<PREC>(), 1e-13)) lerror=true;

    CorrField<false,PREC>  CPUfield1(commBase, corrTools.vol4);

    CPUfield1.one();

    LatticeContainerAccessor _CPUfield1(CPUfield1.getAccessor());
    for(int m=0; m<corrTools.vol4; m++) {
        _CPUfield1.getValue(m,corrScalar);
        if(corrScalar!=1.) lerror=true;
    }

    /// -------------------------------------------------------------------- TEST CORRELATOR AND FIELD OBJECT OPERATIONS

    PREC norm, ratnorm;
    int dx, dy, dz, dt, r2;

    CorrField<false,PREC>  CPUfield2(commBase, corrTools.vol4);
    Correlator<false,PREC> CPUcorr(commBase, corrTools.UAr2max);
    Correlator<false,PREC> CPUnorm(commBase, corrTools.UAr2max);

    rootLogger.info("vol4, UAr2max = " ,  corrTools.vol4 ,  ",  " ,  corrTools.UAr2max);

    CorrField<false,PREC>  CPUSfield1(commBase, corrTools.vol3);
    CorrField<false,PREC>  CPUSfield2(commBase, corrTools.vol3);
    Correlator<false,PREC> CPUScorr(commBase, corrTools.USr2max);
    Correlator<false,PREC> CPUSnorm(commBase, corrTools.USr2max);

    rootLogger.info("vol3, USr2max = " ,  corrTools.vol3 ,  ",  " ,  corrTools.USr2max);

    CPUfield2.one();
    CPUSfield1.one();
    CPUSfield2.one();

    LatticeContainerAccessor _CPUnorm(CPUnorm.getAccessor());
    LatticeContainerAccessor _CPUSnorm(CPUSnorm.getAccessor());

    /// --------------------------------------------------------------------------------------------- 1.0 x 1.0 TEST

    rootLogger.info("Running symmetric UA 1 x 1 test.");
    corrTools.correlateAt<PREC,PREC,AxB<PREC>>("spacetime", CPUfield1, CPUfield2, CPUnorm, CPUcorr, true);
    compareWithOne(CPUcorr, CPUnorm, "symmetric UA 1 x 1:", corrTools.UAr2max, lerror);

    rootLogger.info("Running asymmetric UA 1 x 1 test.");
    corrTools.correlateAt<PREC,PREC,AxB<PREC>>("spacetime", CPUfield1, CPUfield2, CPUnorm, CPUcorr, false);
    compareWithOne(CPUcorr, CPUnorm, "asymmetric UA 1 x 1:", corrTools.UAr2max, lerror);

    rootLogger.info("Running symmetric US 1 x 1 test.");
    corrTools.correlateAt<PREC,PREC,AxB<PREC>>("spatial", CPUSfield1, CPUSfield2, CPUSnorm, CPUScorr, true);
    compareWithOne(CPUScorr, CPUSnorm, "symmetric US 1 x 1:", corrTools.USr2max, lerror);

    rootLogger.info("Running asymmetric US 1 x 1 test.");
    corrTools.correlateAt<PREC,PREC,AxB<PREC>>("spatial", CPUSfield1, CPUSfield2, CPUSnorm, CPUScorr, false);
    compareWithOne(CPUScorr, CPUSnorm, "asymmetric US 1 x 1:", corrTools.USr2max, lerror);

    /// ------------------------------------------------------------------------------------------- id_3 x id_3 TEST

    GSU3<PREC> id_3 = gsu3_one<PREC>();

    CorrField<false,GSU3<PREC>> CPUfield3(commBase, corrTools.vol4);
    CorrField<false,GSU3<PREC>> CPUfield4(commBase, corrTools.vol4);
    Correlator<false,GCOMPLEX(PREC)> CPUcorrComplex1(commBase, corrTools.UAr2max);
    LatticeContainerAccessor _CPUcorrComplex1(CPUcorrComplex1.getAccessor());
    CPUfield3.one();
    CPUfield4.one();

    CorrField<false,GSU3<PREC>> CPUSfield3(commBase, corrTools.vol3);
    CorrField<false,GSU3<PREC>> CPUSfield4(commBase, corrTools.vol3);
    Correlator<false,GCOMPLEX(PREC)> CPUScorrComplex(commBase, corrTools.USr2max);
    LatticeContainerAccessor _CPUScorrComplex(CPUScorrComplex.getAccessor());
    CPUSfield3.one();
    CPUSfield4.one();

    rootLogger.info("Running symmetric UA id_3 x id_3 test.");
    timer.start();
    corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),AxB<PREC>>("spacetime", CPUfield3, CPUfield4, CPUnorm, CPUcorrComplex1, true);
    timer.stop();
    compareWithId3(CPUcorrComplex1, CPUnorm, "symmetric UA id_3 x id_3:", corrTools.UAr2max, lerror);
    rootLogger.info("Time for symmetric UA id_3 x id_3: " ,  timer);
    timer.reset();

    rootLogger.info("Running asymmetric UA id_3 x id_3 test.");
    timer.start();
    corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),AxB<PREC>>("spacetime", CPUfield3, CPUfield4, CPUnorm, CPUcorrComplex1, false);
    timer.stop();
    compareWithId3(CPUcorrComplex1, CPUnorm, "asymmetric UA id_3 x id_3:", corrTools.UAr2max, lerror);
    rootLogger.info("Time for asymmetric UA id_3 x id_3: " ,  timer);
    timer.reset();

    rootLogger.info("Running symmetric US id_3 x id_3 test.");
    timer.start();
    corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),AxB<PREC>>("spatial", CPUSfield3, CPUSfield4, CPUSnorm, CPUScorrComplex, true);
    timer.stop();
    compareWithId3(CPUScorrComplex, CPUSnorm, "symmetric US id_3 x id_3:", corrTools.USr2max, lerror);
    rootLogger.info("Time for symmetric US id_3 x id_3: " ,  timer);
    timer.reset();

    rootLogger.info("Running asymmetric US id_3 x id_3 test.");
    timer.start();
    corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),AxB<PREC>>("spatial", CPUSfield3, CPUSfield4, CPUSnorm, CPUScorrComplex, false);
    timer.stop();
    compareWithId3(CPUScorrComplex, CPUSnorm, "asymmetric US id_3 x id_3:", corrTools.USr2max, lerror);
    rootLogger.info("Time for asymmetric US id_3 x id_3: " ,  timer);
    timer.reset();

    /// ------------------------------------------------------------------------------- gaugefield x gaugefield TEST

    if(HAVECONFIG) {

        Correlator<false,GCOMPLEX(PREC)> CPUcorrComplex2(commBase, corrTools.UAr2max);
        LatticeContainerAccessor _CPUcorrComplex2(CPUcorrComplex2.getAccessor());

        rootLogger.info("Running generalized UA gaugefield(mu=2) x gaugefield(mu=3) test.");

        size_t m, n;
        Correlator<false,PREC> CPUnormControl(commBase, corrTools.UAr2max);
        CPUnormControl.zero();
        LatticeContainerAccessor _CPUnormControl(CPUnormControl.getAccessor());
        LatticeContainerAccessor _CPUfield3(CPUfield3.getAccessor());
        LatticeContainerAccessor _CPUfield4(CPUfield4.getAccessor());
        gaugeAccessor<PREC> _gauge(gauge.getAccessor());

        /// Load gauge fields
        for(int m=0; m<corrTools.vol4; m++) {
            _CPUfield3.setValue(m, _gauge.getLink(GInd::getSiteMu(m,2)));
            _CPUfield4.setValue(m, _gauge.getLink(GInd::getSiteMu(m,3)));
        }

        Correlator<false,GCOMPLEX(PREC)> CPUcorrComplexControl1(commBase, corrTools.UAr2max);
        CPUcorrComplexControl1.zero();
        LatticeContainerAccessor _CPUcorrComplexControl1(CPUcorrComplexControl1.getAccessor());
        Correlator<false,GCOMPLEX(PREC)> CPUcorrComplexControl2(commBase, corrTools.UAr2max);
        CPUcorrComplexControl2.zero();
        LatticeContainerAccessor _CPUcorrComplexControl2(CPUcorrComplexControl2.getAccessor());

        /// Calculate the correlations on the CPU, independently of the kernel. Unlike in the kernel, the strategy
        /// of this calculation is to loop over all spacetime pairs, calculate the displacement vector (dx,dy,dz,dt)
        /// from that, and use them to determine correlations and distances.
        for (int mx = 0; mx < corrTools.Nx; mx++)
        for (int my = 0; my < corrTools.Ny; my++)
        for (int mz = 0; mz < corrTools.Nz; mz++)
        for (int mt = 0; mt < corrTools.Nt; mt++) {

            m=GInd::getSite(mx,my,mz,mt).isite;
            _CPUfield3.getValue<GSU3<PREC>>(m,field1);

            for (int nx = 0; nx < corrTools.Nx; nx++)
            for (int ny = 0; ny < corrTools.Ny; ny++)
            for (int nz = 0; nz < corrTools.Nz; nz++)
            for (int nt = 0; nt < corrTools.Nt; nt++) {

                n=GInd::getSite(nx,ny,nz,nt).isite;
                gSite site = GInd::getSite(mx,my,mz,mt);
                _CPUfield4.getValue<GSU3<PREC>>(n,field2);

                dx = nx-mx;
                dy = ny-my;
                dz = nz-mz;
                dt = nt-mt;
                if ( abs(dx) > corrTools.Nx/2 ) {
                    if ( dx > 0) {
                        dx = -corrTools.Nx + dx;
                    } else if ( dx < 0) {
                        dx =  corrTools.Nx + dx;
                    }
                }
                if ( abs(dy) > corrTools.Ny/2 ) {
                    if ( dy > 0) {
                        dy = -corrTools.Ny + dy;
                    } else if ( dy < 0) {
                        dy =  corrTools.Ny + dy;
                    }
                }
                if ( abs(dz) > corrTools.Nz/2 ) {
                    if ( dz > 0) {
                        dz = -corrTools.Nz + dz;
                    } else if ( dz < 0) {
                        dz =  corrTools.Nz + dz;
                    }
                }
                if ( abs(dt) > corrTools.Nt/2 ) {
                    if ( dt > 0) {
                        dt = -corrTools.Nt + dt;
                    } else if ( dt < 0) {
                        dt =  corrTools.Nt + dt;
                    }
                }

                r2 = dx*dx + dy*dy + dz*dz + dt*dt;
                _CPUnormControl.getValue(r2,ratnorm);
                ratnorm+=1.0;
                _CPUnormControl.setValue(r2,ratnorm);

                /// tr A x B
                _CPUcorrComplexControl1.getValue<GCOMPLEX(PREC)>(r2,corrComplex);
                corrComplex += tr_c(field1*field2);
                _CPUcorrComplexControl1.setValue<GCOMPLEX(PREC)>(r2,corrComplex);

                /// tr A x Bt
                _CPUcorrComplexControl2.getValue<GCOMPLEX(PREC)>(r2,corrComplex);
                corrComplex += tr_c(field1*dagger(field2));
                _CPUcorrComplexControl2.setValue<GCOMPLEX(PREC)>(r2,corrComplex);

            }
        }

        /// Normalize CPU results
        for (int ir2=0; ir2<corrTools.UAr2max+1; ir2++) {
            _CPUnormControl.getValue<PREC>(ir2,ratnorm);
            if(ratnorm > 0.) {

                /// tr A x B
                _CPUcorrComplexControl1.getValue<GCOMPLEX(PREC)>(ir2,corrComplex);
                corrComplex /= ratnorm;
                _CPUcorrComplexControl1.setValue<GCOMPLEX(PREC)>(ir2,corrComplex);

                /// tr A x Bt
                _CPUcorrComplexControl2.getValue<GCOMPLEX(PREC)>(ir2,corrComplex);
                corrComplex /= ratnorm;
                _CPUcorrComplexControl2.setValue<GCOMPLEX(PREC)>(ir2,corrComplex);

            }
        }

        /// GPU tr A x B
        timer.start();
        corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),AxB<PREC>>("spacetime", CPUfield3, CPUfield4, CPUnorm, CPUcorrComplex1);
        timer.stop();
        timer.reset();

        /// GPU tr A x Bt
        timer.start();
        corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),trAxBt<PREC>>("spacetime", CPUfield3, CPUfield4, CPUnorm, CPUcorrComplex2);
        timer.stop();

        for(int ir2=0; ir2<corrTools.UAr2max+1; ir2++) {
            _CPUnorm.getValue<PREC>(ir2,norm);
            _CPUnormControl.getValue<PREC>(ir2,ratnorm);
            ratnorm/=corrTools.vol4;
            if(ratnorm > 0.) {

                /// Check tr A x B
                _CPUcorrComplex1.getValue<GCOMPLEX(PREC)>(ir2,corrComplex);
                _CPUcorrComplexControl1.getValue<GCOMPLEX(PREC)>(ir2,corrComplexControl);
                if(!(corrComplex==corrComplexControl)) {
                    rootLogger.error("UA tr A x B: ir2, tr(test), tr(control) = " ,  ir2 ,  ", "
                                       ,  corrComplex ,  ", " ,  corrComplexControl);
                    lerror=true;
                }

                /// Check tr A x Bt
                _CPUcorrComplex2.getValue<GCOMPLEX(PREC)>(ir2,corrComplex);
                _CPUcorrComplexControl2.getValue<GCOMPLEX(PREC)>(ir2,corrComplexControl);
                if(!(corrComplex==corrComplexControl)) {
                    rootLogger.error("UA tr A x Bt: ir2, tr(test), tr(control) = " ,  ir2 ,  ", "
                                       ,  corrComplex ,  ", " ,  corrComplexControl);
                    lerror=true;
                }

            }
        }

        rootLogger.info("Time for UA gaugefield(mu=2) x gaugefield(mu=3): " ,  timer);
        timer.reset();

        rootLogger.info("Running US gaugefield(mu=2) x gaugefield(mu=3) test.");

        Correlator<false,PREC> CPUSnormControl(commBase, corrTools.USr2max);
        LatticeContainerAccessor _CPUSfield3(CPUSfield3.getAccessor());
        LatticeContainerAccessor _CPUSfield4(CPUSfield4.getAccessor());

        /// Load gauge fields
        for(int m=0; m<corrTools.vol3; m++) { /// If it doesn't work, check this guy maybe he's the reason
            _CPUSfield3.setValue(m, _gauge.getLink(GInd::getSiteSpatialMu(m,2)));
            _CPUSfield4.setValue(m, _gauge.getLink(GInd::getSiteSpatialMu(m,3)));
        }

        /// Calculate the correlations on the CPU, independently of the kernel. Unlike in the kernel, the strategy
        /// of this calculation is to loop over all spacetime pairs, calculate the displacement vector (dx,dy,dz,dt)
        /// from that, and use them to determine correlations and distances.
        Correlator<false,GCOMPLEX(PREC)> CPUScorrComplexControl(commBase, corrTools.USr2max);
        CPUScorrComplexControl.zero();
        CPUSnormControl.zero();
        LatticeContainerAccessor _CPUScorrComplexControl(CPUScorrComplexControl.getAccessor());
        LatticeContainerAccessor _CPUSnormControl(CPUSnormControl.getAccessor());
        for (int mx = 0; mx < corrTools.Nx; mx++)
        for (int my = 0; my < corrTools.Ny; my++)
        for (int mz = 0; mz < corrTools.Nz; mz++) {
                m=GInd::getSiteSpatial(mx,my,mz,0).isite;
                _CPUSfield3.getValue<GSU3<PREC>>(m,field1);
                for (int nx = 0; nx < corrTools.Nx; nx++)
                for (int ny = 0; ny < corrTools.Ny; ny++)
                for (int nz = 0; nz < corrTools.Nz; nz++) {
                        n=GInd::getSiteSpatial(nx,ny,nz,0).isite;
                        _CPUSfield4.getValue<GSU3<PREC>>(n,field2);
                        dx = nx-mx;
                        dy = ny-my;
                        dz = nz-mz;
                        if ( abs(dx) > corrTools.Nx/2 ) {
                            if ( dx > 0) {
                                dx = -corrTools.Nx + dx;
                            } else if ( dx < 0) {
                                dx =  corrTools.Nx + dx;
                            }
                        }
                        if ( abs(dy) > corrTools.Ny/2 ) {
                            if ( dy > 0) {
                                dy = -corrTools.Ny + dy;
                            } else if ( dy < 0) {
                                dy =  corrTools.Ny + dy;
                            }
                        }
                        if ( abs(dz) > corrTools.Nz/2 ) {
                            if ( dz > 0) {
                                dz = -corrTools.Nz + dz;
                            } else if ( dz < 0) {
                                dz =  corrTools.Nz + dz;
                            }
                        }
                        r2 = dx*dx + dy*dy + dz*dz;
                        _CPUSnormControl.getValue(r2,ratnorm);
                        if (m == n) {
                            ratnorm+=1.0;
                        } else {
                            ratnorm+=0.5;
                        }
                        _CPUSnormControl.setValue(r2,ratnorm);
                        _CPUScorrComplexControl.getValue<GCOMPLEX(PREC)>(r2,corrComplex);
                        if (m == n) {
                            corrComplex += tr_c(field1*field2);
                        } else {
                            corrComplex += tr_c(field1*field2)/2.;
                        }
                        _CPUScorrComplexControl.setValue<GCOMPLEX(PREC)>(r2,corrComplex);
                }
        }

        for (int ir2=0; ir2<corrTools.USr2max+1; ir2++) {
            _CPUSnormControl.getValue<PREC>(ir2,ratnorm);
            if(ratnorm > 0.) {
                _CPUScorrComplexControl.getValue<GCOMPLEX(PREC)>(ir2,corrComplex);
                corrComplex /= ratnorm;
                _CPUScorrComplexControl.setValue<GCOMPLEX(PREC)>(ir2,corrComplex);
            }
        }

        timer.start();
        corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),AxB<PREC>>("spatial", CPUSfield3, CPUSfield4, CPUSnorm, CPUScorrComplex);
        timer.stop();

        for(int ir2=0; ir2<corrTools.USr2max+1; ir2++) {
            _CPUSnorm.getValue<PREC>(ir2,norm);
            _CPUSnormControl.getValue<PREC>(ir2,ratnorm);
            ratnorm/=corrTools.vol3;
            if(ratnorm > 0.) {
                _CPUScorrComplex.getValue<GCOMPLEX(PREC)>(ir2,corrComplex);
                _CPUScorrComplexControl.getValue<GCOMPLEX(PREC)>(ir2,corrComplexControl);
                if(!(corrComplex==corrComplexControl)) {
                    rootLogger.error("US gauge x gauge: ir2, tr(test), tr(control) = " ,  ir2 ,  ", "
                                       ,  corrComplex ,  ", " ,  corrComplexControl);
                    lerror=true;
                }
            }
        }
        rootLogger.info("Time for US gaugefield(mu=2) x gaugefield(mu=3): " ,  timer);
    }

    /// ---------------------------------------- NOW I WANT TO COMPARE THE OUTPUT WITH THE RESTRICTED SPATIAL CORRELATOR

    PolyakovLoopCorrelator<PREC, true, HaloDepth> PLC(gaugeDev);
    PolyakovLoop<PREC,false,HaloDepth>            ploopClass(gauge); /// for measuring Polyakov loops

    std::vector<PREC> vec_plca(corrTools.distmax);
    std::vector<PREC> vec_plc1(corrTools.distmax);
    std::vector<PREC> vec_plc8(corrTools.distmax);
    std::vector<int>  vec_factor(corrTools.distmax);
    std::vector<int>  vec_weight(corrTools.pvol3);
    corrTools.getFactorArray(vec_factor,vec_weight);

    PLC.PLCtoArrays(vec_plca, vec_plc1, vec_plc8, vec_factor, vec_weight, true);

    CorrField<false,GSU3<PREC>>       thermalWilsonLine(commBase, corrTools.vol3);
    Correlator<false,GCOMPLEX(PREC)>  PLoopBareSusc(commBase, corrTools.USr2max);

    LatticeContainerAccessor _thermalWilsonLine(thermalWilsonLine.getAccessor());
    LatticeContainerAccessor _PLoopBareSusc(PLoopBareSusc.getAccessor());

    ploopClass.PloopInArray(_thermalWilsonLine);

    timer.start();
    corrTools.correlateAt<GSU3<PREC>,GCOMPLEX(PREC),trAxtrBt<PREC>>("spatial", thermalWilsonLine, thermalWilsonLine, CPUSnorm, PLoopBareSusc);
    timer.stop();

    GCOMPLEX(PREC) bareSusc;

    rootLogger.info("r2    plca    trAtrBt:");
    for (int r2=0 ; r2<corrTools.distmax ; r2++) {
        if (vec_factor[r2]>0) {
            _PLoopBareSusc.getValue<GCOMPLEX(PREC)>(r2,bareSusc);
            rootLogger.info(vec_plca[r2] ,  "  " ,  real(bareSusc)/9.);
        }
    }

    /// ------------------------------------------------------------------------------------------------- REPORT RESULTS
    if(lerror) {
        rootLogger.error("At least one test failed!");
        return -1;
    } else {
        rootLogger.info("All tests " ,  CoutColors::green ,  "passed!" ,  CoutColors::reset);
    }

    return 0;
}

