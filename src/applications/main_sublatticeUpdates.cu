/* 
 * main_sublatticeUpdates.cu                                                               
 * 
 * Hai-Tao Shu, 6 May 2019
 * 
 */

/***********************************************

the rountine performs local updates for the sub lattices and the contraction 
to get color-electric correlators ( one square lies in sub lattice 1 and the other in sub lattice 2 )
and energy-momentum tensor correlators in bulk & shear channel at finite momentum

     |_|_|_I_|_|_|_I_|_|_|_|_|_|_I_|_|_|_I_|_|_
s    |_|_|_I_|_|_|_I_|_|_|_|_|_|_I_|_|_|_I_|_|_
^    |_|_|_I_|_|_|_I_|_|_|_|_|_|_I_|_|_|_I_|_|_
|    |_|_|_I_|_|_|_I_|_|_|_|_|_|_I_|_|_|_I_|_|_
|    |_|_|_I_|_|_|_I_|_|_|_|_|_|_I_|_|_|_I_|_|_
|    |_|_|_I_|_|_|_I_|_|_|_|_|_|_I_|_|_|_I_|_|_
|    |_|_|_I_|_|_|_I_|_|_|_|_|_|_I_|_|_|_I_|_|_ 
|          |--\/---|             |--\/---|
|        sub lattice1           sub lattice 2
|
---------------------------> t 

The spatial links on the border won't be updated.

***********************************************/


#include "../SIMULATeQCD.h"
#include "../modules/gauge_updates/luscherweisz.h"
#include "../modules/gauge_updates/SubLatMeas.h"

#define PREC double
#define USE_GPU true


template<class floatT>
struct SubLatParam : LatticeParameters {

    //the temporal extent of the sublattice 
    Parameter<int> sublattice_lt;

    //measure how many times
    Parameter<int> num_meas;

    //how many sweeps(=4*OR+1*HB) performed after each measuring
    Parameter<int> num_update;

    //the smallest distance of two squares
    Parameter<int> min_dist;

    //the finite mometum for EMT, directing only in z
    Parameter<int> PzMax;

    //a magic value used to subtract the first a few useless digits appearing in the traceanomaly at zero momentum 
    //does not contribute the bulk correlators. It helps to avoid the numerical problem and can be obtained by running
    //another programm "getMagicTraceAnomaly" on any of the configurations of the same lattice size and beta
    Parameter<PREC> VacuumSubtractionHelper;

    // path of output files 
    Parameter<std::string> out_dir;

    Parameter<bool> EnergyMomentumTensorCorr;
    Parameter<bool> ColorElectricCorr;

    // constructor
    SubLatParam() {
 
        addDefault(sublattice_lt, "sublattice_lt", 6);
        addDefault(num_meas, "num_meas", 100);
        addDefault(num_update, "num_update", 10);
        addDefault(min_dist, "min_dist", 0);
        addDefault(PzMax, "PzMax", 0);
        add(VacuumSubtractionHelper, "VacuumSubtractionHelper");
        add(out_dir, "out_dir");
        addDefault(EnergyMomentumTensorCorr, "EnergyMomentumTensorCorr", false);
        addDefault(ColorElectricCorr, "ColorElectricCorr", false);
    }
};


int main(int argc, char *argv[]) {

    stdLogger.setVerbosity(INFO);

    // initialize the communication base
    CommunicationBase commBase(&argc, &argv);

    // initialize the Sublattice Updates method parameter class
    SubLatParam<PREC> lp;

    // read the parameter file On Root(specified by commBase_tmp) and send it to All
    lp.readfile(commBase, "../parameter/applications/sublatticeUpdates.param", argc, argv);

    // write node dimensions to the communication base
    commBase.init(lp.nodeDim());

    if ( lp.EnergyMomentumTensorCorr() && lp.sublattice_lt() < 4 ) {
        throw std::runtime_error(stdLogger.fatal("Temporal extent of sub lattices can NOT be smaller than 4 when measure energy-"
                       "momentum tensor correlators!!!"));
    }

    if ( lp.ColorElectricCorr() && lp.sublattice_lt() < 3 ) {
        throw std::runtime_error(stdLogger.fatal("Temporal extent of sub lattices can NOT be smaller than 3 when measure "
                       "color-electric correlators!!!"));
    }

    if ( lp.nodeDim()[3] != 1) {
        throw std::runtime_error(stdLogger.fatal("Can NOT split node in temporal direction !!"));
    }

    if ( lp.latDim()[3]%lp.sublattice_lt() != 0) {
        throw std::runtime_error(stdLogger.fatal("Temporal extent of sub lattices must be divisible full temporal extention !!!"));
    }


    const size_t HaloDepth = 1;

    size_t AvailDevice;
    size_t TotalDevice;
    size_t UsedDevice;

    initIndexer(HaloDepth, lp, commBase);

    //this part is for memory check. exit if not enough memory. needed because this algorithm needs a lot of memory
    size_t global_spatial_vol = lp.latDim()[0]*lp.latDim()[1]*lp.latDim()[2]; //global spatial vol
    size_t node_multiplication = lp.nodeDim()[0]*lp.nodeDim()[1]*lp.nodeDim()[2]; //the multiplication of spatial node dim
    size_t elem1 = global_spatial_vol/node_multiplication;//local spatial vol on each gpu(the size of sub ced1)
    size_t elem2 = elem1-(lp.latDim()[0]/lp.nodeDim()[0]-2)*(lp.latDim()[1]/lp.nodeDim()[1]-2)*(lp.latDim()[2]/lp.nodeDim()[2]-2);//the size of sub cec2
    size_t elem3 = 1;
    for (size_t mu=0; mu<4; mu++) {
        if (lp.nodeDim()[mu] > 1) {
            elem3 *= (lp.latDim()[mu]/lp.nodeDim()[mu]+2*HaloDepth);
        } else {
            elem3 *= lp.latDim()[mu];
        }
    }
    size_t elem4 = elem1*lp.latDim()[3];
    size_t P2P = elem3 + 2*(elem3-elem4);

    gpuMemGetInfo( &AvailDevice, &TotalDevice );
    UsedDevice = TotalDevice - AvailDevice;
    double StarterUsedInMB = UsedDevice/1024./1024.;
    rootLogger.info("Memory for the starter[MB]: " ,  StarterUsedInMB);
    double Ratio = (double)(elem2)/elem1; 
    double MemLatNoHalo = sizeof(PREC)*18.*elem1*lp.latDim()[3]*4/1024./1024.;
    double MemLatWithHalo = sizeof(PREC)*18.*P2P*4/1024./1024.;

    double MemAll = StarterUsedInMB + MemLatWithHalo + 16.*elem4/1024./1024.;//last term for seed
    if (lp.ColorElectricCorr()) {
        double TotalEstimation_CEC = MemLatNoHalo*((1+6.*(lp.sublattice_lt()-2)*(1+Ratio))/4.);
        rootLogger.info("Memory to be used for CEC[MB]: " ,  TotalEstimation_CEC);
        MemAll += TotalEstimation_CEC;
    }

    rootLogger.info("Device Memory[MB]: " ,  TotalDevice/1024./1024.);
    if ( MemAll + 20 > TotalDevice/1024./1024. ) { //20 is an estimation for miscellaneous memory comsumption
        throw std::runtime_error(stdLogger.fatal("Device Memory[MB]:  ", TotalDevice / 1024. / 1024., "    Required Memory[MB]:  ", MemAll + 20,
                       ". Memory needed exceeds the device's memory ! exit the program."));
    }

    //field on the host, can not be changed.
    Gaugefield<PREC, false, HaloDepth> gauge_host(commBase);

    rootLogger.info("Read configuration");
    gauge_host.readconf_nersc(lp.GaugefileName());
    gauge_host.updateAll();

    grnd_state<false> host_state;
    grnd_state<true> dev_state; 

    StopWatch<true> timer;
    timer.start();

    std::stringstream datNameCE_ImprovePoly, datNameCE_ImproveColEleCorr;
    std::stringstream datNameCE_NonimprovePoly, datNameCE_NonimproveCorr;
    std::stringstream datNameEMT_ImproveBulk, datNameEMT_NonimproveBulk;
    std::stringstream datNameEMT_ImproveShear, datNameEMT_NonimproveShear;
    std::stringstream datNameEMT_ImproveTraceAnomaly, datNameEMT_NonimproveTraceAnomaly;
    std::stringstream datName_normEMT;

    //gpu memmory to store the polyakov loop within the sublattice and half of the numerator of the color-electric correlator.
    typedef gMemoryPtr<true> MemTypeGPU;
    MemTypeGPU mem50 = MemoryManagement::getMemAt<true>("subpoly_Nt_gpu");
    MemTypeGPU mem51 = MemoryManagement::getMemAt<true>("sub1_cec_Nt_gpu");
    MemTypeGPU mem52 = MemoryManagement::getMemAt<true>("sub2_cec_Nt_gpu");
 
    if (lp.ColorElectricCorr()) {
        mem50->template adjustSize<GSU3<PREC>>(elem1*lp.latDim()[3]);
        mem50->memset(0);
        mem51->template adjustSize<GSU3<PREC>>(elem1*6*(lp.sublattice_lt()-2)*lp.latDim()[3]);
        mem51->memset(0);
        mem52->template adjustSize<GSU3<PREC>>(elem2*6*(lp.sublattice_lt()-2)*lp.latDim()[3]);
        mem52->memset(0);
    }
    MemoryAccessor sub_poly_Nt (mem50->getPointer());
    MemoryAccessor sub1_cec_Nt (mem51->getPointer());
    //same as sub1_cec_Nt but only the part on the boundary of each gpu
    MemoryAccessor sub2_cec_Nt (mem52->getPointer());

    std::vector<PREC> SubBulk_Nt_real((lp.PzMax()+1)*lp.latDim()[3]*(lp.sublattice_lt()-3), 0);
    std::vector<PREC> SubBulk_Nt_imag((lp.PzMax()+1)*lp.latDim()[3]*(lp.sublattice_lt()-3), 0);
    std::vector<Matrix4x4Sym<PREC>> SubShear_Nt_real((lp.PzMax()+1)*lp.latDim()[3]*(lp.sublattice_lt()-3), 0);
    std::vector<Matrix4x4Sym<PREC>> SubShear_Nt_imag((lp.PzMax()+1)*lp.latDim()[3]*(lp.sublattice_lt()-3), 0);


    std::vector<PREC> SubTbarbp00(lp.latDim()[3]*(lp.sublattice_lt()-3), 0);
    std::vector<PREC> SubSbp(lp.latDim()[3]*(lp.sublattice_lt()-3), 0);
    //save SubTbarbc00 and SubSbc as the real and imag part of a complex number.
    std::vector<GCOMPLEX(PREC)> SubTbarbc00_SubSbc(lp.latDim()[3]*(lp.sublattice_lt()-3), 0);

    //only for the zero momentum, where the summation over different lp.num_meas() needs to be specially taken care of.
    std::vector<PREC> SubBulk_Nt_p0(lp.num_meas()*lp.latDim()[3]*(lp.sublattice_lt()-3), 0);
    std::vector<Matrix4x4Sym<PREC>> SubShear_Nt_p0(lp.num_meas()*lp.latDim()[3]*(lp.sublattice_lt()-3), 0);

    MemTypeGPU mem53 = MemoryManagement::getMemAt<true>("sub_E_gpu");
    MemTypeGPU mem54 = MemoryManagement::getMemAt<true>("sub_U_gpu");
    if (lp.EnergyMomentumTensorCorr()) {
        mem53->template adjustSize<PREC>(elem1);
        mem53->memset(0);
        mem54->template adjustSize<Matrix4x4Sym<PREC>>(elem1);
        mem54->memset(0);
    }
    MemoryAccessor sub_E_gpu (mem53->getPointer());
    MemoryAccessor sub_U_gpu (mem54->getPointer());


    for (int local_pos_t=0;local_pos_t<lp.sublattice_lt();local_pos_t++) {
    
        rootLogger.info("measure. update the sub-lattice at shift position: " ,  local_pos_t ,  "/" ,  lp.sublattice_lt());
        // copy field to device 
        Gaugefield<PREC, USE_GPU, HaloDepth> gauge_device(gauge_host.getComm()); 
        gauge_device = gauge_host;
        gauge_device.updateAll();

        // initialize sub-lattice updates 
        LuscherWeisz<PREC, USE_GPU, HaloDepth> luscherweisz(gauge_device);

        // initialize for caculating sub poly and sub cec
        SubLatMeas<PREC, USE_GPU, HaloDepth> sublatmeas(gauge_device,lp.sublattice_lt()); 

        //generate random seed
        auto seed = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch()).count();
        host_state.make_rng_state(seed);
        dev_state = host_state;

        //need include the measure below
        for (int i = 0; i < lp.num_meas(); ++i) {
            for (int j = 0; j < lp.num_update(); ++j) {
                luscherweisz.subUpdateHB(dev_state.state, lp.beta(), lp.sublattice_lt(), local_pos_t, false);
                for (int k = 0; k < 4; k++) { 
                    luscherweisz.subUpdateOR(lp.sublattice_lt(), local_pos_t);
                }
            }

            for (int sub_i=0;sub_i<lp.latDim()[3]/lp.sublattice_lt();sub_i++) {
                //measure the energy-momentum tensor at each possible time slice within each sub lattice 
                if (lp.EnergyMomentumTensorCorr()) {
                    sublatmeas.updateSubNorm(local_pos_t+sub_i*lp.sublattice_lt(), SubTbarbp00, SubSbp, SubTbarbc00_SubSbc); 
                    for ( int dist=0; dist<lp.sublattice_lt()-3; dist++ ) { //minus 3 because T is from F, clover boundary can NOT hit sublattice boudndary
                        for ( int pz=0;pz<=lp.PzMax();pz++ ) {
                            sublatmeas.updateSubEMT(local_pos_t+sub_i*lp.sublattice_lt(), lp.num_meas(), sub_E_gpu, sub_U_gpu, SubBulk_Nt_p0, SubShear_Nt_p0, SubBulk_Nt_real, 
                                                    SubShear_Nt_real, dist, pz, i, lp.VacuumSubtractionHelper(), 0);
                            sublatmeas.updateSubEMT(local_pos_t+sub_i*lp.sublattice_lt(), lp.num_meas(), sub_E_gpu, sub_U_gpu, SubBulk_Nt_p0, SubShear_Nt_p0, SubBulk_Nt_imag, 
                                                    SubShear_Nt_imag, dist, pz, i, lp.VacuumSubtractionHelper(), 1);
                        }
                    }
                }
    
                //measure sub polyakov loop and sub color electric correlator from the locally updated confs 
                if (lp.ColorElectricCorr()) {
                    sublatmeas.updateSubPolyCorr(local_pos_t+sub_i*lp.sublattice_lt(), lp.num_meas(), sub_poly_Nt, sub1_cec_Nt, sub2_cec_Nt);
                }
            }
        }
    }

    if (lp.EnergyMomentumTensorCorr()) {

        datName_normEMT << lp.out_dir() << "MultiLevel" << "_NormEMT" << lp.fileExt();
        FileWriter file_normEMT(gauge_host.getComm(), lp);
        file_normEMT.createFile(datName_normEMT.str());
        LineFormatter newTag_normEMT = file_normEMT.header(15);
        newTag_normEMT << "#Tbarbp00 #Sbp #Tbarbc00 #Sbc" << "\n";

        PREC Tbarbp00 = 0.;
        PREC Sbp = 0.;
        PREC Tbarbc00 = 0.;
        PREC Sbc = 0.;

        for ( int tau=0; tau<lp.latDim()[3]; tau++ ) {
            for ( int dist=0; dist<lp.sublattice_lt()-3; dist++ ) {
                Tbarbp00 += SubTbarbp00[tau*(lp.sublattice_lt()-3)+dist];
                Sbp += SubSbp[tau*(lp.sublattice_lt()-3)+dist];
                Tbarbc00 += imag(SubTbarbc00_SubSbc[tau*(lp.sublattice_lt()-3)+dist]);
                Sbc += real(SubTbarbc00_SubSbc[tau*(lp.sublattice_lt()-3)+dist]);
            }
        }
        Tbarbp00 /= ((lp.sublattice_lt()-3.)*lp.latDim()[3]*lp.num_meas());
        Sbp /= ((lp.sublattice_lt()-3.)*lp.latDim()[3]*lp.num_meas());
        Tbarbc00 /= ((lp.sublattice_lt()-3.)*lp.latDim()[3]*lp.num_meas());
        Sbc /= ((lp.sublattice_lt()-3.)*lp.latDim()[3]*lp.num_meas());

        newTag_normEMT << std::scientific << std::setprecision(15) << Tbarbp00*2. << 2.*(18.-Sbp) << Tbarbc00 << Sbc <<"\n";


        Contraction_cpu<PREC> contraction(lp.sublattice_lt(), lp.latDim()[3]);
        //normalize the zero p traceanomaly by summing over and deviding lp.num_meas(). aim to reduing the numerical error 
        contraction.ImproveNormalizeBulk(SubBulk_Nt_real, SubBulk_Nt_p0, lp.num_meas()); 

        //contraction for the improve emt correlators in bulk channel  
        rootLogger.info("contration to obtain the multi-level improved energy-momentum correlator in bulk channel");
        std::vector<PREC> ImproveBulk_Correlator((lp.PzMax()+1)*lp.latDim()[3], 0) ;
        for(int pz=0; pz<=lp.PzMax();pz++) {
            contraction.ImproveContractionBulk(SubBulk_Nt_real, SubBulk_Nt_imag, lp.min_dist(), global_spatial_vol, pz, ImproveBulk_Correlator);  
        }

        //write improve emt correlators in bulk channel
        datNameEMT_ImproveBulk << lp.out_dir() << "MultiLevel_Bulk" << "_PzMax" << lp.PzMax() << lp.fileExt();
        FileWriter file_ImproveBulk(gauge_host.getComm(), lp);
        file_ImproveBulk.createFile(datNameEMT_ImproveBulk.str());
        LineFormatter newTagImproveBulk = file_ImproveBulk.header(15);
        newTagImproveBulk << "pz " << "tau " << "emte_corr " << "\n";
        for(int pz=0; pz<=lp.PzMax();pz++) {
            for ( int tau=0; tau<lp.latDim()[3]/2+1; tau++ ) {
                newTagImproveBulk << pz << tau << std::scientific << std::setprecision(15)
                                     << ImproveBulk_Correlator[pz*lp.latDim()[3]+tau] << "\n";
            }
        }

        PREC ImproveTraceAnomaly_real = 0.;
        for ( int tau=0; tau<lp.latDim()[3]; tau++ ) {
            for ( int dist=0; dist<lp.sublattice_lt()-3; dist++ ) {
                ImproveTraceAnomaly_real += SubBulk_Nt_real[tau*(lp.sublattice_lt()-3)+dist];
            }
        }
        ImproveTraceAnomaly_real /= ((lp.sublattice_lt()-3.)*lp.latDim()[3]);

        //write improve trace anomaly
        datNameEMT_ImproveTraceAnomaly << lp.out_dir() << "MultiLevel_TraceAnomaly" << "_PzMax" << lp.PzMax() << lp.fileExt();
        FileWriter file_ImproveTraceAnomaly(gauge_host.getComm(), lp);
        file_ImproveTraceAnomaly.createFile(datNameEMT_ImproveTraceAnomaly.str());
        LineFormatter newTagImproveTraceAnomaly = file_ImproveTraceAnomaly.header(15);
        newTagImproveTraceAnomaly << "only need zero momentum value, thus only real part" << "\n"; 
        newTagImproveTraceAnomaly << std::scientific << std::setprecision(15) << ImproveTraceAnomaly_real << "\n"; 

        /******************************** shear part below **************************************************/

        //normalize the shear channel element
        contraction.ImproveNormalizeShear(SubShear_Nt_real, SubShear_Nt_p0, lp.num_meas()); 
        //contraction for the improve emt correlators in bulk channel  
        rootLogger.info("contration to obtain the multi-level improved energy-momentum correlator in shear channel");
        std::vector<PREC> ImproveShear_Correlator((lp.PzMax()+1)*lp.latDim()[3], 0) ;
        for(int pz=0; pz<=lp.PzMax();pz++) {
            contraction.ImproveContractionShear(SubShear_Nt_real, SubShear_Nt_imag, lp.min_dist(), global_spatial_vol, pz, ImproveShear_Correlator);
        }

        //write improve emt correlators in bulk channel and the improve traceless parts of this conf  
        datNameEMT_ImproveShear << lp.out_dir() << "MultiLevel_Shear" << "_PzMax" << lp.PzMax() << lp.fileExt();
        FileWriter file_ImproveShear(gauge_host.getComm(), lp);
        file_ImproveShear.createFile(datNameEMT_ImproveShear.str());
        LineFormatter newTagImproveShear = file_ImproveShear.header(15);

        newTagImproveShear << "pz " << "tau " << "emtu_corr " << "\n";
        for(int pz=0; pz<=lp.PzMax();pz++) {
            for ( int tau=0; tau<lp.latDim()[3]/2+1; tau++ ) {
                newTagImproveShear << pz << tau << std::scientific << std::setprecision(15)
                                     << ImproveShear_Correlator[pz*lp.latDim()[3]+tau] << "\n";
            }
        }
    }

    if (lp.ColorElectricCorr()) {
        //copy the original lattice to device again for the final contraction
        Gaugefield<PREC, USE_GPU, HaloDepth> gauge_device(gauge_host.getComm());
        gauge_device = gauge_host;
        SubLatMeas<PREC, USE_GPU, HaloDepth> sublatmeas(gauge_device,lp.sublattice_lt());


        //calculate standard polyakovloop
        //PolyakovLoop<PREC, USE_GPU, HaloDepth> StandardPolyloop(gauge_device);
        //GCOMPLEX(PREC) NonimprovePolyakovLoop = StandardPolyloop.getPolyakovLoop();

        ////calculate standard  color electric correlator 
        //std::vector<GCOMPLEX(PREC)> NonimproveColorElectricCorrelator;
        //ColorElectricCorr<PREC, USE_GPU, HaloDepth> StandardCEC(gauge_device);
        //NonimproveColorElectricCorrelator = StandardCEC.getColorElectricCorr();

        ////write nonimprove polyakovloop
        //rootLogger.info("calculate the nonimprove polyakovloop ");
        //datNameCE_NonimprovePoly << lp.out_dir() << "Standard_polyakovloop" << lp.fileExt();
        //FileWriter file_NonimprovePoly(gauge_host.getComm(), lp);
        //file_NonimprovePoly.createFile(datNameCE_NonimprovePoly.str());
        //LineFormatter newTagNonimprovePoly = file_NonimprovePoly.header();
        //newTagNonimprovePoly << "Re(PolyLoop) " << "Im(PolyLoop) " << "\n";
        //newTagNonimprovePoly << std::scientific << std::setprecision(15) << real(NonimprovePolyakovLoop) << " " << imag(NonimprovePolyakovLoop) << "\n";

        ////write nonimprove color electric correlator  
        //rootLogger.info("calculate the nonimprove colorelectric correlator");
        //datNameCE_NonimproveCorr << lp.out_dir() << "Standard_colorelectriccorr" << lp.fileExt();
        //FileWriter file_NonimproveCorr(gauge_host.getComm(), lp);
        //file_NonimproveCorr.createFile(datNameCE_NonimproveCorr.str());
        //LineFormatter newTagNonimproveCorr = file_NonimproveCorr.header();
        //newTagNonimproveCorr << "tau " << "Re(ColorElectricCorrelator) " << "Im(ColorElectricCorrelator) " << "\n";
        //for (int i = 1; i < lp.latDim()[3]/2+1; i++ ) {
        //    newTagNonimproveCorr << i << " " << std::scientific << std::setprecision(15) << real(NonimproveColorElectricCorrelator[i-1]) << " " << imag(NonimproveColorElectricCorrelator[i-1]) << "\n";
        //}


        //contraction for the improve polyakovloop
        GCOMPLEX(PREC) ImprovePolyakovLoop ;
        ImprovePolyakovLoop = sublatmeas.contraction_poly(sub_poly_Nt, lp.min_dist());

        //contraction for the improve color electric correlator 
        std::vector<GCOMPLEX(PREC)> ImproveColorElectricCorrelator(lp.latDim()[3]/2+1, 0) ;
        ImproveColorElectricCorrelator = sublatmeas.contraction_cec(sub1_cec_Nt, sub2_cec_Nt, lp.min_dist());

        //write improve polyakovloop
        rootLogger.info("contract to obtain improve polyakovloop ");
        datNameCE_ImprovePoly << lp.out_dir() << "MultiLevel_polyakovloop" << lp.fileExt();
        FileWriter file_ImprovePoly(gauge_host.getComm(), lp);
        file_ImprovePoly.createFile(datNameCE_ImprovePoly.str());
        LineFormatter newTagImprovePoly = file_ImprovePoly.header();
        newTagImprovePoly << "Re(PolyLoop) " << "Im(PolyLoop) " << "\n";
        newTagImprovePoly << std::scientific << std::setprecision(15) << real(ImprovePolyakovLoop) << " " << imag(ImprovePolyakovLoop) << "\n";

        //write improve color electric correlator  
        rootLogger.info("contract to obtain improve colorelectric correlator");
        datNameCE_ImproveColEleCorr << lp.out_dir() << "MultiLevel_colorelectriccorr" << lp.fileExt();
        FileWriter file_ImproveColEleCorr(gauge_host.getComm(), lp);
        file_ImproveColEleCorr.createFile(datNameCE_ImproveColEleCorr.str());
        LineFormatter newTagImproveColEleCorr = file_ImproveColEleCorr.header();
        newTagImproveColEleCorr << "tau " << "Re(ColorElectricCorrelator) " << "Im(ColorElectricCorrelator) " << "\n";
        //for (int i = 1; i < lp.min_dist()+3; i++ ) { //the uncalculated values will be filled by the nonimproved values.
        //    newTagImproveColEleCorr << i << " " << std::scientific << std::setprecision(15) << real(NonimproveColorElectricCorrelator[i-1]) << " " << imag(NonimproveColorElectricCorrelator[i-1]) << "\n";
        //}
        for (int i = lp.min_dist()+3; i < lp.latDim()[3]/2+1; i++ ) {
            newTagImproveColEleCorr << i << " " << std::scientific << std::setprecision(15) << real(ImproveColorElectricCorrelator[i]) << " " << imag(ImproveColorElectricCorrelator[i]) << "\n";
        }
    }

    timer.stop();
    rootLogger.info("complete time = " ,  timer);

    return 0;
}

