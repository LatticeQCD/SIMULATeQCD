#include "SubLatMeas.h"
#include "../../gauge/gauge_kernels.cpp"


template<class floatT, bool onDevice, size_t HaloDepth>
struct SubTbarbp00Kernel {
    gaugeAccessor<floatT> gAcc;
    UtauMinusUsigmaKernel<floatT, onDevice, HaloDepth, R18> UtauMinusUsigma;
    int tau;
    SubTbarbp00Kernel(Gaugefield<floatT, onDevice, HaloDepth> &gauge, int tau) : gAcc(gauge.getAccessor()),    UtauMinusUsigma(gauge), tau(tau) {}
    __device__ __host__ inline floatT operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;

        sitexyzt coord = site.coord;
        gSite newSite = GInd::getSite(coord[0],coord[1],coord[2],tau);
        return UtauMinusUsigma(newSite);
    }
};

template<class floatT, bool onDevice, size_t HaloDepth>
struct SubSbpKernel {
    gaugeAccessor<floatT> gAcc;
    plaquetteKernel<floatT, onDevice, HaloDepth, R18> plaq;
    int tau;
    SubSbpKernel(Gaugefield<floatT, onDevice, HaloDepth> &gauge, int tau) : gAcc(gauge.getAccessor()), plaq(gauge), tau(tau) {}
    __device__ __host__ inline floatT operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;

        sitexyzt coord = site.coord;
        gSite newSite = GInd::getSite(coord[0],coord[1],coord[2],tau);
        return plaq(newSite);
    }
};

template<class floatT, bool onDevice, size_t HaloDepth>
struct SubTbarbc00SubSbcKernel {
    gaugeAccessor<floatT> gAcc;
    Ftau2PlusMinusFsigma2Elements<floatT, onDevice, HaloDepth, R18> Ftau2PlusMinusFsigma2;
    int tau;
    SubTbarbc00SubSbcKernel(Gaugefield<floatT, onDevice, HaloDepth> &gauge, int tau) : gAcc(gauge.getAccessor()), Ftau2PlusMinusFsigma2(gauge), tau(tau) {}
    __device__ __host__ inline GCOMPLEX(floatT) operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;

        sitexyzt coord = site.coord;
        gSite newSite = GInd::getSite(coord[0],coord[1],coord[2],tau);
        return Ftau2PlusMinusFsigma2(newSite);
    }
};



template<class floatT, bool onDevice, size_t HaloDepth>
void SubLatMeas<floatT, onDevice, HaloDepth>::updateSubNorm(int pos_t, std::vector<floatT> &SubTbarbp00, std::vector<floatT> &SubSbp, std::vector<GCOMPLEX(floatT)> &SubTbarbc00_SubSbc) {

    for (int dist=0;dist<_sub_lt-3;dist++) {
        _redBaseE.template iterateOverSpatialBulk<All, HaloDepth>(SubTbarbp00Kernel<floatT, onDevice, HaloDepth>(_gauge, (pos_t+dist+2)%_Nt));
        floatT result_tmp = 0.;
        _redBaseE.reduce(result_tmp, _elems1);
        SubTbarbp00[pos_t*(_sub_lt-3)+dist] += result_tmp/_spatialvol;

        _redBaseE.template iterateOverSpatialBulk<All, HaloDepth>(SubSbpKernel<floatT, onDevice, HaloDepth>(_gauge, (pos_t+dist+2)%_Nt));
        result_tmp = 0.;
        _redBaseE.reduce(result_tmp, _elems1);
        SubSbp[pos_t*(_sub_lt-3)+dist] += result_tmp/_spatialvol;

        _redBaseCE.template iterateOverSpatialBulk<All, HaloDepth>(SubTbarbc00SubSbcKernel<floatT, onDevice, HaloDepth>(_gauge, (pos_t+dist+2)%_Nt));
        GCOMPLEX(floatT) result_tmp1(0,0);
        _redBaseCE.reduce(result_tmp1, _elems1);
        SubTbarbc00_SubSbc[pos_t*(_sub_lt-3)+dist] += result_tmp1/_spatialvol;
    }

}


template<class floatT, bool onDevice, size_t HaloDepth>
void SubLatMeas<floatT, onDevice, HaloDepth>::updateSubEMT(int pos_t, int count, MemoryAccessor &sub_E_gpu, MemoryAccessor &sub_U_gpu, 
    std::vector<floatT> &SubBulk_Nt_p0, std::vector<Matrix4x4Sym<floatT>> &SubShear_Nt_p0, std::vector<floatT> &SubBulk_Nt, 
    std::vector<Matrix4x4Sym<floatT>> &SubShear_Nt, int dist, int pz, int count_i, floatT displacement, int flag_real_imag) {

    _redBaseE.template iterateOverSpatialBulk<All, HaloDepth>(energyMomentumTensorEKernel<floatT, HaloDepth, onDevice>(_gauge, sub_E_gpu, (pos_t+dist+2)%_Nt, pz, flag_real_imag, displacement));

    floatT resultE_tmp = 0.;
    _redBaseE.reduce(resultE_tmp, _elems1);

    if ( pz == 0 && !flag_real_imag ) {
        resultE_tmp /= _spatialvol;
        SubBulk_Nt_p0[pos_t*(_sub_lt-3)*count+dist*count+count_i] = resultE_tmp;
    } else {
        resultE_tmp /= (_spatialvol*count);
        SubBulk_Nt[pz*_Nt*(_sub_lt-3)+pos_t*(_sub_lt-3)+dist] += resultE_tmp;
    }

    _redBaseU.template iterateOverSpatialBulk<All, HaloDepth>(energyMomentumTensorUKernel<floatT, HaloDepth, onDevice>(_gauge, sub_E_gpu, sub_U_gpu, (pos_t+dist+2)%_Nt, pz, flag_real_imag));

    Matrix4x4Sym<floatT> resultU_tmp(0);
    _redBaseU.reduce(resultU_tmp, _elems1);

    if ( pz == 0 && !flag_real_imag ) {
        resultU_tmp /= _spatialvol;
        SubShear_Nt_p0[pos_t*(_sub_lt-3)*count+dist*count+count_i] = resultU_tmp;
    } else {
        resultU_tmp /= (_spatialvol*count);
        SubShear_Nt[pz*_Nt*(_sub_lt-3)+pos_t*(_sub_lt-3)+dist] += resultU_tmp;
    }
}


template<class floatT, bool onDevice, size_t HaloDepth>
struct updateSubPoly_Kernel{
    gaugeAccessor<floatT> gAcc;
    MemoryAccessor sub_poly_Nt;
    int sub_lt;
    int pos_t;
    int count;

    updateSubPoly_Kernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor sub_poly_Nt, int sub_lt, int pos_t, int count) :
            gAcc(gauge.getAccessor()), sub_poly_Nt(sub_poly_Nt), sub_lt(sub_lt), pos_t(pos_t), count(count){}

    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;
        int Nx = (int)GInd::getLatData().lx;
        int Ny = (int)GInd::getLatData().ly;
        int Nz = (int)GInd::getLatData().lz;
        const size_t elems1 = GInd::getLatData().vol3;

        gSite shifted_site = GInd::getSite(site.coord.x,site.coord.y,site.coord.z,pos_t);
        sitexyzt coord = shifted_site.coord;
        size_t Id = coord[0] + coord[1]*Nx + coord[2]*Ny*Nx;

        GSU3<floatT> temp1 = gAcc.getLink(GInd::getSiteMu(shifted_site, 3));
        for(int tau = 1; tau < sub_lt; tau++){
            shifted_site = GInd::site_up(shifted_site,3);
            temp1*= gAcc.getLink(GInd::getSiteMu(shifted_site, 3));
        }
        GSU3<floatT> temp2;
        sub_poly_Nt.getValue<GSU3<floatT>>(pos_t*elems1+Id, temp2);

        floatT factor = (1./count);
        sub_poly_Nt.setValue<GSU3<floatT>>(pos_t*elems1+Id, temp1*factor+temp2);
    }
};


 /*| vol3 spatial links:  x global_vol3
   | not one the border:  x (sub_lt-2)
   | mu=0,1,2 for the vertical links: x 3 
   | flipped diagram: x 2  
   | so a vector of global_vol3*(sub_lt-2)*6 entries */

template<class floatT, bool onDevice, size_t HaloDepth>
struct updateSubCorr_Kernel{
    gaugeAccessor<floatT> gAcc;
    MemoryAccessor sub1_cec_Nt;
    MemoryAccessor sub2_cec_Nt;
    int sub_lt;
    int pos_t;
    int count;
    MemoryAccessor mapping;

    updateSubCorr_Kernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor sub1_cec_Nt, MemoryAccessor sub2_cec_Nt, int sub_lt, int pos_t, int count, MemoryAccessor mapping) :
            gAcc(gauge.getAccessor()), sub1_cec_Nt(sub1_cec_Nt), sub2_cec_Nt(sub2_cec_Nt), sub_lt(sub_lt), pos_t(pos_t), count(count), mapping(mapping) {}

    __device__ __host__ void operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;
        int Nx = (int)GInd::getLatData().lx;
        int Ny = (int)GInd::getLatData().ly;
        int Nz = (int)GInd::getLatData().lz;
        const size_t elems1 = GInd::getLatData().vol3;
        const size_t elems2 = elems1 - (Nx-2)*(Ny-2)*(Nz-2);

        gSite shifted_site = GInd::getSite(site.coord.x,site.coord.y,site.coord.z,pos_t);

        sitexyzt coord = shifted_site.coord;

        size_t Id1 = coord[0] + coord[1]*Nx + coord[2]*Ny*Nx;
        size_t Id2;

        GSU3<floatT> temp1, temp2;
        /*caculate contribution from the left sublattice

 |      ^ ----------> | 
 |      |     |       | 
 |      |  -  |       | 
 |-----> --->         | 
 |                    | and its flipped one*/

        //loop over possible square positions of sub correlator. 
        //We do not update squares at the border of the sublattice. so sqPos=0 corresponds to postion of square - left border = 1
        for(int sqPos = 0; sqPos < sub_lt - 2; ++sqPos){
            for( size_t mu = 0; mu <= 2; ++mu){

                GSU3<floatT> p_up = gsu3_one<floatT>();
                GSU3<floatT> p_dn = gsu3_one<floatT>();
                gSite looppos_up = shifted_site; //initialize loop positions
                gSite looppos_dn = shifted_site;
        
                for (int j = 0; j < sqPos+1; ++j){
                    p_up *= gAcc.getLink(GInd::getSiteMu(looppos_up, 3));
                    p_dn *= gAcc.getLink(GInd::getSiteMu(looppos_dn, 3));
                    looppos_up = GInd::site_up(looppos_up, 3);
                    looppos_dn = GInd::site_up(looppos_dn, 3);
                }
        
                //rectangle
                p_up *= gAcc.getLink(GInd::getSiteMu(looppos_up, mu))
                      * gAcc.getLink(GInd::getSiteMu(GInd::site_up(looppos_up, mu), 3))
                      - gAcc.getLink(GInd::getSiteMu(looppos_up, 3))
                      * gAcc.getLink(GInd::getSiteMu(GInd::site_up(looppos_up, 3), mu));

                p_dn *= gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(looppos_dn, mu), mu))
                      * gAcc.getLink(GInd::getSiteMu(GInd::site_dn(looppos_dn, mu), 3))
                      - gAcc.getLink(GInd::getSiteMu(looppos_dn, 3))
                      * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(looppos_dn, 3, mu), mu));
        
                looppos_up = GInd::site_up_up(looppos_up, mu, 3);
                looppos_dn = GInd::site_up_dn(looppos_dn, 3, mu);
        
                //To the end
                for (int j = sqPos + 1; j < sub_lt - 1; ++ j){
                    p_up *= gAcc.getLink(GInd::getSiteMu(looppos_up, 3));
                    p_dn *= gAcc.getLink(GInd::getSiteMu(looppos_dn, 3));
                    looppos_up = GInd::site_up(looppos_up, 3);
                    looppos_dn = GInd::site_up(looppos_dn, 3);
                }
                //Distinct indexing for the combination
                sub1_cec_Nt.getValue<GSU3<floatT>>(pos_t*(elems1*6*(sub_lt-2)) + 6*elems1*sqPos + mu*elems1 + Id1, temp1);
                floatT factor = (1./count);
                sub1_cec_Nt.setValue<GSU3<floatT>>(pos_t*(elems1*6*(sub_lt-2)) + 6*elems1*sqPos + mu*elems1 + Id1, temp1 + p_up*factor);
                sub1_cec_Nt.getValue<GSU3<floatT>>(pos_t*(elems1*6*(sub_lt-2)) + 6*elems1*sqPos + (mu+3)*elems1 + Id1, temp2);
                sub1_cec_Nt.setValue<GSU3<floatT>>(pos_t*(elems1*6*(sub_lt-2)) + 6*elems1*sqPos + (mu+3)*elems1 + Id1, temp2 + p_dn*factor);
            }
        }
/*caculate contribution from the right sublattice. consider only the ones on the boundary.

 |------> ---->       | 
 |      |     |       | 
 |      |  -  |       | 
 |      ----->------> | 
 |                    | and its flipped one*/
        bool is_onBoundary = (coord[0] == 0 || coord[0]== Nx-1 || coord[1]==0 || coord[1]==Ny-1 || coord[2]==0 || coord[2]==Nz-1);
        if ( is_onBoundary ) {

            size_t Id2_tmp=0;
            for (int i=0;i<Nx;i++)
            {
                for (int j=0;j<Ny;j++)
                {
                    for(int k=0;k<Nz;k++)
                    {
                        if(i == 0 || i==Nx-1 || j==0 || j==Ny-1 || k==0 || k==Nz-1) {
                            mapping.setValue<size_t>(Id2_tmp, i+j*Nx+k*Nx*Ny);
                            Id2_tmp ++;
                        }
                    }
                }
            }

            size_t index;
            for (size_t i=0; i<elems2; i++) {
                mapping.getValue<size_t>(i, index); 
                if ( Id1 == index)
                {
                    Id2 = i;
                    break;
                }
            }

            for(int sqPos = 0; sqPos < sub_lt - 2; ++sqPos){ //loop over possible square positions of sub correlator. We do not update squares at the border of the sublattice. so sqPos=0 corresponds to postion of square - left border = 1
                for( size_t mu = 0; mu <= 2; ++mu){

                    GSU3<floatT> p_up = gsu3_one<floatT>();
                    GSU3<floatT> p_dn = gsu3_one<floatT>();
                    gSite looppos_up = shifted_site; //initialize loop positions
                    gSite looppos_dn = shifted_site;
 
                    looppos_up = GInd::site_up(looppos_up, mu);
                    looppos_dn = GInd::site_dn(looppos_dn, mu);

           
                    for (int j = 0; j < sqPos+1; ++j){
                        p_up *= gAcc.getLink(GInd::getSiteMu(looppos_up, 3));
                        p_dn *= gAcc.getLink(GInd::getSiteMu(looppos_dn, 3));
                        looppos_up = GInd::site_up(looppos_up, 3);
                        looppos_dn = GInd::site_up(looppos_dn, 3);
                    }
            
                    //rectangle
                    p_up *= gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_dn(looppos_up, mu), mu))
                            * gAcc.getLink(GInd::getSiteMu(GInd::site_dn(looppos_up, mu), 3))
                            - gAcc.getLink(GInd::getSiteMu(looppos_up, 3))
                            * gAcc.getLinkDagger(GInd::getSiteMu(GInd::site_up_dn(looppos_up, 3, mu), mu));
                    p_dn *= gAcc.getLink(GInd::getSiteMu(looppos_dn, mu))
                            * gAcc.getLink(GInd::getSiteMu(GInd::site_up(looppos_dn, mu), 3))
                            - gAcc.getLink(GInd::getSiteMu(looppos_dn, 3))
                            * gAcc.getLink(GInd::getSiteMu(GInd::site_up(looppos_dn, 3), mu));
 
                    looppos_up = GInd::site_up_dn(looppos_up, 3, mu);
                    looppos_dn = GInd::site_up_up(looppos_dn, 3, mu);
            
                    //To the end
                    for (int j = sqPos + 1; j < sub_lt - 1; ++ j){
                        p_up *= gAcc.getLink(GInd::getSiteMu(looppos_up, 3));
                        p_dn *= gAcc.getLink(GInd::getSiteMu(looppos_dn, 3));
                        looppos_up = GInd::site_up(looppos_up, 3);
                        looppos_dn = GInd::site_up(looppos_dn, 3);
                    }
                    //Distinct indexing for the combination
                    sub2_cec_Nt.getValue<GSU3<floatT>>(pos_t*(elems2*6*(sub_lt-2)) + 6*elems2*sqPos + mu*elems2 + Id2, temp1);
                    floatT factor = (1./count);
                    sub2_cec_Nt.setValue<GSU3<floatT>>(pos_t*(elems2*6*(sub_lt-2)) + 6*elems2*sqPos + mu*elems2 + Id2, temp1 + p_up*factor);
                    sub2_cec_Nt.getValue<GSU3<floatT>>(pos_t*(elems2*6*(sub_lt-2)) + 6*elems2*sqPos + (mu+3)*elems2 + Id2, temp2);
                    sub2_cec_Nt.setValue<GSU3<floatT>>(pos_t*(elems2*6*(sub_lt-2)) + 6*elems2*sqPos + (mu+3)*elems2 + Id2, temp2 + p_dn*factor);
                }
            }
        }
    }
};


template<class floatT, bool onDevice, size_t HaloDepth>
struct contractPoly_Kernel{
    gaugeAccessor<floatT> gAcc;
    MemoryAccessor sub_poly_Nt;
    int sub_lt;
    int min_dist;
    contractPoly_Kernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor sub_poly_Nt, int sub_lt, int min_dist) :
            gAcc(gauge.getAccessor()), sub_poly_Nt(sub_poly_Nt), sub_lt(sub_lt), min_dist(min_dist) { }

    __device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;
        int Ntau = (int)GInd::getLatData().globLT;
        int Nx = (int)GInd::getLatData().lx;
        int Ny = (int)GInd::getLatData().ly;
        int Nz = (int)GInd::getLatData().lz;
        const size_t elems1 = GInd::getLatData().vol3;

        sitexyzt coord = site.coord;
        size_t Id = coord[0] + coord[1]*Nx + coord[2]*Ny*Nx;

        GSU3<floatT> tmp1; //sub poly in the first sublattice at some spatial site
        GSU3<floatT> tmp2; //sub poly in the second sublattice at the same spatial site as tmp1

        int count = 0;
        GCOMPLEX(floatT) result(0,0);
        //loop over position of the first sublattice
        for (int i = 0; i < Ntau; ++i) {

            gSite running_site = GInd::getSite(site.coord.x,site.coord.y,site.coord.z,i);
            //loop over position of the second sublattice. position is not j but j+sub_lt
            for ( int j = i + min_dist; j <= i + (Ntau-2*sub_lt-min_dist); ++j) {
                int sub_dist = j - i;

                GSU3<floatT> p = gsu3_one<floatT>();
                sub_poly_Nt.getValue<GSU3<floatT>>(i*elems1+Id, tmp1);
                sub_poly_Nt.getValue<GSU3<floatT>>(((j+sub_lt)%Ntau)*elems1+Id, tmp2);

                p *= tmp1; //links of 1st sublattice
                for (int k = 0; k<sub_lt; k++ ) { //move site for the 1st sublattice
                    running_site = GInd::site_up(running_site,3);
                }

                for (int k = 0; k<sub_dist; k++ ) { //gap between the 1st and 2nd sublattice
                    p *= gAcc.getLink(GInd::getSiteMu(running_site, 3));
                    running_site = GInd::site_up(running_site,3);
                }

                p *= tmp2; //links of 2nd sublattice
                for (int k = 0; k<sub_lt; k++ ) { //move site for the 2nd sublattice
                    running_site = GInd::site_up(running_site,3);
                }

                for (int k = 0; k<Ntau-2*sub_lt-sub_dist; k++ ) { //links after the 2nd sublattice
                    p *= gAcc.getLink(GInd::getSiteMu(running_site, 3));
                    running_site = GInd::site_up(running_site,3);
                }

                result += tr_c(p);
                count++;
            }
        }
        return result/(count*3.0);
    }
};


template<class floatT, bool onDevice, size_t HaloDepth>
struct contractCorr_Kernel{
    gaugeAccessor<floatT> gAcc;
    MemoryAccessor sub1_cec_Nt;
    MemoryAccessor sub2_cec_Nt;
    int sub_lt;
    int min_dist;
    int tau; //will be the sqDist 
    MemoryAccessor mapping;
    contractCorr_Kernel(Gaugefield<floatT,onDevice,HaloDepth> &gauge, MemoryAccessor sub1_cec_Nt, MemoryAccessor sub2_cec_Nt, int sub_lt, int min_dist, int tau, MemoryAccessor mapping) :
            gAcc(gauge.getAccessor()), sub1_cec_Nt(sub1_cec_Nt), sub2_cec_Nt(sub2_cec_Nt), sub_lt(sub_lt), min_dist(min_dist), tau(tau), mapping(mapping)  { }

    __device__ __host__ GCOMPLEX(floatT) operator()(gSite site) {

        typedef GIndexer<All,HaloDepth> GInd;
        int Ntau = (int)GInd::getLatData().globLT;
        int Nx = (int)GInd::getLatData().lx;
        int Ny = (int)GInd::getLatData().ly;
        int Nz = (int)GInd::getLatData().lz;
        const size_t elems1 = GInd::getLatData().vol3;
        const size_t elems2 = elems1 - (Nx-2)*(Ny-2)*(Nz-2);

        sitexyzt coord = site.coord;
        size_t Id1 = coord[0] + coord[1]*Nx + coord[2]*Ny*Nx;
        size_t Id2;

        bool is_onBoundary = (coord[0] == 0 || coord[0]== Nx-1 || coord[1]==0 || coord[1]==Ny-1 || coord[2]==0 || coord[2]==Nz-1);

        GSU3<floatT> tmp1_up;
        GSU3<floatT> tmp1_dn;
        GSU3<floatT> tmp2_up;
        GSU3<floatT> tmp2_dn;

        int count = 0;
        GCOMPLEX(floatT) result(0,0);
        //loop over position of the first sublattice
        for (int i = 0; i < Ntau; ++i) {
            //loop over position of the second sublattice. position is not j but j+sub_lt
            for ( int j = i + min_dist; j <= i + (Ntau-2*sub_lt-min_dist); ++j) {
                int sub_dist = j - i;
                //loop over possible square positions within the first sublattice
                for (int sqPos1 = 0; sqPos1 < sub_lt-2; ++sqPos1) {
                    //Distance from the 1st square position to the right border of the left sublattice
                    int sqDist1 = sub_lt-sqPos1-1;
                    //Distance from the 2nd square position to the left border of the right sublattice
                    int sqDist2 = tau - sqDist1 - sub_dist;
                    int sqPos2 = sqDist2 - 1;
                    if ( 0 <= sqPos2 && sqPos2 < sub_lt-2) {
                        //loop over possible spatial direction for the link directing spatially in the square
                        for( size_t mu = 0; mu <= 2; ++mu) {

                            gSite running_site_up = GInd::getSite(site.coord.x,site.coord.y,site.coord.z,i);
                            gSite running_site_dn = GInd::getSite(site.coord.x,site.coord.y,site.coord.z,i);
                            //get the sub contribution in direction mu from the 1st square at sqPos1
                            sub1_cec_Nt.getValue<GSU3<floatT>>(i*(6*elems1*(sub_lt-2)) + 6*elems1*sqPos1+mu*elems1+Id1, tmp1_up);    
                            sub1_cec_Nt.getValue<GSU3<floatT>>(i*(6*elems1*(sub_lt-2)) + 6*elems1*sqPos1+(mu+3)*elems1+Id1, tmp1_dn);    

                            GSU3<floatT> p_up =gsu3_one<floatT>();                            
                            GSU3<floatT> p_dn =gsu3_one<floatT>();                            

                            //contribution from 1st sublattice
                            p_up *= tmp1_up;  
                            p_dn *= tmp1_dn;
 
                            for(int k=0; k<sub_lt; k++) { //move site for the 1st sublattice
                                running_site_up = GInd::site_up(running_site_up, 3);
                                running_site_dn = GInd::site_up(running_site_dn, 3);
                            }
                            running_site_up = GInd::site_up(running_site_up, mu);
                            running_site_dn = GInd::site_dn(running_site_dn, mu);

                            for(int k=0; k<sub_dist; k++) { // gap
                                p_up *= gAcc.getLink(GInd::getSiteMu(running_site_up, 3));
                                running_site_up = GInd::site_up(running_site_up,3);
                                p_dn *= gAcc.getLink(GInd::getSiteMu(running_site_dn, 3));
                                running_site_dn = GInd::site_up(running_site_dn,3);
                            }

                            if ( is_onBoundary ) {
                                size_t Id2_tmp=0;
                                for (int i=0;i<Nx;i++)
                                {
                                    for (int j=0;j<Ny;j++)
                                    {
                                        for(int k=0;k<Nz;k++)
                                        {
                                            if(i == 0 || i==Nx-1 || j==0 || j==Ny-1 || k==0 || k==Nz-1) {
                                                mapping.setValue<size_t>(Id2_tmp, i+j*Nx+k*Nx*Ny);
                                                Id2_tmp ++;
                                            }
                                        }
                                    }
                                }

                                size_t index;
                                for (size_t i=0; i<elems2; i++) {
                                    mapping.getValue<size_t>(i, index);
                                    if ( Id1 == index)
                                    {
                                        Id2 = i;
                                        break;
                                    }
                                }
 
                                //contribution from the second square, using the values on the boundary saved in sub2_cec_Nt
                                sub2_cec_Nt.getValue<GSU3<floatT>>(((j+sub_lt)%Ntau)*(6*elems2*(sub_lt-2)) + 6*elems2*sqPos2+mu*elems2+Id2, tmp2_up);
                                sub2_cec_Nt.getValue<GSU3<floatT>>(((j+sub_lt)%Ntau)*(6*elems2*(sub_lt-2)) + 6*elems2*sqPos2+(mu+3)*elems2+Id2, tmp2_dn);
                            } else {
                                size_t Id3 = coord[0]+(mu==0) + (coord[1]+(mu==1))*Nx + (coord[2]+(mu==2))*Ny*Nx;
                                size_t Id4 = coord[0]-(mu==0) + (coord[1]-(mu==1))*Nx + (coord[2]-(mu==2))*Ny*Nx;
                                //contribution from the second squar, using the same part of sub1
                                sub1_cec_Nt.getValue<GSU3<floatT>>(((j+sub_lt)%Ntau)*(6*elems1*(sub_lt-2)) + 6*elems1*sqPos2+(mu+3)*elems1+Id3, tmp2_up);
                                sub1_cec_Nt.getValue<GSU3<floatT>>(((j+sub_lt)%Ntau)*(6*elems1*(sub_lt-2)) + 6*elems1*sqPos2+mu*elems1+Id4, tmp2_dn);
                            }
                            //contribution from 2nd sublattice
                            p_up *= tmp2_up;   
                            p_dn *= tmp2_dn;

                            for(int k=0; k<sub_lt; k++) { //move site for the 2nd sublattice
                                running_site_up = GInd::site_up(running_site_up, 3);
                                running_site_dn = GInd::site_up(running_site_dn, 3);
                            }
                            running_site_up = GInd::site_dn(running_site_up, mu);
                            running_site_dn = GInd::site_up(running_site_dn, mu);
                            
                            //links after the 2nd sublattice
                            for (int k = 0; k<Ntau-2*sub_lt-sub_dist; k++ ) { 
                                p_up *= gAcc.getLink(GInd::getSiteMu(running_site_up, 3));
                                running_site_up = GInd::site_up(running_site_up,3);
                                p_dn *= gAcc.getLink(GInd::getSiteMu(running_site_dn, 3));
                                running_site_dn = GInd::site_up(running_site_dn,3);
                            }
                            result += tr_c(p_up + p_dn);
                        }
                        //counter for normelization
                        count++; 
                    }
                }
            }        
        }
        if(count == 0) {
            result = GCOMPLEX(floatT)(0,0);
        } else {
            result /= (count*3.0);
        }
        return result;
    }
};

template<class floatT, bool onDevice, size_t HaloDepth>
void SubLatMeas<floatT, onDevice, HaloDepth>::updateSubPolyCorr(int pos_t, int count, MemoryAccessor &sub_poly_Nt, MemoryAccessor &sub1_cec_Nt, MemoryAccessor &sub2_cec_Nt) {

    typedef gMemoryPtr<true>  MemTypeGPU;
    
    MemTypeGPU mem47 = MemoryManagement::getMemAt<true>("mapping_gpu");
    mem47->template adjustSize<size_t>(_elems2);
    MemoryAccessor mapping (mem47->getPointer());     

    ReadIndexSpatial<HaloDepth> calcReadIndexSpatial;
    //calculate sub poly and sub cec 
    iterateFunctorNoReturn<onDevice>(updateSubPoly_Kernel<floatT,onDevice,HaloDepth>(_gauge, sub_poly_Nt, _sub_lt, pos_t, count), calcReadIndexSpatial, _elems1);
    iterateFunctorNoReturn<onDevice>(updateSubCorr_Kernel<floatT,onDevice,HaloDepth>(_gauge, sub1_cec_Nt, sub2_cec_Nt, _sub_lt, pos_t, count, mapping), calcReadIndexSpatial, _elems1);

}

template<class floatT, bool onDevice, size_t HaloDepth>
GCOMPLEX(floatT) SubLatMeas<floatT, onDevice, HaloDepth>::contraction_poly(MemoryAccessor &sub_poly_Nt, int min_dist) {

     GCOMPLEX(floatT) PolyakovLoop_result;
    _redBaseCE.template iterateOverSpatialBulk<All, HaloDepth>(contractPoly_Kernel<floatT,onDevice,HaloDepth>(_gauge, sub_poly_Nt, _sub_lt, min_dist));
    _redBaseCE.reduce(PolyakovLoop_result, _elems1);
    return PolyakovLoop_result / _spatialvol;
}


template<class floatT, bool onDevice, size_t HaloDepth>
std::vector<GCOMPLEX(floatT)> SubLatMeas<floatT, onDevice, HaloDepth>::contraction_cec(MemoryAccessor &sub1_cec_Nt, MemoryAccessor &sub2_cec_Nt, int min_dist) {

    typedef gMemoryPtr<true>  MemTypeGPU;
    MemTypeGPU mem48 = MemoryManagement::getMemAt<true>("mapping_gpu");
    mem48->template adjustSize<size_t>(_elems2);
    mem48->memset(0);
    MemoryAccessor mapping (mem48->getPointer());

    //final contraction for color electric corr
    std::vector<GCOMPLEX(floatT)> ColorElectricCorr_result(_Nt/2+1, 0);
    for ( int tau=0; tau<_Nt/2+1; tau++ ) {
        _redBaseCE.template iterateOverSpatialBulk<All, HaloDepth>(contractCorr_Kernel<floatT,onDevice,HaloDepth>(_gauge, sub1_cec_Nt, sub2_cec_Nt, _sub_lt, min_dist, tau, mapping));
        _redBaseCE.reduce(ColorElectricCorr_result[tau], _elems1);
        ColorElectricCorr_result[tau] /= _spatialvol;
    }
    return ColorElectricCorr_result;
}


#define CLASS_INIT(floatT, HALO) \
template class SubLatMeas<floatT,true,HALO>; \

INIT_PH(CLASS_INIT)
