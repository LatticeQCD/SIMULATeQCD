/*
 * SubLatMeas.cpp
 *
 * Hai-Tao Shu, 31.07.2019
 *
 * Implementations of contractions of both standard operators and multi-level improved operators.
 *
 **/

#include "SubLatMeas.h"

template<class floatT>
void Contraction_cpu<floatT>::ImproveNormalizeBulk(std::vector<floatT> &SubBulk_Nt_real, std::vector<floatT> &SubBulk_Nt_p0, int count) {

    for ( int pos_t=0;pos_t<_Nt;pos_t++ ) {
        for ( int dist=0; dist<_sub_lt-3; dist++ ) {
            int mid = count;
            int count_cp=count;
            while(mid != 0)
            {
                 mid = count_cp/2;
                 for (int i=0;i<mid;i++ )
                 {
                     int span = count_cp-i-1;
                     SubBulk_Nt_p0[pos_t*(_sub_lt-3)*count+dist*count+i] += SubBulk_Nt_p0[pos_t*(_sub_lt-3)*count+dist*count+span];
                 }
                 if ( count_cp%2==0 )
                     count_cp /= 2;
                 else
                     count_cp = (count_cp+1)/2;
            }
            SubBulk_Nt_real[pos_t*(_sub_lt-3)+dist] = SubBulk_Nt_p0[pos_t*(_sub_lt-3)*count+dist*count]/count;
        }
    }
}

template<class floatT>
void Contraction_cpu<floatT>::ImproveNormalizeShear(std::vector<Matrix4x4Sym<floatT> > &SubShear_Nt_real, std::vector<Matrix4x4Sym<floatT> > &SubShear_Nt_p0, int count) {

    for ( int pos_t=0;pos_t<_Nt;pos_t++ ) {
        for ( int dist=0; dist<_sub_lt-3; dist++ ) {
            int mid = count;
            int count_cp=count;
            while(mid != 0)
            {
                 mid = count_cp/2;
                 for (int i=0;i<mid;i++ )
                 {
                     int span = count_cp-i-1;
                     SubShear_Nt_p0[pos_t*(_sub_lt-3)*count+dist*count+i] += SubShear_Nt_p0[pos_t*(_sub_lt-3)*count+dist*count+span];
                 }
                 if ( count_cp%2==0 )
                     count_cp /= 2;
                 else
                     count_cp = (count_cp+1)/2;
            }
            SubShear_Nt_p0[pos_t*(_sub_lt-3)*count+dist*count] /= count;
            SubShear_Nt_real[pos_t*(_sub_lt-3)+dist] = SubShear_Nt_p0[pos_t*(_sub_lt-3)*count+dist*count];
        }
    }
}

template<class floatT>
void Contraction_cpu<floatT>::ImproveContractionBulk(std::vector<floatT> &SubBulk_Nt_real, std::vector<floatT> &SubBulk_Nt_imag, int min_dist,
                                                     size_t global_spatial_vol, int pz, std::vector<floatT> &Improve_BulkResult) {


    std::vector<int> count(_Nt, 0);

    if ( pz==0 ) { //to reduce numerical error in summation
        std::vector<std::vector<floatT> > Improve_BulkResult_temp(_Nt);
        //loop over position of the first sublattice
        for (int i = 0; i < _Nt; ++i) {
            //loop over position of the second sublattice. position is j
            for (int j = i+min_dist+_sub_lt; j <= i+_Nt-_sub_lt-min_dist; ++j) {
                //loop over possible time position within the first sublattice
                for (int Pos1 = 0; Pos1 < _sub_lt-3; ++Pos1) {
                    //loop over possible time position within the second sublattice
                    for (int Pos2 = 0; Pos2 < _sub_lt-3; ++Pos2) {
                         //the time distance
                         int PosDist = j - i + Pos2 - Pos1;
                         Improve_BulkResult_temp[PosDist].push_back( SubBulk_Nt_real[i*(_sub_lt-3)+Pos1]
                                                                   * SubBulk_Nt_real[(j%_Nt)*(_sub_lt-3)+Pos2] );
                         count[PosDist] ++ ;
                    }
                }
            }
        }
        for (int i = 0; i < _Nt; ++i) {
            if( count[i] != 0 ) {
                int mid = count[i];
                int count_cp=count[i];
                while(mid != 0)
                {
                     mid = count_cp/2;
                     for (int j=0;j<mid;j++ )
                     {
                         int span = count_cp-j-1;
                         Improve_BulkResult_temp[i][j] += Improve_BulkResult_temp[i][span];
                     }
                     if ( count_cp%2==0 )
                         count_cp /= 2;
                     else
                         count_cp = (count_cp+1)/2;
                }
                Improve_BulkResult[i] = Improve_BulkResult_temp[i][0];
            }
        }
    } else {

        //loop over position of the first sublattice
        for (int i = 0; i < _Nt; ++i) {
            //loop over position of the second sublattice. position is j
            for (int j = i+min_dist+_sub_lt; j <= i+_Nt-_sub_lt-min_dist; ++j) {
                //loop over possible time position within the first sublattice
                for (int Pos1 = 0; Pos1 < _sub_lt-3; ++Pos1) {
                    //loop over possible time position within the second sublattice
                    for (int Pos2 = 0; Pos2 < _sub_lt-3; ++Pos2) {
                         //the time distance
                         int PosDist = j - i + Pos2 - Pos1;
                         Improve_BulkResult[pz*_Nt+PosDist] += SubBulk_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1]
                                                             * SubBulk_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2]
                                                             + SubBulk_Nt_imag[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1]
                                                             * SubBulk_Nt_imag[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2];
                         count[PosDist] ++ ;
                    }
                }
            }
        }
    }
    for (int i = 0; i < _Nt; ++i) {
        if ( count[i] != 0 ) {
            Improve_BulkResult[pz*_Nt+i] *= global_spatial_vol*1./count[i];
        }
    }
}


template<class floatT>
void Contraction_cpu<floatT>::ImproveContractionShear(std::vector<Matrix4x4Sym<floatT> > &SubShear_Nt_real,
                                                      std::vector<Matrix4x4Sym<floatT> > &SubShear_Nt_imag, int min_dist,
                                                      size_t global_spatial_vol, int pz, std::vector<floatT> &Improve_ShearResult) {

    std::vector<int> count(_Nt, 0);
    if ( pz == 0 ) {
        std::vector<std::vector<floatT> > Improve_ShearResult_temp(_Nt);
        //loop over position of the first sublattice
        for (int i = 0; i < _Nt; ++i) {
            //loop over position of the second sublattice. position is j
            for (int j = i+min_dist+_sub_lt; j <= i+_Nt-_sub_lt-min_dist; ++j) {
                //loop over possible time position within the first sublattice
                for (int Pos1 = 0; Pos1 < _sub_lt-3; ++Pos1) {
                    //loop over possible time position within the second sublattice
                    for (int Pos2 = 0; Pos2 < _sub_lt-3; ++Pos2) {
                        //the time distance
                        int PosDist = j - i + Pos2 - Pos1;
                        Improve_ShearResult_temp[PosDist].push_back(0.25*( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[0]
                                                                         - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[1])
                                                                        *( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[0]
                                                                         - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[1])
                                                                  + 0.25*( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[0]
                                                                         - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[2])
                                                                        *( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[0]
                                                                         - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[2])
                                                                  + 0.25*( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[1]
                                                                         - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[2])
                                                                        *( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[1]
                                                                         - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[2]));
                        count[PosDist] ++ ;
                    }
                }
            }
        }
        for (int i = 0; i < _Nt; ++i) {
            if( count[i] != 0 ) {
                int mid = count[i];
                int count_cp=count[i];
                while(mid != 0)
                {
                     mid = count_cp/2;
                     for (int j=0;j<mid;j++ )
                     {
                         int span = count_cp-j-1;
                         Improve_ShearResult_temp[i][j] += Improve_ShearResult_temp[i][span];
                     }
                     if ( count_cp%2==0 )
                         count_cp /= 2;
                     else
                         count_cp = (count_cp+1)/2;
                }
                Improve_ShearResult[i] = Improve_ShearResult_temp[i][0];
            }
        }
    } else {
        //loop over position of the first sublattice
        for (int i = 0; i < _Nt; ++i) {
            //loop over position of the second sublattice. position is j
            for (int j = i+min_dist+_sub_lt; j <= i+_Nt-_sub_lt-min_dist; ++j) {
                //loop over possible time position within the first sublattice
                for (int Pos1 = 0; Pos1 < _sub_lt-3; ++Pos1) {
                    //loop over possible time position within the second sublattice
                    for (int Pos2 = 0; Pos2 < _sub_lt-3; ++Pos2) {
                        //the time distance
                        int PosDist = j - i + Pos2 - Pos1;
                        Improve_ShearResult[pz*_Nt+PosDist] += 0.25*( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[0]
                                                                    - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[2])
                                                                   *( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[0]
                                                                    - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[2])
                                                             + 0.25*( SubShear_Nt_imag[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[0]
                                                                    - SubShear_Nt_imag[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[2])
                                                                   *( SubShear_Nt_imag[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[0]
                                                                    - SubShear_Nt_imag[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[2]);
                        Improve_ShearResult[pz*_Nt+PosDist] += 0.25*( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[1]
                                                                    - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[2])
                                                                   *( SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[1]
                                                                    - SubShear_Nt_real[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[2])
                                                             + 0.25*( SubShear_Nt_imag[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[1]
                                                                    - SubShear_Nt_imag[pz*_Nt*(_sub_lt-3)+i*(_sub_lt-3)+Pos1].elems[2])
                                                                   *( SubShear_Nt_imag[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[1]
                                                                    - SubShear_Nt_imag[pz*_Nt*(_sub_lt-3)+(j%_Nt)*(_sub_lt-3)+Pos2].elems[2]);
                        count[PosDist] ++ ;
                    }
                }
            }
        }
    }


    for (int i = 0; i < _Nt; ++i) {
        if ( count[i] != 0 ) {
            if ( pz == 0 ) {
                Improve_ShearResult[pz*_Nt+i] *= global_spatial_vol/3./count[i];
            } else {
                Improve_ShearResult[pz*_Nt+i] *= global_spatial_vol/2./count[i];
            }
        }
    }
}

template class Contraction_cpu<double>;
template class Contraction_cpu<float>;
