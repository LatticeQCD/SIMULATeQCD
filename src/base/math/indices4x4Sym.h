//
// Created by Jonas Winter on 05.09.2025
//

#pragma once
#include "../../define.h"


enum indices4x4Sym {
    index00, index11, index22, index33, index01, index02, index03, index12, index13, index23
};


// helper function for turning two indiviual indices (mu and nu) into the indexPair index
// index00, index11, index22, index33, index01, index02, index03, index12, index13, index23
//    0        1        2        3        4        5        6        7        8        9
// e.g. mu,nu=1,3 to index13=8 and mu,nu=2,0 to index02=5
inline int twoIndicesToIndexPairIndex(int mu, int nu) {

    if (mu == 0 && nu == 0) return index00;
    if (mu == 0 && nu == 1) return index01;
    if (mu == 0 && nu == 2) return index02;
    if (mu == 0 && nu == 3) return index03;
    
    if (mu == 1 && nu == 0) return index01;
    if (mu == 1 && nu == 1) return index11;
    if (mu == 1 && nu == 2) return index12;
    if (mu == 1 && nu == 3) return index13;
    
    if (mu == 2 && nu == 0) return index02;
    if (mu == 2 && nu == 1) return index12;
    if (mu == 2 && nu == 2) return index22;
    if (mu == 2 && nu == 3) return index23;
    
    if (mu == 3 && nu == 0) return index03;
    if (mu == 3 && nu == 1) return index13;
    if (mu == 3 && nu == 2) return index23;
    if (mu == 3 && nu == 3) return index33;

    // if indices are not matched above, return negative index
    return -1;
    
}

// // helper function for turning two indiviual indices (mu and nu) into the indexPair index
// // symmetric 4x4 matrix has 10 entries:
// // index00, index11, index22, index33, index01, index02, index03, index12, index13, index23
// //  0    1    2    3    4    5    6    7    8    9
// // coresponding to two indices mu, nu according to the upper variable names
// inline int twoIndicesToIndexPairIndex(int mu, int nu) {
//     // if outside of range, return error via negativ index
//     if (mu > 3 || mu < 0 || nu > 3 || nu < 0) return -1;

//     if (mu == nu) return mu; // the diagonal components mu,mu (indexPairIndices 0 1 2 3 = mu)
//     else if (mu == 0) return nu + 3; // the temporal components 0,nu (indexPairIndices 4, 5, 6 = nu + 3)
//     else if (nu == 0) return mu + 3; // the temporal components 0,mu (indexPairIndices 4, 5, 6 = mu + 3)
//     else return mu + nu + 4; // the purely spatial components mu,nu (indexPairIndices 7, 8, 9 = mu + nu + 4)
// }
