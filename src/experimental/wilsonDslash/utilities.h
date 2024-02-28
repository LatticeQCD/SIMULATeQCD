
template<class floatT, size_t HaloDepth>
struct ScalarProductKernel{

    //Gauge accessor to access the gauge field
    //SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor_lhs;
    SpinorColorAcc<floatT> _SpinorColorAccessor_rhs;
    using SpinorLHS=Spinorfield<floatT, true, All, HaloDepth, 12>;
    using SpinorRHS=Spinorfield<floatT, true, All, HaloDepth, 12>;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    ScalarProductKernel(SpinorLHS &spinor_LHS, SpinorRHS &spinor_RHS) 
                : _SpinorColorAccessor_lhs(spinor_LHS.getAccessor()),
                  _SpinorColorAccessor_rhs(spinor_RHS.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(floatT) operator()(gSite site) {
      
      ColorVect<floatT> spinCol;
      COMPLEX(floatT) res(0.0,0.0);
      ColorVect<floatT> spinCol_lhs = _SpinorColorAccessor_lhs.getColorVect(site);
      ColorVect<floatT> spinCol_rhs = _SpinorColorAccessor_rhs.getColorVect(site);
      for(int i = 0 ; i < 4 ; i++){
          res+= re_dot_prod(spinCol_lhs[i], spinCol_rhs[i]);
      }
        return res;
    }
};
template<class floatT, size_t HaloDepth>
COMPLEX(floatT) ScalarProduct(Spinorfield<floatT, true, All, HaloDepth, 12>& spinor_lhs,Spinorfield<floatT, true, All, HaloDepth, 12>& spinor_rhs){
  LatticeContainer<true,COMPLEX(double)> _redBase(spinor_rhs.getComm());

  _redBase.adjustSize(spinor_lhs.getNumberElements());

  _redBase.template iterateOverBulk<All, HaloDepth>(
      ScalarProductKernel<floatT, HaloDepth>(spinor_lhs, spinor_rhs));

  COMPLEX(double) result = 0;
  _redBase.reduce(result, spinor_lhs.getNumberElements());   
  
    return result; 
}

template<class floatT, size_t HaloDepth>
struct ScalarProductKernelEven{

    //Gauge accessor to access the gauge field
    //SU3Accessor<floatT> _SU3Accessor;
    SpinorColorAcc<floatT> _SpinorColorAccessor_lhs;
    SpinorColorAcc<floatT> _SpinorColorAccessor_rhs;
    using SpinorLHS=Spinorfield<floatT, true, Even, HaloDepth, 12>;
    using SpinorRHS=Spinorfield<floatT, true, Even, HaloDepth, 12>;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    ScalarProductKernelEven(SpinorLHS &spinor_LHS, SpinorRHS &spinor_RHS) 
                : _SpinorColorAccessor_lhs(spinor_LHS.getAccessor()),
                  _SpinorColorAccessor_rhs(spinor_RHS.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(floatT) operator()(gSite site) {
      
      ColorVect<floatT> spinCol;
      COMPLEX(floatT) res(0.0,0.0);
      ColorVect<floatT> spinCol_lhs = _SpinorColorAccessor_lhs.getColorVect(site);
      ColorVect<floatT> spinCol_rhs = _SpinorColorAccessor_rhs.getColorVect(site);
      for(int i = 0 ; i < 4 ; i++){
          res+= re_dot_prod(spinCol_lhs[i], spinCol_rhs[i]);
      }
        return res;
    }
};
template<class floatT, size_t HaloDepth>
COMPLEX(floatT) ScalarProductEven(Spinorfield<floatT, true, Even, HaloDepth, 12>& spinor_lhs,Spinorfield<floatT, true, Even, HaloDepth, 12>& spinor_rhs){
  LatticeContainer<true,COMPLEX(double)> _redBase(spinor_rhs.getComm());

  _redBase.adjustSize(spinor_lhs.getNumberElements());

  _redBase.template iterateOverBulk<All, HaloDepth>(
      ScalarProductKernelEven<floatT, HaloDepth>(spinor_lhs, spinor_rhs));

  COMPLEX(double) result = 0;
  _redBase.reduce(result, spinor_lhs.getNumberElements());   
  
    return result; 
}

template<class floatT, size_t HaloDepth>
struct PointSourceKernel{

    //Gauge accessor to access the gauge field
    //SU3Accessor<floatT> _SU3Accessor;
 //   SpinorColorAcc<floatT> _SpinorColorAccessor_lhs;
    SpinorColorAcc<floatT> _SpinorColorAccessor_rhs;
//    using SpinorLHS=Spinorfield<floatT, true, All, HaloDepth, 12>;
    using SpinorRHS=Spinorfield<floatT, true, All, HaloDepth, 12>;

    typedef GIndexer<All, HaloDepth > GInd;
    //Constructor to initialize all necessary members.
    PointSourceKernel(SpinorRHS &spinor_RHS) 
                : _SpinorColorAccessor_rhs(spinor_RHS.getAccessor())
    { }

    //This is the operator that is called inside the Kernel
    __device__ __host__ COMPLEX(floatT) operator()(gSite site, int spin, int color) {
      
      ColorVect<floatT> spinCol;
      COMPLEX(floatT) res(0.0,0.0);
//      ColorVect<floatT> spinCol_lhs = _SpinorColorAccessor_lhs.getColorVect(site);
      ColorVect<floatT> spinCol_rhs = _SpinorColorAccessor_rhs.getColorVect(site);
      spinCol_rhs[spin][color]=1.0;
      return ;
    }
};
template<class floatT, size_t HaloDepth>
COMPLEX(floatT) PointSource(Spinorfield<floatT, true, All, HaloDepth, 12>& spinor_rhs, gSite site, int spin, int color){
  LatticeContainer<true,COMPLEX(double)> _redBase(spinor_rhs.getComm());
//  _redBase.adjustSize(spinor_lhs.getNumberElements());
  _redBase.template iterateOverBulk<All, HaloDepth>(
      PointSourceKernel<floatT, HaloDepth>(spinor_rhs, site, spin, color));

  COMPLEX(double) result = 0;
//  _redBase.reduce(result, spinor_lhs.getNumberElements());   
  
    return result; 
}
