/*
 * Test-only general-Mobius force-workspace view.
 *
 * For the Mobius operator M = D_W(U) Din + Shift, Din and Shift are
 * gauge-independent, so dM = dD_W Din and
 *
 *     dS = -2 sum_i a_i Re[ eta_i^\dagger dD_W (Din chi_i) ],
 *
 * with chi_i = (M^\dagger M + sigma_i)^{-1} phi and eta_i = M chi_i from an
 * existing MDWFFermionForceWorkspace built on the Mobius normal/forward
 * operators. This is the Shamir contraction with the right vector chi_i
 * replaced by Din chi_i (validated per selected link by
 * mdwfMobiusForceContractionTest).
 *
 * The existing contraction and all-link storage helpers read the right
 * vector through chi(term). This view therefore returns Din chi_i from
 * chi(term), and the unmodified base chi_i from baseChi(term); eta(term) and
 * the solver diagnostics are forwarded from the base workspace. It computes
 * Din chi_i once at construction (halo refreshed, since the Wilson and clover
 * contractions read spatial neighbors) and does not define a production
 * force, projection, ipdot convention, HMC sign, or MPI ownership.
 */

#pragma once

#include "MDWFFifthDim.h"

#include <memory>
#include <string>
#include <vector>

template<class floatT, size_t HaloDepth, size_t Ls, class BaseWorkspace>
class MDWFMobiusForceWorkspaceView {
public:
    using Spinor = MDWFSpinor<floatT, true, All, HaloDepth, Ls>;

private:
    const BaseWorkspace &_base;
    std::vector<std::unique_ptr<Spinor>> _din_chi;

public:
    MDWFMobiusForceWorkspaceView(const BaseWorkspace &base,
                                 const MDWFFifthDimCoefficients<floatT> &din_coeff,
                                 const std::string &name = "MDWF_mobius_force_workspace_view")
        : _base(base) {
        _din_chi.reserve(base.size());
        for (size_t term = 0; term < base.size(); term++) {
            _din_chi.emplace_back(new Spinor(base.chi(term).getComm(),
                                             name + "_din_chi_" + std::to_string(term)));
            applyMDWFFifthDimCoupling<floatT, true, All, HaloDepth, Ls>(
                *_din_chi.back(), base.chi(term), din_coeff, true);
        }
    }

    MDWFMobiusForceWorkspaceView(const MDWFMobiusForceWorkspaceView &) = delete;
    MDWFMobiusForceWorkspaceView &operator=(const MDWFMobiusForceWorkspaceView &) = delete;

    size_t size() const {
        return _din_chi.size();
    }

    // Right contraction vector Din chi_i, under the name the existing helpers read.
    const Spinor &chi(size_t term) const {
        return *_din_chi.at(term);
    }

    const Spinor &baseChi(size_t term) const {
        return _base.chi(term);
    }

    const Spinor &eta(size_t term) const {
        return _base.eta(term);
    }

    const auto &shiftInfo() const {
        return _base.shiftInfo();
    }

    bool converged() const {
        return _base.converged();
    }
};
