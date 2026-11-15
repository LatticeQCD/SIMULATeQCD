/*
 * Opt-in dump of the MD state at every position where the force of the first light Hasenbusch terms is evaluated
 * (deflation study, Step 0 of DSDRMDWFruns/SQCDruns/deflation/DESIGN.md). Off by default: with enabled = false the
 * HMC only reads this flag, so the trajectory is bit-identical to a build without the dump.
 *
 * The application (mdwfTwoPlusOneHmcRun, defl_dump = 1) enables the control for one trajectory. Then
 *   MDWFHmcDriver::computeTermForce   for term t < terms, before the force: a new "evaluation" e for every new gauge
 *                                     field (several terms at the same field share it) with the thin links U and the
 *                                     momenta P,            <dir>/eNNN_U.r<rank>, <dir>/eNNN_P.r<rank>
 *   MDWFEvenOddTwoFlavorRatioFermionAction::force
 *                                     the right-hand side psi and the solution chi of the force solve
 *                                     (A A^+) chi = psi (light_form = 1; A^+ A for light_form = 0), A = Mhat(m_t),
 *                                     psi = Mhat(m_t+1) phi_t,  <dir>/eNNN_t<t>_rhs.r<rank>, <dir>/eNNN_t<t>_sol.r<rank>
 *   MDWFHmcDriver::computeTermForce   after the force, one line of <dir>/index.tsv (root):
 *                                     eval  term  kind  md_time  gauge_version  solver_iterations  force_seconds
 *                                     kind 0 = a position of the trajectory, 1 = the displaced field exp(c F) U of a
 *                                     force-gradient kick (momenta as before the kick); md_time = sum of the Q steps
 *                                     since the dump was enabled (this code's units, t_GPT = sqrt(2) t).
 * All fields in the format of MDWFFieldDump.h (bit-exact, any node decomposition on reading).
 */

#pragma once

#include "MDWFFieldDump.h"

#include <cstdio>
#include <string>

struct MDWFDeflationDumpControl {
    bool enabled = false;      // application: dump during this trajectory
    std::string dir;           // existing directory
    int terms = 2;             // force terms 0 ... terms - 1
    long eval = -1;            // index of the last dumped gauge field
    long gaugeVersion = -1;    // the driver's gauge version of that field
    double mdTimeStart = 0.0;  // the driver's MD clock when the dump was enabled
    bool clockSet = false;
    // Driver -> Hasenbusch factor, around one force evaluation:
    bool armed = false;        // the next ratio-factor force solve is dumped (cleared by the factor)
    int term = -1;
    int solveIterations = -1;  // set by the factor
};

inline MDWFDeflationDumpControl &mdwfDeflationDump() {
    static MDWFDeflationDumpControl control;
    return control;
}

inline std::string mdwfDeflationDumpEval(const std::string &dir, long eval) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "e%03ld", eval);
    return dir + "/" + buffer;
}

inline std::string mdwfDeflationDumpTerm(const std::string &dir, long eval, int term, const char *what) {
    return mdwfDeflationDumpEval(dir, eval) + "_t" + std::to_string(term) + "_" + what;
}
