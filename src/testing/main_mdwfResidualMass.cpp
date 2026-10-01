/*
 * Residual-mass measurement for Mobius (clover) domain-wall fermions
 * (MDWFResidualMass.h, Grid conventions).
 *
 * For one gauge configuration and one point source: 12 even/odd
 * preconditioned solves, each contracted along all four directions. The
 * analysis direction (`direction` = x | y | z | t | spatial; spatial = sum
 * over x, y, z, which needs equal spatial extents) gives C_PP(d), C_J5q(d)
 * and m_res(d) = C_J5q / C_PP as functions of the distance d from the source
 * along that direction, folded (d and L - d averaged), written to
 * <measurements_dir>/<output_name>, plus the plateau value over d in
 * [plateau_min, plateau_max] as ratio of summed folded correlators and as
 * mean of the folded ratios. On finite-temperature lattices use the spatial
 * (screening) correlators; the temporal ones on N_t = 8 reach only d = 4.
 *
 * Each solve's contribution (all four directions) is written to
 * <output_name>.cNN (NN = source component 00..11) as soon as it is done. A
 * rerun with the same physics parameters reads the finished components
 * instead of solving again, so an interrupted job loses at most the solve in
 * progress, and jobs with disjoint `components` ranges can run in parallel
 * on one configuration. The final output is written by whichever run finds
 * all 12 present; a run with nothing left to solve only assembles, so another
 * `direction` or plateau costs no solves.
 *
 * Regression test against Grid/GPT: with reference_file set, the raw per-slice
 * correlators (all four directions) are compared with a reference file
 * (lines "mu n C_PP C_J5q"), and the run fails if any slice of C_PP, C_J5q or
 * m_res differs by more than reference_tolerance (relative). Reference:
 * test_conf/mdwf_mres_grid_reference_b1801_b.txt, cross-checked against an
 * independent Grid/GPT measurement; setup in
 * parameter/tests/mdwfResidualMassGridReference.param (needs the HotQCD
 * configuration, not in the repository; 12 solves, about 1 h on one A100).
 *
 * Usage (from the build's testing directory):
 *   ./mdwfResidualMass <param file> [key=value ...]
 * Parameters: parameter/tests/mdwfResidualMass.param. Gaugefile = NERSC
 * configuration; without it the field is random (start = random) or unit
 * (start = unit), for tests. ls = 8 or 16. Single rank.
 */

#include "../simulateqcd.h"
#include "../experimental/mdwf/MDWFResidualMass.h"
#include "../gauge/gaugeAction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class MDWFResidualMassParameters : public LatticeParameters {
public:
    Parameter<double> M5;
    Parameter<double> mf;
    Parameter<double> b5;
    Parameter<double> csw;
    Parameter<int> ls;
    Parameter<int> antiperiodic_t;
    Parameter<int, 4> source;
    Parameter<int, 2> components;
    Parameter<std::string> direction;
    Parameter<int> plateau_min;
    Parameter<int> plateau_max;
    Parameter<int> max_iter;
    Parameter<double> precision;
    Parameter<int> seed;
    Parameter<std::string> start;
    Parameter<std::string> output_name;
    Parameter<std::string> reference_file;
    Parameter<double> reference_tolerance;

    MDWFResidualMassParameters() {
        addDefault(M5, "M5", 1.8);
        addDefault(mf, "mf", 0.1);
        addDefault(b5, "b5", 1.5);
        addDefault(csw, "c_sw", 0.0);
        addDefault(ls, "ls", 8);
        addDefault(antiperiodic_t, "antiperiodic_t", 1);
        static const int defaultSource[4] = {0, 0, 0, 0};
        addDefault(source, "source", defaultSource);
        static const int defaultComponents[2] = {0, 11};
        addDefault(components, "components", defaultComponents);
        addDefault(direction, "direction", std::string("spatial"));
        addDefault(plateau_min, "plateau_min", 2);
        addDefault(plateau_max, "plateau_max", 4);
        addDefault(max_iter, "max_iter", 40000);
        addDefault(precision, "precision", 1e-10);
        addDefault(seed, "seed", 20261007);
        addDefault(start, "start", std::string("random"));
        addDefault(output_name, "output_name", std::string("mdwfResidualMass.dat"));
        addDefault(reference_file, "reference_file", std::string(""));
        addDefault(reference_tolerance, "reference_tolerance", 1e-4);
    }
};

std::string mdwfResidualMassComponentPath(const std::string &base, int a) {
    std::ostringstream path;
    path << base << ".c" << std::setw(2) << std::setfill('0') << a;
    return path.str();
}

// Reads a finished component; false if the file does not exist. A file written with other parameters is fatal.
bool mdwfReadResidualMassComponent(const std::string &path, const std::string &key, int a,
                                   const std::array<size_t, 4> &extents, MDWFResidualMassComponent &c) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    std::getline(in, line);
    if (line != key) {
        throw std::runtime_error(stdLogger.fatal("MDWF m_res: ", path, " was written with different parameters:\n  ",
                                                 line, "\nexpected\n  ", key));
    }
    int fileA = -1;
    in >> fileA >> c.iterations >> c.residue >> c.seconds;
    for (int mu = 0; mu < 4; mu++) {
        int fileMu = -1;
        size_t fileL = 0;
        in >> fileMu >> fileL;
        if (fileMu != mu || fileL != extents[mu]) {
            in.setstate(std::ios::failbit);
            break;
        }
        c.pp[mu].assign(extents[mu], 0.0);
        c.j5q[mu].assign(extents[mu], 0.0);
        for (size_t n = 0; n < extents[mu]; n++) {
            size_t fileN = 0;
            in >> fileN >> c.pp[mu].at(n) >> c.j5q[mu].at(n);
            if (fileN != n) {
                in.setstate(std::ios::failbit);
            }
        }
    }
    if (!in || fileA != a) {
        throw std::runtime_error(stdLogger.fatal("MDWF m_res: ", path, " is incomplete or corrupt"));
    }
    return true;
}

// Written to a temporary file and renamed, so a killed job never leaves a partial component behind.
// Format: key line; "a iterations residue seconds"; per direction mu: "mu L", then L lines "n C_PP C_J5q".
void mdwfWriteResidualMassComponent(const std::string &path, const std::string &key, int a,
                                    const MDWFResidualMassComponent &c) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        out << key << "\n" << std::setprecision(17);
        out << a << " " << c.iterations << " " << c.residue << " " << c.seconds << "\n";
        for (int mu = 0; mu < 4; mu++) {
            out << mu << " " << c.pp[mu].size() << "\n";
            for (size_t n = 0; n < c.pp[mu].size(); n++) {
                out << n << " " << c.pp[mu][n] << " " << c.j5q[mu][n] << "\n";
            }
        }
        out.flush();
        if (!out) {
            throw std::runtime_error(stdLogger.fatal("MDWF m_res cannot write ", tmp));
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        throw std::runtime_error(stdLogger.fatal("MDWF m_res cannot rename ", tmp, " to ", path));
    }
}

// Compares the raw per-slice correlators with a reference file (lines "mu n C_PP C_J5q", '#' comments).
// True if every slice of every direction is present and C_PP, C_J5q and m_res agree to the relative tolerance.
bool mdwfCompareResidualMassReference(const std::string &path, const MDWFResidualMassCorrelators &c,
                                      double tolerance) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(stdLogger.fatal("MDWF m_res cannot open reference file ", path));
    }
    std::array<double, 4> maxPP{}, maxJ5q{}, maxRatio{}, sumPPRef{}, sumJ5qRef{};
    std::array<size_t, 4> count{};
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream row(line);
        int mu = -1;
        size_t n = 0;
        double pp = 0.0, j5q = 0.0;
        if (!(row >> mu >> n >> pp >> j5q) || mu < 0 || mu > 3 || n >= c.pp[mu].size()) {
            throw std::runtime_error(stdLogger.fatal("MDWF m_res reference ", path, ": bad line '", line, "'"));
        }
        maxPP[mu] = std::max(maxPP[mu], std::abs(c.pp[mu][n] - pp) / std::abs(pp));
        maxJ5q[mu] = std::max(maxJ5q[mu], std::abs(c.j5q[mu][n] - j5q) / std::abs(j5q));
        maxRatio[mu] = std::max(maxRatio[mu], std::abs((c.j5q[mu][n] / c.pp[mu][n]) / (j5q / pp) - 1.0));
        sumPPRef[mu] += pp;
        sumJ5qRef[mu] += j5q;
        count[mu]++;
    }
    bool passed = true;
    for (int mu = 0; mu < 4; mu++) {
        double sumPP = 0.0, sumJ5q = 0.0;
        for (size_t n = 0; n < c.pp[mu].size(); n++) {
            sumPP += c.pp[mu][n];
            sumJ5q += c.j5q[mu][n];
        }
        const bool complete = count[mu] == c.pp[mu].size();
        const bool directionPassed = complete && maxPP[mu] <= tolerance && maxJ5q[mu] <= tolerance
                                     && maxRatio[mu] <= tolerance;
        rootLogger.info("MDWF m_res reference ", "xyzt"[mu], ": ", count[mu], " of ", c.pp[mu].size(),
                        " slices, max relative difference C_PP ", maxPP[mu], ", C_J5q ", maxJ5q[mu], ", m_res ",
                        maxRatio[mu], "; volume sums C_PP ", sumPP / sumPPRef[mu] - 1.0, ", C_J5q ",
                        sumJ5q / sumJ5qRef[mu] - 1.0, " (relative); passed = ", directionPassed);
        passed = passed && directionPassed;
    }
    return passed;
}

template<size_t Ls>
void runMDWFResidualMass(CommunicationBase &commBase, MDWFResidualMassParameters &param) {
    const size_t HaloDepth = 2;
    typedef GIndexer<All, HaloDepth> GInd;
    using Gauge = Gaugefield<double, true, HaloDepth, R18>;

    const LatticeData lat = GInd::getLatData();
    const std::array<size_t, 4> extents = mdwfLatticeExtents(lat);
    const std::string directionNames = "xyzt";

    // Directions summed for the analysis.
    std::vector<int> directions;
    if (param.direction() == "spatial") {
        if (lat.lx != lat.ly || lat.lx != lat.lz) {
            throw std::runtime_error(stdLogger.fatal("MDWF m_res: direction = spatial needs equal spatial extents"));
        }
        directions = {0, 1, 2};
    } else if (param.direction().size() == 1 && directionNames.find(param.direction()) != std::string::npos) {
        directions = {static_cast<int>(directionNames.find(param.direction()))};
    } else {
        throw std::runtime_error(stdLogger.fatal("MDWF m_res: direction must be x, y, z, t or spatial, got ",
                                                 param.direction()));
    }
    const int length = static_cast<int>(extents[directions.front()]);
    if (param.plateau_min() < 0 || param.plateau_max() > length / 2 || param.plateau_min() > param.plateau_max()) {
        throw std::runtime_error(stdLogger.fatal("MDWF m_res: plateau must lie in [0, ", length / 2, "] along ",
                                                 param.direction()));
    }

    Gauge gauge(commBase, "MDWF_mres_gauge");
    Gauge fermionGauge(commBase, "MDWF_mres_fgauge");
    if (param.GaugefileName.isSet()) {
        rootLogger.info("MDWF m_res: reading NERSC configuration ", param.GaugefileName());
        gauge.readconf_nersc(param.GaugefileName());
    } else if (param.start() == "unit") {
        rootLogger.info("MDWF m_res: unit gauge field (test)");
        gauge.one();
    } else {
        rootLogger.info("MDWF m_res: random gauge field (test), seed ", param.seed());
        grnd_state<false> h_rand;
        grnd_state<true> d_rand;
        h_rand.make_rng_state(param.seed());
        d_rand = h_rand;
        gauge.random(d_rand.state);
    }
    gauge.updateAll();
    GaugeAction<double, true, HaloDepth, R18> action(gauge);
    const double plaquette = static_cast<double>(action.plaquette());
    mdwfFermionGaugeField<HaloDepth>(fermionGauge, gauge, param.antiperiodic_t() != 0);

    const std::array<int, 4> src = {param.source[0], param.source[1], param.source[2], param.source[3]};
    rootLogger.info("MDWF m_res: ", lat.globLX, "x", lat.globLY, "x", lat.globLZ, "x", lat.globLT, ", Ls = ", Ls,
                    ", M5 = ", param.M5(), ", b5 = ", param.b5(), ", c5 = ", param.b5() - 1.0, ", mf = ", param.mf(),
                    ", c_sw = ", param.csw(), ", ", param.antiperiodic_t() ? "antiperiodic" : "periodic",
                    " fermion BCs in time, source (", src[0], ",", src[1], ",", src[2], ",", src[3],
                    "), plaquette = ", plaquette, ", solver precision = ", param.precision(),
                    ", analysis direction ", param.direction());

    const int first = param.components[0], last = param.components[1];
    if (first < 0 || last > 11 || first > last) {
        throw std::runtime_error(stdLogger.fatal("MDWF m_res: components must satisfy 0 <= first <= last <= 11, got ",
                                                 first, " ", last));
    }
    const std::string outputPath = param.measurements_dir() + "/" + param.output_name();
    std::ostringstream keyStream;
    keyStream << std::setprecision(17) << "# mdwfResidualMass component v2: lattice " << lat.globLX << "x"
              << lat.globLY << "x" << lat.globLZ << "x" << lat.globLT << " Ls " << Ls << " M5 " << param.M5()
              << " mf " << param.mf() << " b5 " << param.b5() << " c_sw " << param.csw() << " bc_t "
              << (param.antiperiodic_t() ? "antiperiodic" : "periodic") << " source " << src[0] << " " << src[1]
              << " " << src[2] << " " << src[3] << " gauge ";
    if (param.GaugefileName.isSet()) {
        keyStream << param.GaugefileName();
    } else {
        keyStream << param.start() << " seed " << param.seed();
    }
    const std::string key = keyStream.str();

    std::vector<MDWFResidualMassComponent> components(12);
    std::vector<bool> present(12, false);
    for (int a = 0; a < 12; a++) {
        present[a] = mdwfReadResidualMassComponent(mdwfResidualMassComponentPath(outputPath, a), key, a, extents,
                                                   components[a]);
    }
    MDWFResidualMassMeasurement<HaloDepth, Ls> measurement(fermionGauge, param.M5(), param.mf(), param.b5(),
                                                           param.csw(), "MDWF_mres_meas");
    for (int a = first; a <= last; a++) {
        const std::string path = mdwfResidualMassComponentPath(outputPath, a);
        if (present[a]) {
            rootLogger.info("MDWF m_res: source component ", a, " already done (", path, "), skipped");
            continue;
        }
        // Another job on a disjoint range may have finished it meanwhile.
        if (mdwfReadResidualMassComponent(path, key, a, extents, components[a])) {
            present[a] = true;
            continue;
        }
        components[a] = measurement.measureComponent(a, src[0], src[1], src[2], src[3], param.max_iter(),
                                                     param.precision());
        if (commBase.IamRoot()) {
            mdwfWriteResidualMassComponent(path, key, a, components[a]);
        }
        present[a] = true;
    }
    std::ostringstream missing;
    int nMissing = 0;
    for (int a = 0; a < 12; a++) {
        if (!present[a]) {
            present[a] = mdwfReadResidualMassComponent(mdwfResidualMassComponentPath(outputPath, a), key, a, extents,
                                                       components[a]);
        }
        if (!present[a]) {
            missing << " " << a;
            nMissing++;
        }
    }
    if (nMissing > 0) {
        rootLogger.info("MDWF m_res: ", 12 - nMissing, " of 12 source components done, missing:", missing.str(),
                        "; rerun (or run the missing components) to assemble ", outputPath);
        return;
    }

    MDWFResidualMassCorrelators c{{}, {}, 0, 0.0};
    mdwfResizeDirectionalCorrelators(c.pp, lat);
    mdwfResizeDirectionalCorrelators(c.j5q, lat);
    double seconds = 0.0;
    for (const MDWFResidualMassComponent &component : components) {
        c.iterations += component.iterations;
        c.maxResidue = std::max(c.maxResidue, component.residue);
        seconds += component.seconds;
        for (int mu = 0; mu < 4; mu++) {
            for (size_t n = 0; n < extents[mu]; n++) {
                c.pp[mu].at(n) += component.pp[mu].at(n);
                c.j5q[mu].at(n) += component.j5q[mu].at(n);
            }
        }
    }

    // Distance from the source along the analysis direction(s), folded, summed over the directions.
    std::vector<double> pp(length / 2 + 1, 0.0), j5q(length / 2 + 1, 0.0);
    for (const int mu : directions) {
        auto at = [&](const std::vector<double> &v, int d) {
            return v.at(static_cast<size_t>(((src[mu] + d) % length + length) % length));
        };
        for (int d = 0; d <= length / 2; d++) {
            pp.at(d) += 0.5 * (at(c.pp[mu], d) + at(c.pp[mu], length - d));
            j5q.at(d) += 0.5 * (at(c.j5q[mu], d) + at(c.j5q[mu], length - d));
        }
    }

    std::ofstream out;
    if (commBase.IamRoot()) {
        out.open(outputPath, std::ios::out | std::ios::trunc);
        if (!out) {
            throw std::runtime_error(stdLogger.fatal("MDWF m_res cannot open ", outputPath));
        }
        out << "# direction " << param.direction() << " (";
        for (const int mu : directions) {
            out << directionNames[mu];
        }
        out << "), length " << length << ", plaquette " << std::setprecision(12) << plaquette << std::endl;
        out << "# d  C_PP(d)  C_J5q(d)  m_res(d) = C_J5q / C_PP   (folded, distance from source)" << std::endl;
    }
    double sumPP = 0.0, sumJ5q = 0.0, sumRatio = 0.0;
    int nPlateau = 0;
    for (int d = 0; d <= length / 2; d++) {
        const double ratio = j5q.at(d) / pp.at(d);
        rootLogger.info("MDWF m_res: ", param.direction(), " d = ", d, ", C_PP = ", pp.at(d), ", C_J5q = ", j5q.at(d),
                        ", m_res(d) = ", ratio);
        if (commBase.IamRoot()) {
            out << d << " " << pp.at(d) << " " << j5q.at(d) << " " << ratio << std::endl;
        }
        if (d >= param.plateau_min() && d <= param.plateau_max()) {
            sumPP += pp.at(d);
            sumJ5q += j5q.at(d);
            sumRatio += ratio;
            nPlateau++;
        }
    }
    rootLogger.info("MDWF m_res: ", param.direction(), " plateau d in [", param.plateau_min(), ", ",
                    param.plateau_max(), "]: m_res = ", sumJ5q / sumPP, " (ratio of sums), ", sumRatio / nPlateau,
                    " (mean of ratios); 12 solves, ", c.iterations, " CG iterations, max residue ", c.maxResidue,
                    ", ", seconds, " s of solves");

    if (!param.reference_file().empty()) {
        rootLogger.info("MDWF m_res: comparing with reference ", param.reference_file(), ", relative tolerance ",
                        param.reference_tolerance());
        if (!mdwfCompareResidualMassReference(param.reference_file(), c, param.reference_tolerance())) {
            throw std::runtime_error(stdLogger.fatal("MDWF m_res reference comparison failed"));
        }
        rootLogger.info("MDWF m_res reference comparison passed");
    }
}

int main(int argc, char **argv) {
    try {
        stdLogger.setVerbosity(INFO);

        MDWFResidualMassParameters param;
        CommunicationBase commBase(&argc, &argv, true);
        param.readfile(commBase, "../parameter/tests/mdwfResidualMass.param", argc, argv);
        commBase.init(param.nodeDim());

        const int HaloDepth = 2;
        initIndexer(HaloDepth, param, commBase);

        if (param.ls() == 8) {
            runMDWFResidualMass<8>(commBase, param);
        } else if (param.ls() == 16) {
            runMDWFResidualMass<16>(commBase, param);
        } else {
            throw std::runtime_error(stdLogger.fatal("MDWF m_res supports ls = 8 or 16, got ", param.ls()));
        }
        return 0;
    }
    catch (const std::runtime_error &error) {
        rootLogger.error("There has been a runtime error!");
        rootLogger.error(error.what());
        return -1;
    }
}
