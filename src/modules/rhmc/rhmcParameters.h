#ifndef RHMC_PARAM
#define RHMC_PARAM

#include "../../base/latticeParameters.h"
#include "../../base/IO/parameterManagement.h"
#include <math.h>
#include "../../base/math/floatComparison.h"

//! Class with parameters for the RHMC update, additionally to base parameters
class RhmcParameters: public LatticeParameters {
public:

    Parameter<int> no_pf;
    Parameter<int> no_md;
    Parameter<int> no_step_sf;
    Parameter<int> no_sw;
    Parameter <double> mu_f;
    Parameter<double> step_size;
    Parameter<int> integrator;
    Parameter<double> residue;
    Parameter<double> residue_force;
    Parameter<int> cgMax;
    Parameter<double> residue_meas;
    Parameter<int> cgMax_meas;
    Parameter<bool> cgMixedPrec;
    Parameter<double> cgMixedPrec_delta;
    Parameter<bool> always_acc;
    Parameter<int> no_updates;
    Parameter<int> write_every;
    Parameter<std::string> rat_file;
    Parameter <double> m_ud;
    Parameter <double> m_s;
    Parameter <std::string> rand_file;
    Parameter <std::string> gauge_file; // This is just the prefix for the gauge file name
    Parameter <int> seed;
    Parameter <bool> load_rand;
    Parameter <int> load_conf;

    RhmcParameters(){
        addDefault(no_pf, "no_pf", 1);
        add(no_md, "no_md");
        add(no_step_sf, "no_step_sf");
        add(no_sw, "no_sw");
        add(step_size, "step_size");
        addDefault(mu_f, "mu0", 0.0);
        addDefault(integrator, "integrator", 0);
        addDefault(residue, "residue", 1e-12);
        addDefault(residue_force, "residue_force", 1e-7);
        addDefault(cgMax, "cgMax", 10000);
        addDefault(residue_meas, "residue_meas", 1e-12);
        addDefault(cgMax_meas, "cgMax_meas", 20000);
        addDefault(always_acc, "always_acc", false);
        addDefault(cgMixedPrec, "cgMixedPrec", false);
        addDefault(cgMixedPrec_delta, "cgMixedPrec_delta",0.1);
        add(no_updates, "no_updates");
        addDefault(write_every, "write_every", 1);
        add(rat_file, "rat_file");
        addOptional(m_ud, "mass_ud");
        addOptional(m_s, "mass_s");
        addOptional(rand_file, "rand_file");
        addOptional(gauge_file, "gauge_file");
        addOptional(seed, "seed");
        addOptional(load_rand, "rand_flag");
        addDefault(load_conf, "load_conf", 0);
    };
};

class RationalCoeff : virtual public ParameterList {

private:

double get_exp(double r, double x)
{
    return log(r)/log(x);
}

public:

    // RationalCoeff for (Ms^+ * Ms)^(3/8) used in pseudo-fermion heatbath, higher order
    Parameter<double> r_inv_sf_const;
    DynamicParameter<double> r_inv_sf_num;
    DynamicParameter<double> r_inv_sf_den;

    // RationalCoeff for (Ms^+ * Ms)^(-3/8) used in action, higher order
    Parameter<double> r_sf_const;
    DynamicParameter<double> r_sf_num;
    DynamicParameter<double> r_sf_den;

    // RationalCoeff for (Ms^+ * Ms)^(-3/4) used in force, lower order
    Parameter<double> r_bar_sf_const;
    DynamicParameter<double> r_bar_sf_num;
    DynamicParameter<double> r_bar_sf_den;

    // RationalCoeff for (Mud^+ * Mud)^(1/4) used in pseudo-fermion heatbath, higher order
    Parameter<double> r_inv_lf_const;
    DynamicParameter<double> r_inv_lf_num;
    DynamicParameter<double> r_inv_lf_den;

    // RationalCoeff for (Mud^+ * Mud)^(-1/4) used in action, higher order
    Parameter<double> r_lf_const;
    DynamicParameter<double> r_lf_num;
    DynamicParameter<double> r_lf_den;

    // RationalCoeff for (Mud^+ * Mud)^(-1/2) used in force, lower order
    Parameter<double> r_bar_lf_const;
    DynamicParameter<double> r_bar_lf_num;
    DynamicParameter<double> r_bar_lf_den;

    RationalCoeff(){

        add(r_inv_sf_const, "r_inv_1f_const");
        add(r_inv_sf_num, "r_inv_1f_num");
        add(r_inv_sf_den, "r_inv_1f_den");
        add(r_sf_const, "r_1f_const");
        add(r_sf_num, "r_1f_num");
        add(r_sf_den, "r_1f_den");
        add(r_bar_sf_const, "r_bar_1f_const");
        add(r_bar_sf_num, "r_bar_1f_num");
        add(r_bar_sf_den, "r_bar_1f_den");

        add(r_inv_lf_const, "r_inv_2f_const");
        add(r_inv_lf_num, "r_inv_2f_num");
        add(r_inv_lf_den, "r_inv_2f_den");
        add(r_lf_const, "r_2f_const");
        add(r_lf_num, "r_2f_num");
        add(r_lf_den, "r_2f_den");
        add(r_bar_lf_const, "r_bar_2f_const");
        add(r_bar_lf_num, "r_bar_2f_num");
        add(r_bar_lf_den, "r_bar_2f_den");

    };

    void check_rat(RhmcParameters param)
    {
        double x = 0.01;

        double z = 2e-5;

        double y = z/(z+ param.m_s()*param.m_s() - param.m_ud() * param.m_ud());

        double rs = r_sf_const();
        double rsinv = r_inv_sf_const();
        double rsbar = r_bar_sf_const();

        double rl = r_lf_const();
        double rlinv = r_inv_lf_const();
        double rlbar = r_bar_lf_const();

        bool tpo = false;
        bool error = false;

        for (size_t i = 0; i < r_sf_num.numberValues(); ++i)
        {
            rs += r_sf_num[i]/(x + r_sf_den[i]);
            rsinv += r_inv_sf_num[i]/(x + r_inv_sf_den[i]);

            rl += r_lf_num[i]/(z + r_lf_den[i]);
            rlinv += r_inv_lf_num[i]/(z + r_inv_lf_den[i]);
        }

        for (size_t i = 0; i < r_bar_sf_num.numberValues(); ++i)
        {
            rsbar += r_bar_sf_num[i]/(x + r_bar_sf_den[i]);
            rlbar += r_bar_lf_num[i]/(z + r_bar_lf_den[i]);
        }

        if (cmp_rel(get_exp(rs,x), -get_exp(rsinv,x), 0.0001 , 0.0001) &&
            cmp_rel(get_exp(rs,x), 0.5*get_exp(rsbar,x), 0.0001 , 0.0001) )
        {
            if(cmp_rel(get_exp(rs,x),-3.0/8.0, 0.0001 , 0.0001))
                tpo = true;
        } else {
            rootLogger.error("strange quark rational approximations are not consistent");
            error = true;
        }

        if (cmp_rel(get_exp(rl,y), -get_exp(rlinv,y), 0.0001 , 0.0001) &&
            cmp_rel(get_exp(rl,y), 0.5*get_exp(rlbar,y), 0.0001 , 0.0001) )
        {
            if(cmp_rel(get_exp(rl,y),-0.25, 0.0001 , 0.0001))
                tpo = tpo && true;
            else
                tpo = tpo && false;
        } else {
            rootLogger.error("light quark rational approximations are not consistent");
            rootLogger.error(get_exp(rl,x) ,  " " ,  - get_exp(rlinv,x) ,  " " ,  0.5*get_exp(rlbar,x));
            error = true;
        }

        if(error)
            throw std::runtime_error(stdLogger.fatal("There was atleast one error!"));

        if(tpo)
            rootLogger.info("You seem to be simulating 2+1f with std. Hasenbusch preconditioning!");
        else
            rootLogger.warn("Rational approximation unknown, but consistent!");
    }
};


#endif

