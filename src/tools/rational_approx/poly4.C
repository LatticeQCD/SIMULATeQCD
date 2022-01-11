/*
  Code for making rational functions for MILC ks_imp_rhmc code
  Version 1   5/10/06 Doug Toussaint
  Based on "test" code by Mike Clark, using Mike Clarks alg_remez.C
 
  Last modified by Qing Yuan
  Email: qyuan@mails.ccnu.edu.cn
  Usage: ./poly4 < [params_in_file] > [params_rhmc_file]

  Calculate three rational approximations, for gaussian random heatbath
  update, and fermion action computation, molecular dynamics evolution.

  Heatbath ("GR")
	PROD  X^(+nf_i/8) where X = M^dagger M
  Fermion Acttion:
	PROD  X^(-nf_i/8) where X = M^dagger M
  Molecular dynamics:
	PROD  X^(-nf_i/4) where X = M^dagger M
 
2018/1/30 Qing Yuan

I modified the subroutine poly4.C and creats a parameter file in 
a format suitable for input to the BielefeldGPUCode.

The format of input file:

  number of pseudofermions
  y1 y2 mh mq order1 order2 lambda_low lambda_high precision  (If no hasenbusch, y2 = mh = 0)
  :
  :

For example:
In Bielefeld GPU code we want to calculate the following determinent: 

        #--------------------------------------------#
        #    [detMl/detMs]^(1/2) * det(Ms)^(3/4)     #
        #--------------------------------------------#

The convention in this case is x =(D^+)*D + ml^2 for psf1, x = (D^+)*D + ms^2 for psf2.

So you want to obtain rational coefficients for this following functions:

 x^(3/8)
 x^(-3/8)                                   ( for psf2 )
 x^(-3/4)

 x^(1/4) * ( x + ms^2 - ml^2 )^(-1/4)
 x^(-1/4) * ( x + ms^2 - ml^2 )^(1/4)       ( for psf1 )
 x^(-1/2) * ( x + ms^2 - ml^2 )^(1/2)

where ms = 0.0591, ml = 0.00591
then we get:

  lambda_low = ms^2 = 0.00349281 for psf2
  lambda_low = ml^2 = 0.0000349281 for psf1
  lambda_high = 5 for psf1 and psf2
  order1 = 14, order2 = 12 for psf2 and psf1
  y1 = 3, y2 = 0 for psf2
  y1 = 2, y2 = -2, for psf1

So the input file is:

#---input filename--#
#					#
#   2				#
#   3				#
#	0				#
#	0				#
#	0.0591			#
#	14				#
#	12				#
#	0.00349281		#
#	5.0				#
#	50              #
#   2				#
#	-2				#
#	0.0591			#
#	0.00591			#
#	14				#
#	12				#
#	0.0000349281	#
#	5				#
#	160      		#
#					#
#----end of file----#

Call it like this.

./poly4 < "inputfilenme" > "outputfilename"

This input file generates coefficients for these following functions:

x^(3/8)
x^(-3/8)      ( for psf2 )
x^(-3/4)

x^(1/4) * ( x + 0.0588037575330012^2 )^(-1/4)
x^(-1/4) * ( x + 0.0588037575330012^2 )^(1/4)       ( for psf1 )
x^(-1/2) * ( x + 0.0588037575330012^2 )^(1/2)

*/



#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include"alg_remez.h"
#define naik_term_epsilon 0.0
using namespace std;
int work( double y1, double z1, double m1, double y2, double z2, double m2, double y3, double z3, double m3, double y4, double z4, double m4, int order, double lambda_low, double lambda_high, int precision, const char *tag1, const char *tag2 ,int index,int interval);

int main (int argc, char* argv[]) {
  double m1=0.0; // The mass for flavor 1
  double m2=0.0;
  double y3=0.0; // The numerator of the exponent for flavor 3
  double m3=0.0; // The mass for flavor 3
  double y4=0.0; // The numerator of the exponent for flavor 4
  double m4=0.0; // The mass for flavor 4
  int iphi,n_pseudo;
  int index;
//help information:
if(argc != 2)
{
fprintf(stderr,"usage:\n");
fprintf(stderr,"./poly4 input.dat\n");
fprintf(stderr,"Please read the README to get the format of the input file\n");
exit(0);
}

// Read the number of pseudofermion fields
	int i=0;
	ifstream fin;
	string str[100];
	fin.open(argv[1]);
	while( !fin.eof())
		{
			getline(fin,str[i]);
			if(!str[i].empty())
			{
				i++;
			}
		}
	
	fin.close();

// convert string to int, double
	n_pseudo = atoi(str[0].c_str());   //get the number of the fermions
	
	if(i!=(n_pseudo*9+1)){
	fprintf(stderr,"ERROR: The first line in your input file should be the number of pseudofermions, it is %d, So you need %d parameters, But you only give %d parameters, please check it!\n", n_pseudo, (n_pseudo*9+1), i);
	exit(0);}

double y1[n_pseudo];
double y2[n_pseudo];
double mh[n_pseudo];
double mq[n_pseudo];
int order1[n_pseudo];
int order2[n_pseudo];
int precision[n_pseudo];
double lambda_low[n_pseudo];
double lambda_high[n_pseudo];

for(iphi = 0; iphi<n_pseudo; iphi++)   // read all the parameters
{
	y1[iphi] = atof(str[iphi*9+1].c_str());
	y2[iphi] = atof(str[iphi*9+2].c_str());
	mh[iphi] = atof(str[iphi*9+3].c_str());
	mq[iphi] = atof(str[iphi*9+4].c_str());
	order1[iphi] = atoi(str[iphi*9+5].c_str());
	order2[iphi] = atoi(str[iphi*9+6].c_str());
	lambda_low[iphi] = atof(str[iphi*9+7].c_str());
	lambda_high[iphi] = atof(str[iphi*9+8].c_str());
	precision[iphi] = atoi(str[iphi*9+9].c_str());
}
//It is special when you have two pseudofermions
if(n_pseudo==2&&(y2[0]!=0||mh[0]!=0))
     {
	fprintf(stderr,"ERROR: According to the convention of Bielefeld GPU code with two PSFs, the PSF without hasenbusch should be set up prior to the other one.\n");
	fprintf(stderr,"Please refer to in.sample/in.sample.bielefeldconvention_2psf\n");
	exit(0);	
     }
//hasenbusch quark mass should be larger than light quark mass
for(iphi=0; iphi<n_pseudo; iphi++)
{
	if(y2[iphi]!=0&&(mh[iphi] < mq[iphi]))
    {
	   fprintf(stderr,"ERROR! In case of Hasenbusch mass preconditioning the hasenbusch quark mass( 3rd parameter from left) should be larger than light quark mass (4th parameter from left).\n");
		exit(0);
    }
//lambda_low must smaller than mass**2
    if((10000*lambda_low[iphi])>(10000*pow(mq[iphi],2)))
	{  
	   fprintf(stderr,"ERROR! lambda_low( 3rd parameter from right) must be smaller (or equal) than square of mass(4th parameter from left).\n");
	   fprintf(stderr,"Please refer to README.\n");
	   exit(0);
	}
//order1 should be larger than order2    
    if(order1[iphi]<order2[iphi])
	{
	   fprintf(stderr,"ERROR! order1( 5th parameter from left) should be larger than order2( 6th parameter from left).\n");
	   fprintf(stderr,"Please refer to README.\n");
	   exit(0);
	}

}

 // printf("n_pseudo %d\n\n",n_pseudo);
 printf("#=====================================================#\n");
printf("# FILE:     rational.hzt                              #\n");
printf("# COMMENTS: Contains the constants for all the        #\n");
printf("#           rational approximations required for the  #\n");
printf("#           RHMC + Hasenbusch                         #\n");
printf("#           ( all the coefficients were obtained      #\n");
printf("#             using MILC code )                        #\n");
printf("#                                                     #\n");
printf("#=====================================================#\n");
printf("#-----------------------------------------------------------------------------------\n");
printf("# Coefficients for rational approximations:\n");
printf("#   r(x) = r_const + r_num[0]/(r_den[0] + x) + ... \n");
printf("#          + r_num[r_order-1]/(r_den[r_order - 1] + x)\n");
printf("#\n");
printf("#-----------------------------------------------------------------------------------\n");
printf("\n");
printf("\n");

  for(iphi=0; iphi<n_pseudo; iphi++){
    index=iphi+1;
    // Set the exponents and masses
    fprintf(stderr,"For pseudofermion %d\n",iphi);

   m2=sqrt(fabs(mh[iphi]*mh[iphi] - mq[iphi]*mq[iphi]));
   int interval;
   interval=order1[iphi]-order2[iphi];
    work( y1[iphi],8.0,m1, y2[iphi],8.0,m2, y3,8.0,m3, y4,8.0,m4, order1[iphi], lambda_low[iphi], lambda_high[iphi], precision[iphi], "GR","FA",index,interval);
    // For the MD term we need only the inverse
    work( y1[iphi],4.0,m1,  y2[iphi],4.0,m2,  y3,4.0,m3,  y4,4.0,m4, order2[iphi], 
	  lambda_low[iphi], lambda_high[iphi], precision[iphi], "OMIT", "MD",index,interval );
    // The random source term takes the function and action term,
    // the inverse
  }
printf("#=================== End of File =====================#\n");
}

void print_check_ratfunc(double y1, double y2, double y3, double y4,
			 double z1, double z2, double z3, double z4,
			 double m1, double m2, double m3, double m4,
			 double lambda_low, int order,
			 double norm,  double *res, double *pole, const char *tag,int index,int interval){

  if( strcmp(tag,"OMIT") == 0 )return;

if(strcmp(tag,"GR")==0)
 {
 printf("#-----------------------------------------------------------------------------------\n");
 printf("#  r_inv_%df\n", index);
 printf("#-----------------------------------------------------------------------------------\n");
 printf("\n");
 printf("# real r_inv_%df_const, r_inv_%df_num[r_high_order_2f], r_inv_%df_den[r_high_order_2f]\n",index,index,index);

  printf("r_inv_%df_const = %18.16e\n",index, norm);
  for (int i = 0; i < order; i++)
  printf("r_inv_%df_num[%d] = %18.16e\n",index,i,res[i]);
  printf("\n");
  for (int i = 0; i < order; i++) 
  {
   printf("r_inv_%df_den[%d] = %18.16e\n",index,i,pole[i]);
  } 
 }
  
  if(strcmp(tag,"FA")==0)
  {
    printf("#-----------------------------------------------------------------------------------\n");
    printf("#  r_inv_%df\n", index);
    printf("#-----------------------------------------------------------------------------------\n");
    printf("\n");
    printf("# real r_%df_const, r_%df_num[r_high_order_2f], r_%df_den[r_high_order_2f]\n",index,index,index);

    printf("r_%df_const = %18.16e\n",index,norm);
    for(int i = 0; i<order;i++)
    printf("r_%df_num[%d] = %18.16e\n",index,i, res[i]);
    printf("\n");
    for(int i=0;i<order;i++)
    {printf("r_%df_den[%d] = %18.16e\n",index,i,pole[i]);}
  }
if(strcmp(tag,"MD")==0)
{
   printf("#-----------------------------------------------------------------------------------\n");
   printf("#  r_bar_%df\n", index);
   printf("#-----------------------------------------------------------------------------------\n");
   printf("\n");
   printf("# real r_bar_%df_const, r_bar_%df_num[r_high_order_2f], r_bar_%df_den[r_high_order_2f]\n",index,index,index);

   printf("r_bar_%df_const = %18.16e\n",index,norm);
   for(int i=0;i<order;i++)
   printf("r_bar_%df_num[%d] = %18.16e\n",index,i,res[i]);
   for(int i=order;i<(interval+order);i++)
   printf("r_bar_%df_num[%d] = 0\n",index,i);
   printf("\n");
   for(int i=0;i<order;i++)
   printf("r_bar_%df_den[%d] = %18.16e\n",index,i,pole[i]);
   for(int i=order;i<(order+interval);i++)
   printf("r_bar_%df_den[%d] = 0\n",index,i);
}


  printf("\n");

  // Check - compute the function at the low endpoint of the interval
  double x,sum;
  int ii;
  for ( x = lambda_low, sum=norm, ii = 0; ii < order; ii++) {
    sum += res[ii]/(x+pole[ii]);
  }
  

  double f1 = pow(x+1*m1*m1,((double)y1)/z1);    
  double f2 = pow(x+1*m2*m2,((double)y2)/z2);    
  double f3 = pow(x+1*m3*m3,((double)y3)/z3);   
  double f4 = pow(x+1*m4*m4,((double)y4)/z4);  

  
   printf("# CHECK: f(%e) = %e = %e?\n",x,sum,f1*f2*f3*f4);
}



int work(double y1, double z1, double m1,  
	 double y2, double z2, double m2,  
	 double y3, double z3, double m3,
	 double y4, double z4, double m4, 
	 int order, 
	 double lambda_low,  double lambda_high, 
	 int precision,
 	 const char *tag1, const char *tag2 ,int index,int interval){
  // The error from the approximation (the relative error is minimised
  // - if another error minimisation is requried, then line 398 in
  // alg_remez.C is where to change it)
  //double error;
  
  double *res = new double[order];
  double *pole = new double[order];
  
  // The partial fraction expansion takes the form 
  // r(x) = norm + sum_{k=1}^{n} res[k] / (x + pole[k])
  double norm;

 // double bulk = exp(0.5*(log(lambda_low)+log(lambda_high)));

  // Instantiate the Remez class,
  AlgRemez remez(lambda_low,lambda_high,precision);

  // Generate the required approximation
  fprintf(stderr,
	  "Generating a (%d,%d) rational function using %d digit precision.\n",
	  order,order,precision);
   remez.generateApprox(order,order,y1,z1,m1,y2,z2,m2,
			       y3,z3,m3,y4,z4,m4);

  // Find the partial fraction expansion of the approximation 
  // to the function x^{y/z} (this only works currently for 
  // the special case that n = d)
  remez.getPFE(res,pole,&norm);
  
  print_check_ratfunc(y1,y2,y3,y4,z1,z2,z3,z4,m1,m2,m3,m4,
		      lambda_low,order,norm,res,pole,tag1,index,interval);

  // Find pfe of the inverse function
  remez.getIPFE(res,pole,&norm);

  print_check_ratfunc(-y1,-y2,-y3,-y4,z1,z2,z3,z4,m1,m2,m3,m4,
		      lambda_low,order,norm,res,pole,tag2,index,interval);

  FILE *error_file = fopen("error.dat", "w");
  for (double x=lambda_low; x<lambda_high; x*=1.01) {
    double f = remez.evaluateFunc(x);
    double r = remez.evaluateApprox(x);
    fprintf(error_file,"%e %e\n", x,  (r - f)/f);
  }
  fclose(error_file);

  delete[] res;
  delete[] pole;
return 0;
}
