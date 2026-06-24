//#include <cstdlib>
#include <iostream>
//#include <vector>
//#include <algorithm>
#include <string>

#include "mt.hpp"

using namespace std;

int * init(string init_conf,unsigned int Nparts,
	   int st, int Nstates, unsigned int vol){
  
  MersenneTwister mt;
  
  int *is = new int[vol];
  for(unsigned int i=0; i<vol; i++) is[i]=0;
  
  if(init_conf.compare("equal") == 0){
    for(unsigned int i=0; i < Nparts ; i++) is[i]=st;
    
  } else if(init_conf.compare("random") == 0){
    
    for(unsigned int i=0; i<vol; i++) is[i]=0;
    
    mt.init_genrand(time(NULL));

    unsigned int i=0;
    while(i<Nparts){
      
      unsigned int j=(int)(mt.random()*(vol));
      cout << j << endl ;
      if( is[j] == 0){ 
	is[j]=(int)(mt.random()*(Nstates)+1);
	i++;
      }
    }


  } else if(init_conf.compare("custom") == 0){
    
    /*
    char fname[100];
    inpfile=

    sprintf(fname, "input_i_0_L_8");
    string str(fname), STR;
    infile.open(fname);
    int i=0;
    int a , b; 
    if(infile.is_open()){
      while( infile >> a >> b ){
	i=a; 
	s[i]=b;
	cout << i << "  " << s[i] << endl ;
      }
    }
    */
    
    //cout << "dawdawdaw" << endl;
  }
  return is;
}
