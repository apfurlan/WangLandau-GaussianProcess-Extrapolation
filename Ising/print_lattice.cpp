#include <iostream>

using namespace std;

void print_lattice(int L,int vol,int ndim,int ncoord,
		   int * s){

  int i;
  
  if(ndim == 2 && ncoord == 4){
    for(i=0;i<vol; i++){
      cout << s[i] << " " ;
      if((i+1) % L == 0 && i > 0) cout <<  endl ;
    }
    
  } else if(ndim == 2 && ncoord == 6){
    
    for(i=0;i<vol; i++){
      cout << s[i] << " " ;
      if((i+1) % L == 0 && i > 0) cout <<  endl ;
      if((i+1) % L == 0 && (i+1)/L % 2 != 0 ) cout <<" "  ;
    }
    
  }
  

}

