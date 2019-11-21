#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist.h"
#include <omp.h>

//#define M_PI 3.14159265358979323846
#define square(x) ((x)*(x))
// #define sgn(x) (x>0? 1 : (x<0?-1:0)) // sign function
#define sgn(x) (x>0? 1 :-1) // sign function
#define act(x) (x>0? x : 0)   //activation function
// #define act(x) 1/(1+exp(-x))   //activation function
// #define actp(x) exp(-x)/square(1+exp(-x)) //activation derivative
#define actp(x) (x>0? 1 : 0)  //activation derivative
#define max(x,y) (y>x? y: x)
#define min(x,y) (y<x? y: x)
#define firstsign(x,y,z) ((x !=0) ? sgn(x) : ((y != 0) ? sgn(y): sgn(z)))
FILE *fptre;
FILE *fptrt;
int m,mt,n,n1,d, d1;
int sw;
double targetred;
double one = 1;
double* xb;
double* x;
double* p;
double** X;
double** Z;
double** dZ;
int* Y;
double** W;
double** Wmin;
double** Wini;
double** dW;
double** ddW;
double** bW;
double* z;
double* bz;
double** Z;
double** M;
double** L;
double* a;
double* c;
double q;      // for the benefit of CGS
double omega;
double e;
int l1=0;
int mnist=1;
double iprod;
double tnorm;
double gnorm;



void resetzero(int l, double* v)
{ for(int i = 0; i< l; i++) v[i] = 0.0; }


int randint = 3;
int prime = 524287;
double myrandnumb()     // between 0 and 1
{ randint = (randint*7)%prime;
//    printf("\n randint %f", randint/(double)prime );
  return randint/(double)prime;}

int setsample()     // allocating memory of features,labels, layers and its derivatives
{  Y = (int*)malloc(m*sizeof(int));      // scalar label
   X = (double**)malloc(m*sizeof(double*));    // feature vectors
   Z = (double**)malloc(m*sizeof(double*));    // middle layer
   dZ = (double**)malloc(m*sizeof(double*));    // middle layer derivative
   *X = (double*)malloc(n1*m*sizeof(double));
   *Z = (double*)malloc(d1*m*sizeof(double));
   *dZ = (double*)malloc(d1*m*sizeof(double));
   load_mnist();
   for(int i =  0; i < m; i++ )
   { X[i] = *X + i*n1;
     Z[i] = *Z + i*d1;
     dZ[i] =*dZ + i*d1;
     for(int j=0; j<n; j++)X[i][j] = train_image[i][j];
     X[i][n] = 1;
     Y[i] = train_label[i];
   };
    return 0;
};

void printmatrix(int m, int n,double** A){
    for (int i=0;i<m;i++)
    {printf("\n row %i ",i);
        for(int j=0;j<n;j++)
            printf("  %f ", A[i][j]);
    };
}

double dot(int l, double* a, double* b)  //Inner product evaluation
{ double val;
    val = 0;
    for(int i=0;i<l;i++) val += a[i]*b[i]; //parallelization could be needed
    return val;
};

double norm(int l, double* a)  //Inner product evaluation or euclidean norm
{ double valmax=0;
  for(int i=0;i<l;i++) valmax = max(valmax, fabs(a[i]));
    if(valmax==0) return 0;
  double val=0;
  for(int i=0;i<l;i++) val += square(a[i]/valmax);
  return sqrt(val)*valmax;
};

void softmax(int l, double* b, double* e)         // b and e can coincide
{ double sum = 0;
  double maxi=0;
  for(int i=0; i<l; i++) maxi=max(maxi, b[i]);  
  for(int i=0; i<l; i++) sum += exp(b[i]-maxi);
  for(int i=0; i<l; i++) e[i]= exp(b[i]-maxi)/sum;  
}



void saxpy(int l, double* a, double q, double* b) // sum of a vector and a scaled one
{ for(int i=0;i<l;i++) b[i] += q*a[i];
};

void scale(int l, double* a, double q, double* b)// scalar product
{ for(int i=0;i<l;i++) b[i] = q*a[i];
};


int pred(double* x,double* z, int l) // prediction function, y last component of z.
{     //for (int i = 0; i< d; i++) z[i] = dot(n1,W[i],x);
int i,k, chunk;
chunk = 100;
#pragma omp parallel shared(x,W,z,chunk) private(i,k)
{

#pragma omp for schedule (static, chunk)
  
    for (i=0; i<d; i++)    
    {       
      for (k=0; k<n1; k++)
        z[i] += W[i][k] * x[k];
    };
};
    softmax(d, z, z);
    z[d] = -log(z[l]); 
    return 0;
};


int bpred(double* x, double* z, int l) //incremental reverse of pred
{   z[l]-=1;
    for(int i=0; i < d; i++){
        saxpy(n1, x, z[i], bW[i]); // parallel region must be implemented*
    };
    z[l]-=1;
    return 0;
}

int b2pred(double* x, double* z, int l) //combination of pred and bpred
{    for (int i = 0; i< d; i++) z[i] = dot(n1,W[i],x);  //has been integrated into bemprisk
     softmax(d, z, z);
     z[d] = -log(z[l]);
     z[l]-=1;
     for(int i=0; i < d; i++) saxpy(n1, x, z[i], bW[i]);
     z[l]+=1;
    return 0;
}

int emprisk(double* risk) //emp risk penalized by the proximal term
{   //*risk = 0.5*q*square(norm(n1*d,*W)); //proximal term
    *risk=0; 
    for (int k = 0; k< m; k++ )   //depends globally on W
    { pred(X[k],Z[k],Y[k]);   // global z serves as workspace
      *risk += Z[k][d]; //cross entropy
      //printf("\n %f ", *risk);
    };
    return 0;
};

double accuracy() // counts how often we are wrong on the test set
{
    int count = 0;
    x[n] = 1;
    for (int k = 0; k< mt; k++ )
    { int l = test_label[k];
      for(int i= 0;i< n ;i++)
          x[i] = test_image[k][i];
      pred(x,z,l);   // global z serves as workspace
      int wrong = 0;
      for(int i= 0;i < 10;i++)
            if (z[i] > z[l]) wrong = 1 ; //label has not maximal probability
      if(wrong) count++;
    };
    return count/((double)mt);
};

int bemprisk(double* risk)  // empirical risk and gradient evaluation
{   *risk=0;
    double scale = 0;    // could always be zero mathematically
    resetzero(n1*d, *bW);
    for (int k = 0; k< m; k++ )
    {   int l = Y[k];
 //       z = Z[k];         intermediate z values do not need to be kept
        x = X[k];
        double sum = 0;
        for (int i = 0; i< d; i++)
        { z[i] = exp(dot(n1,W[i],x)-scale);
            sum += z[i];};
//      softmax(d, z, z);  // was integrated into the main loops to facilitate parallelization
        z[l] -= sum;
        for(int i=0; i < d; i++)
          {z[i] /= sum;
              saxpy(n1, x, z[i], bW[i]);};
        *risk += -log(1+z[l]);
        scale += max(log(sum),-scale*0.1);  // presumably this helps with the numerics
    }
    return 0;
};

// This is for the global method
void setcircle(double den)     // compute the tangent and radial vector
{
 //   scale(d*n1,*dW,1.0/tnorm,*dW);    // (re)normalize the tangent
  //  double gnorm = norm(d*n1,*bW);
    if(den <=0)     /// should never happen
    { omega = 0;
      resetzero(d*n1,*ddW);    //
      printf("\n now we are following straight lines");
      scale(d*n1, *bW, -1/gnorm, *dW); // tangent = normalized steepest descent
      tnorm = 1;
    }
    else
    { tnorm = norm(d*n1,*dW); //norm of the tangent 
      iprod = dot(d*n1,*bW,*dW)/square(tnorm);
      scale(n1*d,*dW,iprod/den,*ddW);  // set second derivative to
      saxpy(n1*d,*bW,-1/den,*ddW);
//      check = dot(n1*d, *ddW, *dW);
      omega = norm(n1*d,*ddW);         // normalize second derivative in Euclidean norm.
      tnorm = norm(d*n1,*dW);
      if(omega != 0) scale(n1*d,*ddW,1/omega,*ddW);   //ddW is either zero or normalized.
    };
//    if(fabs(tnorm-1)+ fabs(check) > 0.00001)
 //       printf("\n tnorm %f,  check %f  omega %e gnorm %e \n",tnorm, check, omega, gnorm);
};

double trigsolve(double omega, double zbar, double zhat,double ztil, double* teast)
{double sigma = firstsign(zbar, zhat, ztil);   // possible here but does is dicy in real switching case
    double tea = *teast; double tau = tea*omega;
    if (sigma < 0) { zbar *= sigma; zhat*= sigma; ztil*= sigma;}
    double ztest = zbar - *teast*(fabs(zhat)-*teast*omega*min(0,ztil)/2);
 // if(ztest<0) printf("\n \n ztest in trigsolve %f zbar %e zhat %f ztil %f", ztest, zbar, zhat, ztil);
    if (ztest >= 0) return tea;   // No change in upper bound since test failed
    if(omega == 0){      // steppest descent case below target or by accident
        if(zbar*zhat < 0) tea = -zbar/zhat;  // second derivative term drops out
     return tea;
    }
    else { zbar*= omega;}    // ztil has like already has one omega in denominator
    {double rho = sqrt(square(zhat)+square(ztil));
     double zbplzt;
     zbplzt = zbar+ztil;
        if(fabs (zbplzt)<=  rho)
        { // printf("\n trigonometrics zbplzt/rho %f ",zbplzt/rho);
            double delta = atan2(zhat,ztil);
            double tautil = acos(zbplzt/rho);
            if ((- delta - tautil > 0) && (zbar > 0.00000001) ) tau = - delta - tautil;
            else if ( tautil - delta > 0.0000001 ) tau = tautil - delta;
            else {tau = 2*M_PI - delta - tautil;}
            double err = zbar+ zhat*sin(tau) + ztil*(1-cos(tau));
            double slope = zhat*cos(tau) + ztil*sin(tau);
            if(fabs(err) >= 0.00001) printf("\n omega %e root test tau %e res %e slope %e", omega, tau, err, slope);
            //if(tau < *teast) *teast = tau;
        };
     //printf("%f targetea, %f tau \n", tau/omega, tau);
     return tau/omega;
    };
};

// This is for the global method


int main(int argc, char *argv[]) 
{
    // omp parameters
    //int	 chunk = 100;
    ///////////

     double y, el, elt, eta;
     n = SIZE;
     n1 = n+1;
     //srand((int)time(0));
     srand(2019);
     //printf("layer size: %i \n",d);
     //printf("Enter data size: \n");
     //scanf("%i",&n);
     printf("feature dimension: %i \n",n);
     //printf("Enter layer size:\n");
     d = 10;
     //q=0.000000000001*2*d;
     q=0;
     e = 1;
     d1 = d+1;
     printf("layer size: %i \n",d);
     //printf("Enter sample size:\n");
     //scanf("%i",&m);
     m = NUM_TRAIN;
     mt = NUM_TEST;
     printf("sample size: %i \n",m);
     int meth = 3;
     printf("method: %i \n",meth);
     sw = m*d1;
        eta = 2*M_PI/60;
     targetred = 0.5 ;
     printf("angle bound: %f sensitivity %f targetred %f \n",eta, e, targetred);
     int maxit =750;
     printf("maxit: %i \n",maxit);
     z = (double*)calloc(d,sizeof(double)); //intermediate layer
     bz = (double*)calloc(d,sizeof(double)); //adjoints of intermediates
     x = (double*)malloc(n1*sizeof(double));  //input data
     xb = (double*)calloc(n1,sizeof(double));  //input gradient
     p = (double*)malloc(d*sizeof(double));   //fixed output weights
     W = (double**)malloc(d*sizeof(double*)); //Weight matrix
     Wmin = (double**)malloc(d*sizeof(double*)); //Weight matrix
     Wini = (double**)malloc(d*sizeof(double*)); //Weight matrix
     dW = (double**)malloc(d*sizeof(double*)); //Weight matrix tangent
     ddW = (double**)malloc(d*sizeof(double*)); //Weight matrix curvature
     bW = (double**)malloc(d*sizeof(double*)); //adjoined weight matrix
     *W = (double*)calloc(d*n1,sizeof(double)); // continuous allocation vector
     *Wini = (double*)calloc(d*n1,sizeof(double)); // continuous allocation vector
     *dW = (double*)calloc(d*n1,sizeof(double)); // continuous allocation vector
     *ddW = (double*)calloc(d*n1,sizeof(double)); // continuous allocation vector
     *bW = (double*)calloc(d*n1,sizeof(double)); // contiguous allocation vector
     *Wmin = (double*)calloc(d*n1,sizeof(double)); // continuous allocation vector
    
     fptre = fopen("./eltdat.txt","w");
  //   fptrt = fopen("./tardat.txt","w");
    for( int i = 0; i < d; i++)                  // allocation of weights and initializations
    { W[i] = *W+n1*i;
      Wmin[i] = *Wmin+n1*i;
      Wini[i] = *Wini+n1*i;
      dW[i] = *dW+n1*i;                     // only needed for TOAST
      ddW[i] = *ddW+n1*i;                   //       "
      bW[i] = *bW+n1*i;                     //       "
    };
    for( int j = 0; j < n; j++) x[j] = (2*(double)rand())/RAND_MAX-1; 
     x[n] = 1;//allocation of single sample poin

 // Verification of adjoints by devided differences

    setsample();    //Initialize the training set of m samples with same random numbers
    //nf(m, n, X);
    // printmatrix(m, n1,X);
    for (int i = 0; i < n1*d; i++)
       (*W)[i] = (*Wini)[i] = myrandnumb()- 0.5 ;
    // printmatrix(d,n,W);
    pred(x,z,1);
    y =z[d];//prediction function evaluation
    printf("\n prediction value %f  \n \n ",y);  //print prediction value
    bpred(x,z,1); //adjoint prediction function evaluation
    x[1] += 0.01;
    pred(x,z,1); //check x-derivatives against divided difference
    printf("%f,  %f  xerror \n \n ",xb[1],(z[d]-y)/0.01);
    x[1] -= 0.01;
    W[1][0] += 0.01;
    pred(x,z,1);    ///check W-derivative against devided differences
    printf("%f,  %f  Werror \n \n ",bW[1][0],(z[d]-y)/0.01);
    W[1][0] -= 0.01;
    W[0][n] += 0.01;
    pred(x,z,1);    // check W-derivative against devided differences
    printf("%12.7f,  %12.7f  berror \n \n ",bW[0][n],(z[d]-y)/0.01);
    emprisk(&el);
    double el0 = el;
    printf(" lossvalue %f,  \n",el);
    bemprisk(&el);
    printf(" lossvalue using bemprisk %f,  \n",el);
    //W[3][0] += 0.01;
    emprisk(&elt);    // check W-derivative against devided differences
    //printf("%12.7f,  %12.7f  Werror \n \n ",bW[3][0],(elt-el)/0.01);
    //W[3][0] -= 0.01;
    //W[2][n] += 0.01;
    //emprisk(&elt);    // check b-derivative against devided differences
    //printf("%12.7f,  %12.7f  berror \n \n ",bW[2][n],(elt-el)/0.01);
    //W[2][n] -= 0.01;
 // End checking against divided difference
 // Begin learning
    if(meth==3)// This is for TOAST
    { 
      double mul1, mul2;
      double taust = 0;
      double teast = 0;
      double target = 0 ;     // not active currently
      double targetol = 0.00000001;
      double t = 0; // length of trajectory
      double elmin = 1/0.0;
      int it, bestit = -1;
        for(it =0;it < maxit; it++){
            bemprisk(&el); // Compute the function value and gradients  bb and bW
            gnorm = norm(n1*d, *bW);
            if(it%10==0) printf("it %i, emprisk %12.4f , target %12.4f , gnorm %12.4f , taustar %8.2f,  teastar %12.4f \n", it, el/m, target/m, gnorm/m, taust/eta, teast);
            if(it%1==0) {fprintf(fptre,"%i, %f \n",it,log(min(1,el/el0))/log(10.));};
                // fprintf(fptrt,"%i, %f \n",it,target);};
            if(el<elmin)
                { elmin = el; bestit = it;
                for (int i = 0; i < n1*d; i++)
                    (*Wmin)[i] = (*W)[i]; };
            if(it == 0) // initialize tangent to perturbed negative gradient
            {   target = el0/2;
            //    target=0.28*60000;
                for (int i = 0; i < d*n1 ; i++)(*dW)[i] = - (*bW)[i]*(1 + 0.1*myrandnumb());   //initialize tangent to perturb SD
                tnorm = norm(n1*d,*dW);
                //printf(" initial gradient size  %18.12f \n ", tnorm);
            };
            if(el<target +targetol){
                target *= targetred;
                printf("\n it %i target reached and reduced el %18.12e target %18.12e \n ", it, el, target);
                for (int i = 0; i < d*n1 ; i++) (*dW)[i] = - (*bW)[i]*(1 + 0.1*myrandnumb());   //initialize tangent to perturb SD
                tnorm = norm(n1*d,*dW);
                if(target<0.0001) exit(50);
                //printf(" reinitial gradient size  %18.12f \n ", tnorm);
            };
            // Compute the circle
            scale(d*n1,*dW,1.0/tnorm,*dW);
            tnorm = 1;
            e = 1.0;
            setcircle((el-target)/e); //
    //        printf("%18.12f omega, %18.12f tnorm, %18.12f iprod \n", omega, tnorm, iprod);
            // compute targetea
            double zbar, zhat, ztil, targetea;
            zbar = (el-target);
            // double umin = -(0.5/q)*square(norm(n1*d,*bW));
            zhat = iprod;
            ztil = 0.0;
            double teastold = teast;
            if(omega == 0)
            {   printf("it %i, omega equal to zero \n", it);
                targetea = fabs((target-el)/zhat);   // go to mimimizer which might be below target
            }
            else{
                ztil = q/omega - zbar*omega;
                teast = 2*M_PI/omega;  //  computing targetea
                targetea = trigsolve(omega, zbar, zhat, ztil, &teast);
            };
            teast = targetea;
            if(it) teast = min(teast, teastold*3);
            //printf("%18.12f aux \n", aux);
            taust = min(teast*omega, eta);
            if( taust == targetea*omega) printf(" Level set in reach %18.12f, target %18.12f, %i, \n ", el, target, it);
            t += teast;
            if(taust == 2*M_PI){
             printf("Adjust c \n");
             exit(1);
            };
            // Update the point and tangent
            //double check0 = norm(n1*d,*dW);
            //double check1 = norm(n1*d,*ddW);
            if(omega == 0){
                saxpy(n1*d,*dW,teast,*W);
            }      // straight line
            else{
                 //mul1= iprod*sin(taust)/omega - (1-cos(taust))*(el-target);
                 //mul2 = 2*(1-cos(taust))/square(omega);
                 //printf("%18.12f taust, %18.12f omega, %18.12f taustom \n", taust, omega, taustom);
                 saxpy(n1*d,*dW,sin(taust)/(omega*tnorm),*W);
                 saxpy(n1*d,*ddW,(1-cos(taust))/omega,*W);
                 scale(n1*d,*dW,cos(taust)/tnorm,*dW);
                 saxpy(n1*d,*ddW,sin(taust),*dW);
  //               double riska;
  //               emprisk(&riska);
                 //q=max(q*0.9, fabs(riska - el - mul1)/mul2);
                 //double check2 = norm(n1*d, *dW);
                 // if(fabs(check0-1)+fabs(check1-1)+fabs(check2-1)> 0.000001)
                 //    printf("\n check0 %18.12f, check1 %18.12f, omega %18.12e,  check2  %18.12f, teast %f, \n ", check0, check1,omega, check2, teast);
            };
            //printf("\n qvalue %f , emprisk= %e, taustar %f, targetea %f", q, riska, taust, targetea);
        };  
     printf("\n maxit reached elmin %18.12f bestit %i trajectory %f target %f  \n ", elmin/m, bestit, t, target/m);
     fclose(fptre);
    //fclose(fptrt);
     W = Wmin;
     for (int i = 0; i < d*n1 ; i++) (*Wini)[i] -= (*Wmin)[i];
        double dist = norm(n1*d, *Wini);
        printf("failure rate %12.4f distance %12.4f \n",accuracy(), dist);
     return 14;
    }
};
