#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512


// number of real galaxies
int NoofReal;
int NoofSim;
// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;

float pi = acos(-1.0);

unsigned int  *histogramDD,*histogramDR, *histogramRR;

/*Function for doing the calculations */
__global__ void histogram_calc(float *ra1, float *decl1,  float *ra2, float *decl2, float pi, int noofnumb, unsigned int *histogram){
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    float theta;
    float pif;
    

    if( index < noofnumb)
        for( int i = 0; i < noofnumb; ++i){
            theta =  acos(sin(decl1[index]) * sin(decl2[i]) + cos(decl1[index]) * cos(decl2[i]) * cos(ra1[index] - ra2[i]));
            pif = theta * (180 /pi);
            atomicAdd(&histogram[(int)(pif*4)], 1);
        }
}


int main(int argc, char *argv[]){
    int readdata(char *argv1, char *argv2);
    int getDevice(int deviceno);

    unsigned long long int histogramDDsum, histogramDRsum, histogramRRsum;
    unsigned int *gpu_DR, *gpu_DD, *gpu_RR;
    float *gpu_real_ra, *gpu_real_dl, *gpu_rand_ra, *gpu_rand_dl;

    int noofblocks;
    double w;

    double start, end, kerneltime;
    struct timeval _ttime;
    struct timezone _tzone;

    FILE *outfil;

    if (argc != 4 ){
        printf("Usage: galaxy.out real_data random_data output_data\n");
        return -1;
    }

    if ( getDevice(0) != 0 ) return(-1);

    if (readdata(argv[1], argv[2]) != 0){
        return -1;
    }

    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    /*Allocate the memory*/
    size_t size = totaldegrees * binsperdegree * sizeof(unsigned int);
    
    histogramDD = (unsigned int *)malloc(size);
    histogramDR = (unsigned int *)malloc(size);
    histogramRR = (unsigned int *)malloc(size);

    //initialize histograms to zero
    for (int i=0; i < totaldegrees * binsperdegree; i++){
         histogramDD[i] = 0U;
         histogramDR[i] = 0U;
         histogramRR[i] = 0U;
    }

    //Allocate memory on the gpu
    cudaMalloc(&gpu_DR, size);
    cudaMalloc(&gpu_DD, size);
    cudaMalloc(&gpu_RR, size);

    size_t size_real = (NoofReal) * sizeof(float);

    cudaMalloc(&gpu_real_ra, size_real);
    cudaMalloc(&gpu_real_dl, size_real);
    cudaMalloc(&gpu_rand_ra, size_real);
    cudaMalloc(&gpu_rand_dl, size_real);


    /* copy the data to GPU */
    cudaMemcpy(gpu_real_ra, ra_real, size_real, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_real_dl, decl_real, size_real, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rand_ra, ra_sim, size_real, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rand_dl, decl_sim, size_real, cudaMemcpyHostToDevice);

    
    noofblocks = (NoofReal + threadsperblock - 1) / threadsperblock;

    /*Launch the kernel and call the function to calculate the histogram*/
    histogram_calc <<<noofblocks, threadsperblock>>>(gpu_real_ra, gpu_real_dl, gpu_real_ra, gpu_real_dl, pi, NoofReal, gpu_DD);
    histogram_calc <<<noofblocks, threadsperblock>>>(gpu_real_ra, gpu_real_dl, gpu_rand_ra, gpu_rand_dl, pi, NoofReal, gpu_DR);
    histogram_calc <<<noofblocks, threadsperblock>>>(gpu_rand_ra, gpu_rand_dl, gpu_rand_ra, gpu_rand_dl, pi, NoofSim, gpu_RR);

    /*Wait for everything to be completed */
    cudaDeviceSynchronize();

    /*Copy the results back */
    cudaMemcpy(histogramDD, gpu_DD, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramDR, gpu_DR, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramRR, gpu_RR, size, cudaMemcpyDeviceToHost);

    cudaFree(gpu_DD);
    cudaFree(gpu_DR);
    cudaFree(gpu_RR);

    cudaFree(gpu_real_ra);
    cudaFree(gpu_real_dl);
    cudaFree(gpu_rand_ra);
    cudaFree(gpu_rand_dl);


    
    /*calculate the sum had chack that they are the correct size*/
    histogramDDsum = 0LLU;
    histogramDRsum = 0LLU;
    histogramRRsum = 0LLU;
    for (int i = 0; i < totaldegrees * binsperdegree; ++i){
        histogramDDsum += histogramDD[i];
        histogramDRsum += histogramDR[i];
        histogramRRsum += histogramRR[i];
    }
    printf("Histogram DD sum = %lld\n", histogramDDsum);
    if (histogramDDsum != 10000000000LLU){
        printf("Incorrect histogram sum\n");
    }
    
    printf("Histogram DR sum = %lld\n", histogramDRsum);
    if (histogramDRsum != 10000000000LLU){
        printf("Incorrect histogram sum\n");
    }
    
    printf("Histogram RR sum = %lld\n", histogramRRsum);
    if (histogramRRsum != 10000000000LLU){
        printf("Incorrect histogram sum\n");
    }


    /*Write the results to a file and print out the first omega values*/
    outfil = fopen(argv[3], "w");
    printf("first Omega values:");
    for (int i = 0; i < totaldegrees; ++i) {
        if (histogramRR[i] > 0){
            w = (histogramDD[i] - 2*histogramDR[i] + histogramRR[i]) / ((double)(histogramRR[i]));
            if (i < 10) {
                printf(" %6.4f", w);
            } 
            fprintf(outfil, "%6.4f\t%15lf\t%d\t%d\t%d\n", ((float)i)/binsperdegree, w, histogramDD[i], histogramDR[i], histogramRR[i]);
        }else {
            if (i < 10) printf(" ");
        }
        
    }
    fclose(outfil);

    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    kerneltime += end-start;
    printf("Time taken = %.2lf s\n", ((float) kerneltime));

    return 0;
}


int readdata(char *argv1, char *argv2){

    int i,linecount;
    char inbuf[180];
    double ra, dec, dpi;
    FILE *infil;
                                         
    printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta in radians:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

    dpi = acos(-1.0);
    infil = fopen(argv1,"r");
    if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

    // read the number of galaxies in the input file
    int announcednumber;
    if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
    linecount =0;
    while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
    rewind(infil);

    if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
    else {
        printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
        return(-1);
        }

    NoofReal = linecount;
    ra_real   = (float *)calloc(NoofReal,sizeof(float));
    decl_real = (float *)calloc(NoofReal,sizeof(float));

    // skip the number of galaxies in the input file
    if ( fgets(inbuf,180,infil) == NULL ) return(-1);
    i = 0;
    while ( fgets(inbuf,80,infil) != NULL ){

        if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) {
            printf("   Cannot read line %d in %s\n",i+1,argv1);
            fclose(infil);
            return(-1);
            }
        /*Added so that phi is used for this filling in ra_real, decl_real, ra_sim and decl_sim*/
        ra_real[i]   = (float)(ra/60.0 * dpi/180.0);
        decl_real[i] = (float)(dec/60.0 *dpi/180.0);
        ++i;
        }

    fclose(infil);

    if ( i != NoofReal ) {
        printf("   Cannot read %s correctly\n",argv1);
        return(-1);
        }

    infil = fopen(argv2,"r");
    if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

    if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
    linecount =0;
    while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
    rewind(infil);

    if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
    else{
        printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
        return(-1);
        }

    NoofSim = linecount;
    ra_sim   = (float *)calloc(NoofSim,sizeof(float));
    decl_sim = (float *)calloc(NoofSim,sizeof(float));

    // skip the number of galaxies in the input file
    if ( fgets(inbuf,180,infil) == NULL ) return(-1);
    i =0;
    while ( fgets(inbuf,80,infil) != NULL ){
        if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) {
            printf("   Cannot read line %d in %s\n",i+1,argv2);
            fclose(infil);
            return(-1);
            }
        ra_sim[i]   = (float)(ra/60.0 * dpi/180.0);
        decl_sim[i] = (float)(dec/60.0 *dpi/180.0);
        ++i;
        }

    fclose(infil);

    if ( i != NoofSim ) {
        printf("   Cannot read %s correctly\n",argv2);
        return(-1);
        }

    return(0);
}




int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != deviceNo ) printf("   Unable to set device %d, using device %d instead",deviceNo, device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

