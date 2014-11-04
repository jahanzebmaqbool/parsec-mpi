#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <fstream>
#include <mpi.h>

#include "OptionDataStruct.h"
#include "blackschole_lib.h"

// Define Configuration...
#define NUM_ITERATIONS 100



int numOptions;
OptionData* optionData;
OptionData* subData;
float     * resultOptionPrices;
float     * chunkOptionPrices;


// Read total number of Options in input File for all processors.

void ReadNumOptions (char* inputFile) 
{

     FILE* file;	
     int rv ;

     // Read input data from file
     file = fopen(inputFile, "r");

     if(file == NULL) {
       printf("ERROR: Unable to open file `%s'.\n", inputFile);
       exit(1);
     }

     rv = fscanf(file, "%i", &numOptions);
     if(rv != 1) {
       printf("ERROR: Unable to read from file `%s'.\n", inputFile);
       fclose(file);
       exit(1);
     }
}



// Read Input Set provided by PARSEC benchmark in an Array of OptionData

void ReadInputFile (char* inputFile)
{
	
     printf("...Struct size %d.\n", sizeof(OptionData));
     FILE* file;	
     int loopnum ;
     int rv ;
  
     printf("...Reading input data in CPU mem %s.\n", inputFile);
     // Read input data from file
     file = fopen(inputFile, "r");
     

     if(file == NULL) {
       printf("ERROR: Unable to open file `%s'.\n", inputFile);
       exit(1);
     }

        optionData = (OptionData*) malloc(numOptions*sizeof(OptionData));	     

	
	int garbageVariable;
        fscanf(file, "%i", &garbageVariable);

        // READING INPUT FROM FILE, and store it in array of STURCT.
		
		std::ofstream fout ("stokeprice.txt", std::ios::app);
		
        for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
        {
    	    rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &optionData[loopnum].s, &optionData[loopnum].strike, &optionData[loopnum].r, &optionData[loopnum].divq, &optionData[loopnum].v, &optionData[loopnum].t, &optionData[loopnum].OptionType, &optionData[loopnum].divs, &optionData[loopnum].DGrefval);
			fout << optionData[loopnum].s << "\n"; 
        }
        
	printf("... SUCCESS: Read data from input file... `%s'.\n", inputFile);	
	printf("... Going to close file... `%s'.\n", inputFile);
	printf("... Num Options....%d\n", numOptions);
	
	rv = fclose(file);
	fout << std::flush;
	fout.close();
	
        if(rv != 0) {
 	     printf("ERROR: Unable to close file `%s'.\n", inputFile);
 	     exit(1);
        }
	
}

////////////////////////////////////////////////////////////////////////////////
// WRITE OUTPUT RESULT FILE 
////////////////////////////////////////////////////////////////////////////////

void writePriceResults (char* filename, float* prices)
{	
	FILE* file;
	int rv;
	int loopIter;
	
	file = fopen(filename, "w+");
	if(file == NULL) {
	      printf("ERROR: Unable to open file `%s'.\n", filename);
	      exit(1);
	}	
	
	for (loopIter = 0 ; loopIter < numOptions ; loopIter ++)	
		rv = fprintf(file, "%.18f\n", prices[loopIter]);
	
	if(rv < 0) {
           printf("ERROR: Unable to write to file `%s'.\n", filename);
           fclose(file);
           exit(1);
        }

	fclose (file);       	
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////


main(int argc, char **argv)
{
  int rank, size, status;
  char* inputFile = argv [1]; 
  char* outputFile = argv [2];	
   	
  // Define Time vars.
    struct timeval fread_start, fread_end, malloc_start, malloc_end, computation_start, computation_end, fwrite_start, fwrite_end;
    long fread_time_usec, fread_time_second, fwrite_time_usec, fwrite_time_second, computation_time_usec, computation_time_second, malloc_time_usec, malloc_time_second;
	double averageCompuationTime;


    //gettimeofday(&start, NULL);

  // defining own data type of OptionData...

  MPI_Datatype optionType, oldtypes [2];
  int blockcounts [2];
  MPI_Aint offsets [2], extent;
  

  // MPI Initialization...
   MPI_Init(&argc,&argv);
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   MPI_Comm_size (MPI_COMM_WORLD, &size);
 
   // setting up description for 8-floats in OptionData...
   offsets [0] = 0;
   oldtypes [0] = MPI_FLOAT;
   blockcounts [0] = 8;
   /* setting up description for 1-char in OptionData */
   /* Need to first figure offset by getting size of MPI_FLOAT */
   MPI_Type_extent (MPI_FLOAT, &extent);
   offsets [1] = 8 * extent;
   oldtypes [1] = MPI_CHAR;
   blockcounts [1] = 1;
   /* Now define structured type and commit it */
   MPI_Type_struct (2, blockcounts, offsets, oldtypes, &optionType);
   MPI_Type_commit (&optionType);
   

  // All proc : Get to know how many num. options are in inputFile.
   ReadNumOptions (inputFile);
   //printf ("Num options are : %d\n", numOptions);
 
  // Setup Task division Configuration...
  int chunkSize = numOptions / size;

  gettimeofday (&malloc_start, NULL);
  subData = (OptionData*) malloc(chunkSize*sizeof(OptionData));	
  // Create an output array to hold the 'optionPrices' as result.
  chunkOptionPrices = (float*) malloc(chunkSize*sizeof(float));	   

  gettimeofday (&malloc_end, NULL);
  //malloc_time = ( (malloc_end.tv_usec - malloc_start.tv_usec));  // time in microsecond (10E-3)
 
  malloc_time_usec = (double)(malloc_end.tv_usec - malloc_start.tv_usec);  
  malloc_time_second = malloc_end.tv_sec - malloc_start.tv_sec; 

  double total_malloc_time_ms = (malloc_time_second*1000) + (malloc_time_usec/1000);
  
  printf("<%d> BlackScholes Mem Aloc Time    : %f msec\n",rank, total_malloc_time_ms);

  // Rank-0 : will be considered as Head node and it will be responsible for reading inputdata from file and
  // copying it into an array of OptionData. Then it will scatter the workload among the compute nodes.
  // Then after computation, it will Gather the results from all the compute nodes and save it in a file.
  
  if (rank == 0) {

    printf ("Chunk Size is : %d\n", chunkSize);
	
    /* Step-1 : Read Option Data from input file. */

	gettimeofday (&fread_start, NULL);
    		ReadInputFile (inputFile);
	gettimeofday (&fread_end, NULL);
	//fread_time = ( (fread_end.tv_usec - fread_start.tv_usec));  // time in microsecond (10E-3)	
	fread_time_usec = (double)(fread_end.tv_usec - fread_start.tv_usec);  
    fread_time_second = fread_end.tv_sec - fread_start.tv_sec; 
    double total_fread_time_ms = (fread_time_second*1000) + (fread_time_usec/1000);	
	
	printf("<%d> BlackScholes File Read Time    : %f msec\n",rank, total_fread_time_ms);
    printf ("Successfull read file and saved data into OptionData array....\n");	 
	
    /* Step-1.1 : Allocate memory for the resultant array to hold the Option prices results */
    resultOptionPrices = (float*) malloc(numOptions*sizeof(float));	   
   
  }
    /* Step-2 : Distribute Workload by Scattering OptionData array among the compute Nodes */
    /* Scattering the OptionData to all processes of the COMM WORLD */

    MPI_Scatter (optionData, chunkSize, optionType, subData, chunkSize, optionType, 0, MPI_COMM_WORLD); 	


    /* Step-3 : Calculate OptionPrices for 'put' and 'call' stock options in 'optionData' */
    /* Calculating Option Prices for each Option in stock */
    /* Each process will be going to do perform this operation */
	

	gettimeofday (&computation_start, NULL);
  	    calculateOptionPrice (subData, chunkOptionPrices, chunkSize, NUM_ITERATIONS);
    gettimeofday (&computation_end, NULL);
	//computation_time = ( (computation_end.tv_usec - computation_start.tv_usec));  // time in seconds (10E-3)
   
	computation_time_usec = (double)(computation_end.tv_usec - computation_start.tv_usec);  
    computation_time_second = computation_end.tv_sec - computation_start.tv_sec; 
    long total_exec_time_ms = (computation_time_second*1000) + (computation_time_usec/1000);	

   
	printf("<%d> BlackScholes Computation Time    : %ld msec\n", rank, total_exec_time_ms);
    
    /* Every Processor has successfully completed his part of work. */
    /* Option Prices have been successfull calculated.              */	
	
	printf ("Process <%d> : Option Prices have been successfully calculated.\n", rank);	

    /* Now, all is done, so gather back 'chunk option Prices' from each processor to 'processos-0' */
	
	printf ("Process <%d> : Gathering Back Data from all processors.\n", rank);	

            MPI_Gather (chunkOptionPrices, chunkSize, MPI_FLOAT, resultOptionPrices, chunkSize, MPI_FLOAT, 0,
			MPI_COMM_WORLD);    

    /* Successfully gathered Data at Process-0 */
	
	printf ("Process <%d> : Successfully gathered Data at Process-0.\n", rank);	
	
    /* Process <0> : Going to write Output to a file */
	

	long *sendbuff, *recvbuff;
	int count = 1;
	sendbuff = (long*) malloc (count*sizeof(long));
	recvbuff = (long*) malloc (count*sizeof(long));
	sendbuff [0] = total_exec_time_ms;

//printf("<%d> BlackScholes Computation Time    : %f msec\n",rank, (float)computation_time/1000);

	MPI_Reduce( sendbuff, recvbuff, count, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD );

      if (rank == 0) {		

		writePriceResults (outputFile, resultOptionPrices);	
        	printf ("Process <%d> : Successfully Saved OptionResult in <%s>.\n", rank, outputFile);		
		
		averageCompuationTime = (float)recvbuff [0]/size;
		printf ("Average Computation time of all processors is = %f msec \n", averageCompuationTime );		
      }


  //MPI_Type_free (&optionType);
  MPI_Finalize();
}
