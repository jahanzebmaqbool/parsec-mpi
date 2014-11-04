/**
* This code is the Distributed Memory implementation of Fluidanimate SPH solver from popular PARSEC Multithreaded Benchmark.
* I used serial and multithreaded code of PARSEC benchmark and further extended it to add support for Message Passing Interface.
* Domain decomposition of original 3-D grid is done in Cartesian manner and proper datastructure and utility functions are added
* to enable MPI based parallel implementaion.

* Original Author: Richard O. Lee
* Multithreaded Modified by: Chirstian Bienia and Christian Fensch
* MPI implementation by: Jahanzeb Maqbool, SEECS, NUST, Pakistan.
* 
*
*/



// Code originally written by Richard O. Lee
// Modified by Christian Bienia and Christian Fensch
// MPI implementation by Jahanzeb Maqbool, SEECS, NUST, Pakistan



#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <assert.h>

#define DEBUG false


static inline int isLittleEndian() {
  union {
    uint16_t word;
    uint8_t byte;
  } endian_test;

  endian_test.word = 0x00FF;
  return (endian_test.byte == 0xFF);
}

union __float_and_int {
  uint32_t i;
  float    f;
};

static inline float bswap_float(float x) {
  union __float_and_int __x;

   __x.f = x;
   __x.i = ((__x.i & 0xff000000) >> 24) | ((__x.i & 0x00ff0000) >>  8) |
           ((__x.i & 0x0000ff00) <<  8) | ((__x.i & 0x000000ff) << 24);

  return __x.f;
}

static inline int bswap_int32(int x) {
  return ( (((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |
           (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24) );
}


/**
 * Originally written by: Christian Bienna
*/
// note: icc-optimized version of this class gave 15% more
// performance than our hand-optimized SSE3 implementation
class Vec3
{
public:
    float x, y, z;

    Vec3() {}
    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    float   GetLengthSq() const         { return x*x + y*y + z*z; }
    float   GetLength() const           { return sqrtf(GetLengthSq()); }
    Vec3 &  Normalize()                 { return *this /= GetLength(); }
	bool      compare(Vec3 vector) {// return true if difference is 0
	Vec3 result = *this - vector;
	if(result.GetLengthSq()==0.0f)
	return true;
	else
	return false;
	}
	bool      isSame(Vec3 vector) {// returns true if both are same
	bool X = x == vector.x;
	bool Y = y == vector.y;
	bool Z = z == vector.z;
	if(X && Y && Z)
	return true;
	else
	return false;
	}
	
	
    Vec3 &  operator += (Vec3 const &v) { x += v.x;  y += v.y; z += v.z; return *this; }
    Vec3 &  operator -= (Vec3 const &v) { x -= v.x;  y -= v.y; z -= v.z; return *this; }
    Vec3 &  operator *= (float s)       { x *= s;  y *= s; z *= s; return *this; }
    Vec3 &  operator /= (float s)       { x /= s;  y /= s; z /= s; return *this; }

    Vec3    operator + (Vec3 const &v) const    { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3    operator - () const                 { return Vec3(-x, -y, -z); }
    Vec3    operator - (Vec3 const &v) const    { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3    operator * (float s) const          { return Vec3(x*s, y*s, z*s); }
    Vec3    operator / (float s) const          { return Vec3(x/s, y/s, z/s); }
	
    float   operator * (Vec3 const &v) const    { return x*v.x + y*v.y + z*v.z; }
	
};


// there is a current limitation of 16 particles per cell
// (this structure use to be a simple linked-list of particles but, due to
// improved cache locality, we get a huge performance increase by copying
// particles instead of referencing them)
struct Cell
{
	Vec3 p[16];
	Vec3 hv[16];
	Vec3 v[16];
	Vec3 a[16];
	float density[16];
};

/**
 * Compares 2 cells in a partitioned subgrid and returns the new result. 
 * (by comparing 2 cells means, comparing their respective Vectors i.e., position, velocity etc.)
 * Added by: Jahanzeb Maqbool
*/
int compareCell(Cell cell,int n,Cell cell2 , int n2){
	if(n!=n2)
		return -1;
	int i=0;
	for(i=0;i<n;i++){
		if(!cell.p[i].isSame(cell2.p[i]))
			return 1;
		if(!cell.hv[i].isSame(cell2.hv[i]))
			return 2;
		if(!cell.v[i].isSame(cell2.v[i]))
			return 3;
		if(!cell.a[i].isSame(cell2.a[i]))
			return 4;
		if(cell.density[i] != cell2.density[i] )
			return 5;
	}
	return 0;
}

/**
 * Subtracts 2 cells in a partitioned subgrid and returns the new result. 
 * (by subtracting 2 cells means, subtracting their respective Vectors i.e., position, velocity etc.)
 * Added by: Jahanzeb Maqbool
*/
Cell diffCell(Cell cell1,Cell cell2,int count1){
	Cell res;
	for(int i=0;i<count1;i++){
		res.p[i] = cell1.p[i] - cell2.p[i];
		res.hv[i] = cell1.hv[i] - cell2.hv[i];
		res.v[i] = cell1.v[i] - cell2.v[i];
		res.a[i] = cell1.a[i] - cell2.a[i];
		res.density[i] = cell1.density[i] - cell2.density[i];
	}
	return res;
}

/**
 * Add 2 cells in a partitioned subgrid and returns the new result. 
 * (by adding 2 cells means, adding their respective Vectors i.e., position, velocity etc.)
 * Added by: Jahanzeb Maqbool
*/
Cell addCell(Cell cell1,Cell cell2,int count1){
	Cell res;
	for(int i=0;i<count1;i++){
		res.p[i] = cell1.p[i] + cell2.p[i];
		res.hv[i] = cell1.hv[i] + cell2.hv[i];
		res.v[i] = cell1.v[i] + cell2.v[i];
		res.a[i] = cell1.a[i] + cell2.a[i];
		res.density[i] = cell1.density[i] + cell2.density[i];
	}
	return res;
}


/**
 * returns the 1-D index out of 3-D index (partitioned to flat). 
 * Added by: Jahanzeb Maqbool
*/
void compareCells(Cell* cells1,int* numPar1,int size1,Cell* cells2,int* numPar2,int size2,int rank){
	if(size1!=size2){
		if (DEBUG)
			printf("Size not equal in compare cells\n");
		return;
	}

	for(int i=0;i<size1;i++){
		int ret = compareCell(cells1[i],numPar1[i],cells2[i],numPar2[i]) ;
		if(ret != 0 ) {
			if (DEBUG)
			printf("On rank <%d> Cell <%d> are not same with error code <%d> \n",rank,i,ret);
		}
	}

}

/**
 * Added this datastructure to present a single Cell in a partitioned sub-grid.
 * Added by: Jahanzeb Maqbool
*/
struct SingleCell
{
    Vec3 p, hv, v;
    int ci, cj, ck;
};


const float timeStep = 0.005f;
const float doubleRestDensity = 2000.f;
const float kernelRadiusMultiplier = 1.695f;
const float stiffness = 1.5f;
const float viscosity = 0.4f;
const Vec3 externalAcceleration(0.f, -9.8f, 0.f);
const Vec3 domainMin(-0.065f, -0.08f, -0.065f);
const Vec3 domainMax(0.065f, 0.1f, 0.065f);

float restParticlesPerMeter, h, hSq;
float densityCoeff, pressureCoeff, viscosityCoeff;

int nx, ny, nz;				// number of grid cells in each dimension
Vec3 delta;				// cell dimensions
int origNumParticles = 0;
int numParticles = 0;
int numCells = 0;
Cell *cells = 0;
Cell *cells2 = 0;
int *cnumPars = 0;
int *cnumPars2 = 0;

int XDIVS = 1;	// number of partitions in X
int ZDIVS = 1;	// number of partitions in Z

#define NUM_GRIDS  ((XDIVS) * (ZDIVS))
#define NUM_FRAMES 1


Cell* sBuff = 0;
Cell* subCells2 = 0;
Cell* subCells = 0;

int GRIDS_SIZE = 0;
int* cnumPars2_s = 0;  // temporary....
int* cnumPars2_sub = 0;
int* cnumPars_sub = 0;
int chunkSize = 0;
MPI_Datatype vecDataType, gridDataType, cellDataType, singleCellDataType; 
std::vector <SingleCell> *swapList;



/**
 * I added this Datastructure to partition the Grod into subgrids. Each subgrid was mapped like Cartesian Topology in 3-D.
 * Further utility methods and additional datastructures were added to support the MPI communication. 
 * Added by: Jahanzeb Maqbool
*/
struct Grid
{
	int sx, sy, sz;
	int ex, ey, ez;
} *grids;

bool *border;			// flags which cells lie on grid boundaries (Ghost Cell)


////////////////////////////////////////////////////////////////////////////////

/* This code snippet is to check whether the no. of proc are in power of 2.
 * hmgweight
 *
 * Computes the hamming weight of x
 *
 * x      - input value
 * lsb    - if x!=0 position of smallest bit set, else -1
 *
 * return - the hamming weight
 */
  unsigned int hmgweight(unsigned int x, int *lsb) {
  unsigned int weight=0;
  unsigned int mask= 1;
  unsigned int count=0;
 
  *lsb=-1;
  while(x > 0) {
    unsigned int temp;
    temp=(x&mask);
    if((x&mask) == 1) {
      weight++;
      if(*lsb == -1) *lsb = count;
    }
    x >>= 1;
    count++;
  }

  return weight;
}



/**
 * Original function for the initialization of the simulation.
 * It initializes all the datastructures and populates a grid of cells having particles.
 * I modified it somehow to initialize my own data-structures needed for MPI communication.
 * 
 * original author: Christian Bienna
 * modified by: Jahanzeb Maqbool
*/

void InitSim(char const *fileName, unsigned int threadnum)
{
	//Compute partitioning based on square root of number of threads
	//NOTE: Other partition sizes are possible as long as XDIVS * ZDIVS == threadnum,
	//      but communication is minimal (and hence optimal) if XDIVS == ZDIVS
	int lsb;
	if(hmgweight(threadnum,&lsb) != 1) {
		std::cerr << "Number of threads must be a power of 2" << std::endl;
		exit(1);
	}
	XDIVS = 1<<(lsb/2);
	ZDIVS = 1<<(lsb/2);
	if(XDIVS*ZDIVS != threadnum) XDIVS*=2;
	assert(XDIVS * ZDIVS == threadnum);

	
	//thread = new pthread_t[NUM_GRIDS];
	grids = new struct Grid[NUM_GRIDS];   // skip this line and creae grids array for all of the processors.

	//Load input particles
	if (DEBUG)
	std::cout << "Loading file \"" << fileName << "\"..." << std::endl;
	std::ifstream file(fileName, std::ios::binary);
	assert(file);

	file.read((char *)&restParticlesPerMeter, 4);
	file.read((char *)&origNumParticles, 4);
        if(!isLittleEndian()) {
          restParticlesPerMeter = bswap_float(restParticlesPerMeter);
          origNumParticles      = bswap_int32(origNumParticles);
        }
	numParticles = origNumParticles;

	h = kernelRadiusMultiplier / restParticlesPerMeter;
	hSq = h*h;
	const float pi = 3.14159265358979f;
	float coeff1 = 315.f / (64.f*pi*pow(h,9.f));
	float coeff2 = 15.f / (pi*pow(h,6.f));
	float coeff3 = 45.f / (pi*pow(h,6.f));
	float particleMass = 0.5f*doubleRestDensity / (restParticlesPerMeter*restParticlesPerMeter*restParticlesPerMeter);
	densityCoeff = particleMass * coeff1;
	if (DEBUG)
	printf("densityCoeff <%f> \n",densityCoeff);
	pressureCoeff = 3.f*coeff2 * 0.5f*stiffness * particleMass;
	viscosityCoeff = viscosity * coeff3 * particleMass;

	Vec3 range = domainMax - domainMin;
	nx = (int)(range.x / h);
	ny = (int)(range.y / h);
	nz = (int)(range.z / h);
	assert(nx >= 1 && ny >= 1 && nz >= 1);
	numCells = nx*ny*nz;
	if (DEBUG)
	std::cout << "Number of cells: " << numCells << std::endl;
	delta.x = range.x / nx;
	delta.y = range.y / ny;
	delta.z = range.z / nz;
	assert(delta.x >= h && delta.y >= h && delta.z >= h);

	assert(nx >= XDIVS && nz >= ZDIVS);
	int gi = 0;
	int sx, sz, ex, ez;
	ex = 0;

	
	for(int i = 0; i < XDIVS; ++i)
	{
		sx = ex;
		ex = int(float(nx)/float(XDIVS) * (i+1) + 0.5f);
		assert(sx < ex);

		ez = 0;
		for(int j = 0; j < ZDIVS; ++j, ++gi)
		{
			sz = ez;
			ez = int(float(nz)/float(ZDIVS) * (j+1) + 0.5f);
			assert(sz < ez);

			grids[gi].sx = sx;
			grids[gi].ex = ex;
			grids[gi].sy = 0;
			grids[gi].ey = ny;
			grids[gi].sz = sz;
			grids[gi].ez = ez;
		}
	}
	assert(gi == NUM_GRIDS);

	border = new bool[numCells];
	for(int i = 0; i < NUM_GRIDS; ++i)
		for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
			for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
				for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
				{
					int index = (iz*ny + iy)*nx + ix;
					border[index] = false;
					for(int dk = -1; dk <= 1; ++dk)
						for(int dj = -1; dj <= 1; ++dj)
							for(int di = -1; di <= 1; ++di)
							{
								int ci = ix + di;
								int cj = iy + dj;
								int ck = iz + dk;

								if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
								if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
								if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;

								if( ci < grids[i].sx || ci >= grids[i].ex ||
									cj < grids[i].sy || cj >= grids[i].ey ||
									ck < grids[i].sz || ck >= grids[i].ez )
									border[index] = true;
							}
				}


 	// Allocating memory at Node - 0
	cells = new Cell[numCells];
	cells2 = new Cell[numCells];
	cnumPars = new int[numCells];
	cnumPars2 = new int[numCells];
	assert(cells && cells2 && cnumPars && cnumPars2);

	memset(cnumPars2, 0, numCells*sizeof(int));

	float px, py, pz, hvx, hvy, hvz, vx, vy, vz;
	for(int i = 0; i < origNumParticles; ++i)
	{
		file.read((char *)&px, 4);
		file.read((char *)&py, 4);
		file.read((char *)&pz, 4);
		file.read((char *)&hvx, 4);
		file.read((char *)&hvy, 4);
		file.read((char *)&hvz, 4);
		file.read((char *)&vx, 4);
		file.read((char *)&vy, 4);
		file.read((char *)&vz, 4);
        if(!isLittleEndian()) {
			px  = bswap_float(px);
            py  = bswap_float(py);
            pz  = bswap_float(pz);
            hvx = bswap_float(hvx);
            hvy = bswap_float(hvy);
            hvz = bswap_float(hvz);
            vx  = bswap_float(vx);
            vy  = bswap_float(vy);
            vz  = bswap_float(vz);
       }

	   int ci = (int)((px - domainMin.x) / delta.x);
	   int cj = (int)((py - domainMin.y) / delta.y);
	   int ck = (int)((pz - domainMin.z) / delta.z);

	   if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
	   if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
	   if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;

	   int index = (ck*ny + cj)*nx + ci;
	   Cell &cell = cells2[index];

	   int np = cnumPars2[index]; 
	   if(np < 16) {
			cell.p[np].x = px;
			cell.p[np].y = py;
			cell.p[np].z = pz;
			cell.hv[np].x = hvx;
			cell.hv[np].y = hvy;
			cell.hv[np].z = hvz;
			cell.v[np].x = vx;
			cell.v[np].y = vy;
			cell.v[np].z = vz;
			++cnumPars2[index];
		}
		else
			--numParticles;
	}
	if (DEBUG)
	std::cout << "Number of particles: " << numParticles << " (" << origNumParticles-numParticles << " skipped)" << std::endl;
}

/**
 * returns the 1-D index out of 3-D index (partitioned to flat). 
 * Added by: Jahanzeb Maqbool
*/
int getLocalIndex (Grid grid, int ix, int iy, int iz) {
    return ((iz-grid.sz)*(grid.ey-grid.sy) + (iy-grid.sy) )*(grid.ex-grid.sx) + (ix-grid.sx);
}

/**
 * Saving file in PARSEC formatting. 
 * Original code by: Christian Bienna
*/
void SaveFile(char const *fileName)
{
	
	std::cout << "Saving file \"" << fileName << "\"..." << std::endl;
	std::ofstream file(fileName, std::ios::binary);
	assert(file);

    if(!isLittleEndian()) {
        float restParticlesPerMeter_le;
        int   origNumParticles_le;

        restParticlesPerMeter_le = bswap_float(restParticlesPerMeter);
        origNumParticles_le      = bswap_int32(origNumParticles);
        file.write((char *)&restParticlesPerMeter_le, 4);
	    file.write((char *)&origNumParticles_le,      4);
    }
	else {
		file.write((char *)&restParticlesPerMeter, 4);
		file.write((char *)&origNumParticles, 4);
    }

	int count = 0;
	for(int i = 0; i < numCells; ++i)
	{
		Cell const &cell = cells[i];
		int np = cnumPars[i];
		for(int j = 0; j < np; ++j)
		{
			if(!isLittleEndian()) {
				float px, py, pz, hvx, hvy, hvz, vx,vy, vz;
				float ax,ay,az,density;
				px  = bswap_float(cell.p[j].x);
				py  = bswap_float(cell.p[j].y);
				pz  = bswap_float(cell.p[j].z);
				hvx = bswap_float(cell.hv[j].x);
				hvy = bswap_float(cell.hv[j].y);
				hvz = bswap_float(cell.hv[j].z);
				vx  = bswap_float(cell.v[j].x);
				vy  = bswap_float(cell.v[j].y);
				vz  = bswap_float(cell.v[j].z);
				// additional code 
				ax =  bswap_float(cell.a[j].x);
				ay =  bswap_float(cell.a[j].y);
				az =  bswap_float(cell.a[j].z);
				density =  bswap_float(cell.density[j]);
				// ends here
				
				file.write((char *)&px,  4);
				file.write((char *)&py,  4);
				file.write((char *)&pz,  4);
				file.write((char *)&hvx, 4);
				file.write((char *)&hvy, 4);
				file.write((char *)&hvz, 4);
				file.write((char *)&vx,  4);
				file.write((char *)&vy,  4);
				file.write((char *)&vz,  4);
				file.write((char *)&ax,  4);
				file.write((char *)&ay,  4);
				file.write((char *)&az,  4);
				file.write((char *)&density,  4);
            } 
			else {
				file.write((char *)&cell.p[j].x,  4);
				file.write((char *)&cell.p[j].y,  4);
				file.write((char *)&cell.p[j].z,  4);
				file.write((char *)&cell.hv[j].x, 4);
				file.write((char *)&cell.hv[j].y, 4);
				file.write((char *)&cell.hv[j].z, 4);
				file.write((char *)&cell.v[j].x,  4);
				file.write((char *)&cell.v[j].y,  4);
				file.write((char *)&cell.v[j].z,  4);
				// additional code 
				file.write((char *)&cell.a[j].x,  4);
				file.write((char *)&cell.a[j].y,  4);
				file.write((char *)&cell.a[j].z,  4);
				file.write((char *)&cell.density[j],  4);
			    // ends here
            }
			++count;
		}
	}
	assert(count == numParticles);

	int numSkipped = origNumParticles - numParticles;
	float zero = 0.f;
    if(!isLittleEndian()) {
		zero = bswap_float(zero);
    }
	for(int i = 0; i < numSkipped; ++i)
	{
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
	}
}

/**
 * flushes the and resets the particles in each grid.
 * Added by: Jahanzeb Maqbool
*/
void ClearParticlesMT(int rank)
{
	for(int iz = grids[rank].sz; iz < grids[rank].ez; ++iz)
		for(int iy = grids[rank].sy; iy < grids[rank].ey; ++iy)
			for(int ix = grids[rank].sx; ix < grids[rank].ex; ++ix)
			{
                             int localIndex = getLocalIndex(grids[rank], ix, iy, iz);
                	     //int index = (iz*ny + iy)*nx + ix;
			     cnumPars_sub[localIndex] = 0;
			}
}


/**
 * return true/false based on whether current particle position lies in its own grid or not
 * Added by: Jahanzeb Maqbool
*/
bool isLocalGridCell (Grid grid, int ci, int cj, int ck) {
  
    if ( (grid.sx <= ci && ci < grid.ex) && (grid.sy <= cj && cj < grid.ey) &&(grid.sz <= ck && ck < grid.ez) )
       return true;
    else
       return false;

}

/**
 * A utility method that calculates the Grid index (partition index).
 * Added by: Jahanzeb Maqbool
*/
int calculateGridIndex (int ci, int cj, int ck) {

    for (int i = 0; i < GRIDS_SIZE; i++) {
       if (isLocalGridCell (grids[i], ci, cj, ck) )
            return i;
    }
 
  return -1;
}


/**
 * Defined a manual MPI datatype for Cell structure.
 * Added by: Jahanzeb Maqbool
*/
void getSingleCellDatatype (){

  // defining own data type of cellDataType...
  MPI_Datatype oldtypes_ [2];
  int blockcounts_ [2];
  MPI_Aint offsets_ [2], extent;
  // setting up description for 4*16*Vec3 in gridDataType...
   offsets_ [0] = 0;
   oldtypes_ [0] = vecDataType;
   blockcounts_ [0] = 3;
   //MPI_Type_extent (vecDataType, &extent);
   extent = sizeof (Vec3);
   offsets_ [1] = 3 * extent;
   oldtypes_ [1] = MPI_INT;
   blockcounts_ [1] = 3;
   /* Now define structured type and commit it */
   MPI_Type_struct (2, blockcounts_, offsets_, oldtypes_, &singleCellDataType);
   MPI_Type_commit (&singleCellDataType);

 //return singleCellDataType;
}



/**
 * A utility function to get the cartesian mapping of the grids.
 * Original Code by: Christina Bienna
 * Modified by: Jahanzeb Maqbool
*/
////////////////////////////////////////////////////////////////////////////////
void RebuildGridMT(int rank)
{
	for(int iz = grids[rank].sz; iz < grids[rank].ez; ++iz)
		for(int iy = grids[rank].sy; iy < grids[rank].ey; ++iy)
			for(int ix = grids[rank].sx; ix < grids[rank].ex; ++ix)
			{
                		//int index = (iz*ny + iy)*nx + ix;
				int index = getLocalIndex (grids[rank], ix, iy, iz);
				Cell const &cell2 = subCells2[index];
				int np2 = cnumPars2_sub[index];
				for(int j = 0; j < np2; ++j)
				{
					int ci = (int)((cell2.p[j].x - domainMin.x) / delta.x);
					int cj = (int)((cell2.p[j].y - domainMin.y) / delta.y);
					int ck = (int)((cell2.p[j].z - domainMin.z) / delta.z);

					if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
					if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
					if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;

					int index2 = (ck*ny + cj)*nx + ci; // index of cell array....
                                        
					// this assumes that particles cannot travel more than one grid cell per time step
					int np;
						
					if (isLocalGridCell) {  // simple copy...
                       int index2Local = getLocalIndex (grids[rank], ci, cj, ck);
     				   np = cnumPars_sub[index2Local]++;		
 					   Cell &cell = subCells[index2Local];
					   cell.p[np].x = cell2.p[j].x;
					   cell.p[np].y = cell2.p[j].y;
					   cell.p[np].z = cell2.p[j].z;
					   cell.hv[np].x = cell2.hv[j].x;
					   cell.hv[np].y = cell2.hv[j].y;
					   cell.hv[np].z = cell2.hv[j].z;
					   cell.v[np].x = cell2.v[j].x;
					   cell.v[np].y = cell2.v[j].y;
					   cell.v[np].z = cell2.v[j].z;
                                       }
				       else { // need to move particle to other grid...
                                          
					 int peerGrid = calculateGridIndex (ci, cj, ck);
					 SingleCell cell;
					 cell.p.x = cell2.p[j].x;
					 cell.p.y = cell2.p[j].y;
					 cell.p.z = cell2.p[j].z;
					 cell.hv.x = cell2.hv[j].x;
					 cell.hv.y = cell2.hv[j].y;
					 cell.hv.z = cell2.hv[j].z;
					 cell.v.x = cell2.v[j].x;
					 cell.v.y = cell2.v[j].y;
					 cell.v.z = cell2.v[j].z;

					 swapList[peerGrid].push_back (cell);  // inserted into vector
					
				       }
 				 }
			}

           MPI_Request request;
           MPI_Status status;
           for (int i = 0; i < GRIDS_SIZE; i++) {
                if (i!=rank) {
                   int size = swapList[i].size();
				   if(size>0) {
						if (DEBUG)
						printf("Send swap list found on <%d> with size <%d>",rank,size);
					}
         	   MPI_Isend (&size, 1, MPI_INT, i, 200, MPI_COMM_WORLD, &request);
		   if (size > 0) {
	   	      SingleCell *sCells = new SingleCell [size];
		      for (int j = 0; j < size; j ++) 
			 sCells [j] = swapList[i][j];

                      MPI_Isend( sCells, size, singleCellDataType, i, 201, MPI_COMM_WORLD, &request);
		   }
                }
           }    

           for (int i = 0; i < GRIDS_SIZE; i++) {
                if (i!=rank) {
                   int size;
         	   MPI_Recv (&size, 1, MPI_INT, i, 200, MPI_COMM_WORLD, &status);
		   if (size > 0) {
	   	      SingleCell *sCells = new SingleCell [size];
			  if (DEBUG)	
				printf("Recv swap list found on <%d> with size <%d>",rank,size);
                      MPI_Recv( sCells, size, singleCellDataType, i, 201, MPI_COMM_WORLD, &status);
                      for (int j =0; j < size; j ++) {
                         int neighLocalIndex = getLocalIndex (grids[rank], sCells[j].ci, sCells[j].cj, sCells[j].ck);
                         int np = cnumPars_sub[neighLocalIndex]++;		
 			 Cell &cell = subCells[neighLocalIndex];
			 cell.p[np].x = sCells[j].p.x;
			 cell.p[np].y = sCells[j].p.y;
			 cell.p[np].z = sCells[j].p.z;
			 cell.hv[np].x = sCells[j].hv.x;
			 cell.hv[np].y = sCells[j].hv.y;
			 cell.hv[np].z = sCells[j].hv.z;
			 cell.v[np].x = sCells[j].v.x;
			 cell.v[np].y = sCells[j].v.y;
			 cell.v[np].z = sCells[j].v.z;
                      }
		   }
                }
           }    
}


/**
 * initializes the forces and densities vectors.
 * Added by: Jahanzeb Maqbool
*/
void InitDensitiesAndForcesMT(int rank)
{
	for(int iz = grids[rank].sz; iz < grids[rank].ez; ++iz)
		for(int iy = grids[rank].sy; iy < grids[rank].ey; ++iy)
			for(int ix = grids[rank].sx; ix < grids[rank].ex; ++ix)
			{
		        	//int index = (iz*ny + iy)*nx + ix; 
  				int index = getLocalIndex (grids[rank], ix, iy, iz);
				Cell &cell = subCells[index];
				int np = cnumPars_sub[index];
				for(int j = 0; j < np; ++j)
				{
					cell.density[j] = 0.f;
					cell.a[j] = externalAcceleration;
				}
			}
}
/**
 * A function to get the 3D cell coordinated from the single 1D index (1D-3D index mapping)
 * Added by: Jahanzeb Maqbool
*/
void calculate3dCellCoordinates (int index, int &cx, int &cy, int &cz) {
     cx = index % nx;
     cy = (index % (nx*ny)) / nx;
     cz = ceil(index / (nx*ny));
}

/**
 * A utility function to get the cartesian mapping of the grids.
 * Added by: Jahanzeb Maqbool
*/
int getCartesianRank (int myrank, int &myrank_x, int &myrank_z) {

	myrank_z = myrank % ZDIVS;
        myrank_x = myrank / ZDIVS;  
         
}

/**
 * A utility function to get the node Rank from cartesian mapping of the grids.
 * Added by: Jahanzeb Maqbool
*/
int getRankfromCartesian (int x, int z) {
   
	return (x*ZDIVS + z);
}

/**
 * A function to initialize the neighboring list in which cells to be transferred will be copied.
 * Added by: Jahanzeb Maqbool
*/
int InitNeighCellList(int ci, int cj, int ck, int *neighCells)
{
	int numNeighCells = 0;

	for(int di = -1; di <= 1; ++di)
		for(int dj = -1; dj <= 1; ++dj)
			for(int dk = -1; dk <= 1; ++dk)
			{
				int ii = ci + di;
				int jj = cj + dj;
				int kk = ck + dk;
				if(ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz)
				{
					int index = (kk*ny + jj)*nx + ii;
					//if(cnumPars[index] != 0)
					//{
						neighCells[numNeighCells] = index;
						++numNeighCells;
					//}
				}
			}

	return numNeighCells;
}


/**
 * A function to Fetch the the neighbour lists indices. (list of particles to be communicated across the nodes)
 * Added by: Jahanzeb Maqbool
*/
int getNeighborListIndex (int myRank, int neighRank){
	int myX, myZ, neighX, neighZ;
    getCartesianRank (myRank, myX, myZ);
	getCartesianRank (neighRank, neighX, neighZ);
	if (myX == neighX) {  // x boundry same
	    if (myZ < neighZ) 
			return 2;	//bottom
	    else 
            return 3;	//up
	}
    else if (myZ == neighZ){
	  if (myX < neighX)
		return 1;	//right
	  else
		return 0;	// left
	}
	else if( (myX<neighX) && (myZ<neighZ) ) {
		return 7;  	// bottom right
	}
	else if( (myX<neighX) && (myZ>neighZ) ) {
		return 5;  	// top right
	}	
	else if( (myX > neighX) && (myZ < neighZ) ) {
		return 6;  	// bottom left
	}
	else if( (myX > neighX) && (myZ > neighZ) ) {
		return 4;  	// top left
	}
	
	else 
		return -1;  // wrong..
		
		// cartesian topology : left, right, down, up
	   
}


/**
 * A function to Calculate the neighbour indices (nodes ranks).
 * Added by: Jahanzeb Maqbool
*/
int calcLocalIndexNeigh (int peerList, int peerRank, int cx, int cy, int cz, int peerListSizeX, int peerListSizeZ ) {

     int lIndex = -999;
     if (peerList == 0) {
	lIndex = ( cz - grids[peerRank].sz )*(grids[peerRank].ey - grids[peerRank].sy) + cy;
	if (lIndex >= peerListSizeX || (cx != grids[peerRank].ex-1)) {
//	  printf ("lIndex = %d,  	 
	  lIndex = -888;
	}
     }     
     if (peerList == 1) {
	 lIndex = (cz - grids[peerRank].sz )*(grids[peerRank].ey - grids[peerRank].sy) + cy;
	if (lIndex >= peerListSizeX || (cx != grids[peerRank].sx))
	  lIndex = -888;
     }
     if (peerList == 2) {
	 lIndex = ( cy - grids[peerRank].sy)*(grids[peerRank].ex - grids[peerRank].sx) + cx;
	if (lIndex >= peerListSizeZ || (cz != grids[peerRank].sz))
	  lIndex = -888;
     }

     if (peerList == 3) {
	lIndex = (cy - grids[peerRank].sy)*(grids[peerRank].ex - grids[peerRank].sx) + cx;
	if (lIndex >= peerListSizeZ || (cz != grids[peerRank].ez-1))
	  lIndex = -888;
     }
	if (peerList > 3 &&  peerList < 8) {
	lIndex = cy;
	}
	return lIndex;
	
}

/**
 * A function to Fetch the the neighbour boundaries.
 * Added by: Jahanzeb Maqbool
*/
void fetchNeighborBorders (int rank, Cell** &peerCellList,  int** &peerParticleCountList,  Cell** &peerCellListBak,  
							int** &peerParticleCountListBak,int &peerSizeX, int &peerSizeZ) {
     
	 	 
     int myX, myZ;
     MPI_Request request, request1;
     MPI_Status status;
     peerCellList = new Cell*[8];
     peerParticleCountList = new int* [8];
     peerCellListBak = new Cell*[8];
     peerParticleCountListBak = new int* [8];
	 
     peerSizeX = ny*(grids[rank].ez-grids[rank].sz);
     peerSizeZ = ny*(grids[rank].ex-grids[rank].sx);
	
	// boundries lists
     peerCellList[0] = new Cell [peerSizeX];	// left boundry list
     peerCellList[1] = new Cell [peerSizeX];	// Right boundry list
     peerCellList[2] = new Cell [peerSizeZ]; 	// bottom boundry list
     peerCellList[3] = new Cell [peerSizeZ];  	// top boundry list
	 peerCellList[4] = new Cell [ny];  			// top left corner list
	 peerCellList[5] = new Cell [ny];  			// top right corner list
	 peerCellList[6] = new Cell [ny];  			// bottom left corner list
	 peerCellList[7] = new Cell [ny];  			// bottom right corner list
	 
     peerCellListBak[0] = new Cell [peerSizeX];	// left boundry list
     peerCellListBak[1] = new Cell [peerSizeX];	// Right boundry list
     peerCellListBak[2] = new Cell [peerSizeZ]; 	// bottom boundry list
     peerCellListBak[3] = new Cell [peerSizeZ];  	// top boundry list
	 peerCellListBak[4] = new Cell [ny];  			// top left corner list
	 peerCellListBak[5] = new Cell [ny];  			// top right corner list
	 peerCellListBak[6] = new Cell [ny];  			// bottom left corner list
	 peerCellListBak[7] = new Cell [ny];  			// bottom right corner list	

     peerParticleCountList[0] = new int [peerSizeX];
     peerParticleCountList[1] = new int [peerSizeX];
     peerParticleCountList[2] = new int [peerSizeZ]; 
     peerParticleCountList[3] = new int [peerSizeZ];  
  	 peerParticleCountList[4] = new int [ny];  			// top left corner list
	 peerParticleCountList[5] = new int [ny];  			// top right corner list
	 peerParticleCountList[6] = new int [ny];  			// bottom left corner list
	 peerParticleCountList[7] = new int [ny];  			// bottom right corner list
	 
     peerParticleCountListBak[0] = new int [peerSizeX];
     peerParticleCountListBak[1] = new int [peerSizeX];
     peerParticleCountListBak[2] = new int [peerSizeZ]; 
     peerParticleCountListBak[3] = new int [peerSizeZ];  
  	 peerParticleCountListBak[4] = new int [ny];  			// top left corner list
	 peerParticleCountListBak[5] = new int [ny];  			// top right corner list
	 peerParticleCountListBak[6] = new int [ny];  			// bottom left corner list
	 peerParticleCountListBak[7] = new int [ny];  			// bottom right corner list
	 
     getCartesianRank (rank, myX, myZ);

	 if (DEBUG) {
		 printf ("rank<%d>, peerSizeX = %d\n", rank, peerSizeX);
		 printf ("rank<%d>, peerSizeZ = %d\n", rank, peerSizeZ);
	}
     //////////////////// For List 1
    if (myX != 0) {
		int index = 0;
        for (int iz = grids[rank].sz; iz < grids[rank].ez; iz++) 
           for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++) {
		//int gIndex = (iz*ny + iy)*nx + grids[rank].sx;
		int lIndex = getLocalIndex (grids[rank], grids[rank].sx, iy, iz);			      
		peerCellListBak[1][index] = subCells[lIndex];
		peerParticleCountListBak[1][index] = cnumPars_sub[lIndex];
		
	   index ++;
	   }
    	
        MPI_Send (peerCellListBak[1], peerSizeX, cellDataType, getRankfromCartesian(myX-1, myZ), 1000, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountListBak[1], peerSizeX, MPI_INT, getRankfromCartesian(myX-1, myZ), 1001, MPI_COMM_WORLD);
     }
      if ((myX+1) != XDIVS){
		MPI_Recv (peerCellList[1], peerSizeX, cellDataType, getRankfromCartesian (myX+1, myZ), 1000, MPI_COMM_WORLD, &status);
		MPI_Recv (peerParticleCountList[1], peerSizeX, MPI_INT, getRankfromCartesian (myX+1, myZ), 1001, MPI_COMM_WORLD, &status);
	
	
    }
	  
    // For List 0
    if ((myX+1) != XDIVS){
		int index = 0;
        for (int iz = grids[rank].sz; iz < grids[rank].ez; iz++) {
           for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++) {
				//int gIndex = (iz*ny + iy)*nx + grids[rank].sx;
				int lIndex = getLocalIndex (grids[rank], grids[rank].ex-1, iy, iz);			      
				peerCellListBak[0][index] = subCells[lIndex];
				peerParticleCountListBak[0][index] = cnumPars_sub[lIndex];
				index ++;
			}
    	}
        MPI_Send (peerCellListBak[0], peerSizeX, cellDataType, getRankfromCartesian(myX+1, myZ), 1100, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountListBak[0], peerSizeX, MPI_INT, getRankfromCartesian(myX+1, myZ), 1101, MPI_COMM_WORLD);
    }
    if (myX != 0) {
		MPI_Recv (peerCellList[0], peerSizeX, cellDataType, getRankfromCartesian (myX-1, myZ), 1100, MPI_COMM_WORLD, &status);
		MPI_Recv (peerParticleCountList[0], peerSizeX, MPI_INT, getRankfromCartesian (myX-1, myZ), 1101, MPI_COMM_WORLD, &status);
    }

    // For List 2
    if (myZ != 0) {
		int index = 0;
		for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++)	
			for (int ix = grids[rank].sx; ix < grids[rank].ex; ix++)  {
				//int gIndex = (iz*ny + iy)*nx + grids[rank].sx;
				int lIndex = getLocalIndex (grids[rank],ix, iy, grids[rank].sz);			      
				peerCellListBak[2][index] = subCells[lIndex];
				peerParticleCountListBak[2][index] = cnumPars_sub[lIndex];
				
				index ++;
		   }
			
		MPI_Send (peerCellListBak[2], peerSizeZ, cellDataType, getRankfromCartesian(myX, myZ-1), 1200, MPI_COMM_WORLD);
		MPI_Send (peerParticleCountListBak[2], peerSizeZ, MPI_INT, getRankfromCartesian(myX, myZ-1), 1201, MPI_COMM_WORLD);
    }
    if ((myZ+1) != ZDIVS){
		MPI_Recv (peerCellList[2], peerSizeZ, cellDataType, getRankfromCartesian (myX, myZ+1), 1200 ,MPI_COMM_WORLD,  &status);
		MPI_Recv (peerParticleCountList[2], peerSizeZ, MPI_INT, getRankfromCartesian (myX, myZ+1),1201 ,MPI_COMM_WORLD, &status);
    }
   
   
   // For List 3
    if ((myZ+1) != ZDIVS){
		int index = 0;
        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++) {
	        for (int ix = grids[rank].sx; ix < grids[rank].ex; ix++)  {
				int lIndex = getLocalIndex (grids[rank],ix, iy, grids[rank].ez-1);			      
				peerCellListBak[3][index] = subCells[lIndex];
				peerParticleCountListBak[3][index] = cnumPars_sub[lIndex];
				index ++;
				
				if (lIndex >= chunkSize) {
					if (DEBUG)
					printf ("<%d>, Invalid Index when copying data for fetching..= %d\n", rank);	
				}
			}
    	}
        MPI_Send (peerCellListBak[3], peerSizeZ, cellDataType, getRankfromCartesian(myX, myZ+1), 1300, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountListBak[3], peerSizeZ, MPI_INT, getRankfromCartesian(myX, myZ+1), 1301, MPI_COMM_WORLD);
    }
	 
    if (myZ != 0) {
		MPI_Recv (peerCellList[3], peerSizeZ, cellDataType, getRankfromCartesian (myX, myZ-1), 1300, MPI_COMM_WORLD, &status);
		MPI_Recv (peerParticleCountList[3], peerSizeZ, MPI_INT, getRankfromCartesian (myX, myZ-1), 1301, MPI_COMM_WORLD, &status);	 
    }
	  
	  
	// For List 4 
    if ( (myZ < (ZDIVS-1)) && (myX < (XDIVS-1) ) ){// all process except last row and last col
		int index = 0;
        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++){
		int lIndex = getLocalIndex (grids[rank],grids[rank].ex-1, iy, grids[rank].ez-1);			      
		peerCellListBak[4][index] = subCells[lIndex];
		peerParticleCountListBak[4][index] = cnumPars_sub[lIndex];
		index ++;
		
		if (lIndex >= chunkSize) {
			if (DEBUG)
			printf ("<%d>, Invalid Index when copying data for fetching..= %d\n", rank);	
		}
	   }
    	
        MPI_Send (peerCellListBak[4], ny, cellDataType, getRankfromCartesian(myX+1, myZ+1), 1400, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountListBak[4], ny, MPI_INT, getRankfromCartesian(myX+1, myZ+1), 1401, MPI_COMM_WORLD);
    }
	 
    if ( (myZ > 0) && (myX > 0) ) {
		MPI_Recv (peerCellList[4], ny, cellDataType, getRankfromCartesian (myX-1, myZ-1), 1400, MPI_COMM_WORLD, &status);
		MPI_Recv (peerParticleCountList[4], ny, MPI_INT, getRankfromCartesian (myX-1, myZ-1), 1401, MPI_COMM_WORLD, &status);
    }	
	  
	  
	// For List 5	  
    if ( (myZ < (ZDIVS-1) ) && (myX > 0) ) {				// all process except last row and first col

		int index = 0;
        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++){
		int lIndex = getLocalIndex (grids[rank],grids[rank].sx, iy, grids[rank].ez-1);			      
		peerCellListBak[5][index] = subCells[lIndex];
		peerParticleCountListBak[5][index] = cnumPars_sub[lIndex];
		index ++;
		
		if (lIndex >= chunkSize) {
			if (DEBUG)
			printf ("<%d>, Invalid Index when copying data for fetching..= %d\n", rank);	
		}
	   }
    	
        MPI_Send (peerCellListBak[5], ny, cellDataType, getRankfromCartesian(myX-1, myZ+1), 1500, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountListBak[5], ny, MPI_INT, getRankfromCartesian(myX-1, myZ+1), 1501, MPI_COMM_WORLD);
     }
     if ( (myZ > 0) && (myX < XDIVS-1) ) {
		MPI_Recv (peerCellList[5], ny, cellDataType, getRankfromCartesian (myX+1, myZ-1), 1500, MPI_COMM_WORLD, &status);
		MPI_Recv (peerParticleCountList[5], ny, MPI_INT, getRankfromCartesian (myX+1, myZ-1), 1501, MPI_COMM_WORLD, &status);
      } 
	  
	  
	 // For List 6 
    if ( (myX < (XDIVS-1) ) && (myZ > 0) ){				// all process except first row and last col
		int index = 0;
        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++){
			int lIndex = getLocalIndex (grids[rank],grids[rank].ex-1, iy, grids[rank].sz);			      
			peerCellListBak[6][index] = subCells[lIndex];
			peerParticleCountListBak[6][index] = cnumPars_sub[lIndex];
			index ++;
			
			if (lIndex >= chunkSize) {
				if (DEBUG)
				printf ("<%d>, Invalid Index when copying data for fetching..= %d\n", rank);	
			}
	    }
    	
        MPI_Send (peerCellListBak[6], ny, cellDataType, getRankfromCartesian(myX+1, myZ-1), 1600, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountListBak[6], ny, MPI_INT, getRankfromCartesian(myX+1, myZ-1), 1601, MPI_COMM_WORLD);
    }
    if ( (myX > 0) && (myZ < ZDIVS-1) ) {
		MPI_Recv (peerCellList[6], ny, cellDataType, getRankfromCartesian (myX-1, myZ+1), 1600, MPI_COMM_WORLD, &status);
		MPI_Recv (peerParticleCountList[6], ny, MPI_INT, getRankfromCartesian (myX-1, myZ+1), 1601, MPI_COMM_WORLD, &status);
    } 	  
	  
    if ( (myZ >0 ) && (myX > 0) ){				// all process except first row and first col
		int index = 0;
        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++){
	     
			int lIndex = getLocalIndex (grids[rank],grids[rank].sx, iy, grids[rank].sz);			      
			peerCellListBak[7][index] = subCells[lIndex];
			peerParticleCountListBak[7][index] = cnumPars_sub[lIndex];
			index ++;
			
			if (lIndex >= chunkSize) {
				if (DEBUG)
				printf ("<%d>, Invalid Index when copying data for fetching..= %d\n", rank);	
			}
  	    }
    	
        MPI_Send (peerCellListBak[7], ny, cellDataType, getRankfromCartesian(myX-1, myZ-1), 1700, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountListBak[7], ny, MPI_INT, getRankfromCartesian(myX-1, myZ-1), 1701, MPI_COMM_WORLD);
    }
    if ( (myX < XDIVS-1) && (myZ < ZDIVS-1) ) {
		MPI_Recv (peerCellList[7], ny, cellDataType, getRankfromCartesian (myX+1, myZ+1), 1700, MPI_COMM_WORLD, &status);
		MPI_Recv (peerParticleCountList[7], ny, MPI_INT, getRankfromCartesian (myX+1, myZ+1), 1701, MPI_COMM_WORLD, &status);
    } 	  
	
}


/**
 * Updates the neighbor boundaries as the result of acting forces on particles.
 * Added by: Jahanzeb Maqbool
*/
void updateNeighborBorders (int rank, Cell** &peerCellList,  int** &peerParticleCountList, Cell** &peerCellListBak,  int** &peerParticleCountListBak,int &peerSizeX, int &peerSizeZ) {

    int myX, myZ;
    MPI_Request request, request1;
    MPI_Status status;
    Cell** temp_peerCellList = new Cell*[3];
    int ** temp_peerParticleCountList = new int* [3];

    temp_peerCellList[0] = new Cell [peerSizeX];
    temp_peerCellList[1] = new Cell [peerSizeZ];
	temp_peerCellList[2] = new Cell [ny];

    temp_peerParticleCountList[0] = new int [peerSizeX];
    temp_peerParticleCountList[1] = new int [peerSizeZ];
  	temp_peerParticleCountList[2] = new int [ny];
    getCartesianRank (rank, myX, myZ);
     	
    // Updating Right Neighbour....
    if ((myX+1) != XDIVS){
       MPI_Send (peerCellList[1], peerSizeX, cellDataType, getRankfromCartesian(myX+1, myZ), 2000, MPI_COMM_WORLD);
       MPI_Send (peerParticleCountList[1], peerSizeX, MPI_INT, getRankfromCartesian(myX+1, myZ), 2001, MPI_COMM_WORLD);
    }
    if (myX != 0) {
		int index = 0;
		MPI_Recv (temp_peerCellList[0], peerSizeX, cellDataType, getRankfromCartesian (myX-1, myZ), 2000, MPI_COMM_WORLD, &status);
		MPI_Recv (temp_peerParticleCountList[0], peerSizeX, MPI_INT, getRankfromCartesian (myX-1, myZ), 2001, MPI_COMM_WORLD, &status);
			for (int iz = grids[rank].sz; iz < grids[rank].ez; iz++) 
			   for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++) {
			//int gIndex = (iz*ny + iy)*nx + grids[rank].sx;
			int lIndex = getLocalIndex (grids[rank], grids[rank].sx, iy, iz);
			cnumPars_sub[lIndex] += (temp_peerParticleCountList[0][index]-peerParticleCountListBak[1][index]);		
			subCells[lIndex] = addCell(subCells[lIndex] , diffCell(temp_peerCellList[0][index],peerCellListBak[1][index],cnumPars_sub[lIndex]) ,cnumPars_sub[lIndex]);

			index++;
		} 
    }
     // Updating Left Neighbour.... 
    if (myX != 0) {
       MPI_Send (peerCellList[0], peerSizeX, cellDataType, getRankfromCartesian(myX-1, myZ), 2100, MPI_COMM_WORLD);
       MPI_Send (peerParticleCountList[0], peerSizeX, MPI_INT, getRankfromCartesian(myX-1, myZ), 2101, MPI_COMM_WORLD);
    }

    if ((myX+1) != XDIVS){
		int index = 0;
		MPI_Recv (temp_peerCellList[0], peerSizeX, cellDataType, getRankfromCartesian (myX+1, myZ), 2100, MPI_COMM_WORLD, &status);
		MPI_Recv (temp_peerParticleCountList[0], peerSizeX, MPI_INT, getRankfromCartesian (myX+1, myZ), 2101, MPI_COMM_WORLD, &status);
		for (int iz = grids[rank].sz; iz < grids[rank].ez; iz++) {
		   for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++) {
		   //int gIndex = (iz*ny + iy)*nx + grids[rank].sx;
		   int lIndex = getLocalIndex (grids[rank], grids[rank].ex-1, iy, iz);	
		   cnumPars_sub[lIndex] += (temp_peerParticleCountList[0][index]-peerParticleCountListBak[0][index]);		
		   subCells[lIndex] = addCell(subCells[lIndex] , diffCell(temp_peerCellList[0][index],peerCellListBak[0][index],cnumPars_sub[lIndex]) ,cnumPars_sub[lIndex]);
		   index++;
		   }
		}
    }

     // Updating Down Neighbour.... 
    if ((myZ+1) != ZDIVS){
        MPI_Send (peerCellList[2], peerSizeZ, cellDataType, getRankfromCartesian(myX, myZ+1), 2200, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountList[2], peerSizeZ, MPI_INT, getRankfromCartesian(myX, myZ+1), 2201, MPI_COMM_WORLD);
     }

    if (myZ != 0) {
		int index = 0;
		MPI_Recv (temp_peerCellList[1], peerSizeZ, cellDataType, getRankfromCartesian (myX, myZ-1), 2200, MPI_COMM_WORLD, &status);
		MPI_Recv (temp_peerParticleCountList[1], peerSizeZ, MPI_INT, getRankfromCartesian (myX, myZ-1), 2201, MPI_COMM_WORLD, &status);
        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++) {
	        for (int ix = grids[rank].sx; ix < grids[rank].ex; ix++)  {
				//int gIndex = (iz*ny + iy)*nx + grids[rank].sx;
				int lIndex = getLocalIndex (grids[rank],ix, iy, grids[rank].sz);	
				cnumPars_sub[lIndex] += (temp_peerParticleCountList[1][index]-peerParticleCountListBak[2][index]);		
				subCells[lIndex] = addCell(subCells[lIndex] , diffCell(temp_peerCellList[1][index],peerCellListBak[2][index],cnumPars_sub[lIndex]) ,cnumPars_sub[lIndex]);
				index++;
		    }
		}
    }

    // Updating Up Neighbour.... 
    if (myZ != 0) {
        MPI_Send (peerCellList[3], peerSizeZ, cellDataType, getRankfromCartesian(myX, myZ-1), 2300, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountList[3], peerSizeZ, MPI_INT, getRankfromCartesian(myX, myZ-1), 2301, MPI_COMM_WORLD);
    }
    if ((myZ+1) != ZDIVS){
		int index = 0;
		MPI_Recv (temp_peerCellList[1], peerSizeZ, cellDataType, getRankfromCartesian (myX, myZ+1), 2300, MPI_COMM_WORLD, &status);
		MPI_Recv (temp_peerParticleCountList[1], peerSizeZ, MPI_INT, getRankfromCartesian (myX, myZ+1), 2301, MPI_COMM_WORLD, &status);
        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++) {
	        for (int ix = grids[rank].sx; ix < grids[rank].ex; ix++)  {
				//int gIndex = (iz*ny + iy)*nx + grids[rank].sx;
				int lIndex = getLocalIndex (grids[rank],ix, iy, grids[rank].ez-1);
				cnumPars_sub[lIndex] += (temp_peerParticleCountList[1][index]-peerParticleCountListBak[3][index]);		
				subCells[lIndex] = addCell(subCells[lIndex] , diffCell(temp_peerCellList[1][index],peerCellListBak[3][index],cnumPars_sub[lIndex]) ,cnumPars_sub[lIndex]);
				index++;
			}
		}
    }
	
    // Updating TOP LEFT  Neighbour.... 	  
     if ( (myZ < (ZDIVS-1)) && (myX < (XDIVS-1) ) ) {
	
		MPI_Recv (temp_peerCellList[2], ny, cellDataType, getRankfromCartesian (myX+1, myZ+1), 2400, MPI_COMM_WORLD, &status);
		MPI_Recv (temp_peerParticleCountList[2], ny, MPI_INT, getRankfromCartesian (myX+1, myZ+1), 2401, MPI_COMM_WORLD, &status);

        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++) {
			int lIndex = getLocalIndex (grids[rank],grids[rank].ex-1, iy, grids[rank].ez-1);			      
			cnumPars_sub[lIndex] += (temp_peerParticleCountList[2][iy]-peerParticleCountListBak[4][iy]);		
			subCells[lIndex] = addCell(subCells[lIndex] , diffCell(temp_peerCellList[2][iy],peerCellListBak[4][iy],cnumPars_sub[lIndex]) ,cnumPars_sub[lIndex]);
		}	
     }
    if ( (myZ > 0) && (myX > 0) ) {
         MPI_Send (peerCellList[4], ny, cellDataType, getRankfromCartesian(myX-1, myZ-1), 2400, MPI_COMM_WORLD);
         MPI_Send (peerParticleCountList[4], ny, MPI_INT, getRankfromCartesian(myX-1, myZ-1), 2401, MPI_COMM_WORLD);

    }	
		  
    if ( (myZ < (ZDIVS-1) ) && (myX > 0) ) {				

		MPI_Recv (temp_peerCellList[2], ny, cellDataType, getRankfromCartesian (myX-1, myZ+1), 2500, MPI_COMM_WORLD, &status);
		MPI_Recv (temp_peerParticleCountList[2], ny, MPI_INT, getRankfromCartesian (myX-1, myZ+1), 2501, MPI_COMM_WORLD, &status);

        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++){
			int lIndex = getLocalIndex (grids[rank],grids[rank].sx, iy, grids[rank].ez-1);			      
			cnumPars_sub[lIndex] += (temp_peerParticleCountList[2][iy]-peerParticleCountListBak[5][iy]);		
			subCells[lIndex] = addCell(subCells[lIndex] , diffCell(temp_peerCellList[2][iy],peerCellListBak[5][iy],cnumPars_sub[lIndex]) ,cnumPars_sub[lIndex]);
		}	
    }
    if ( (myZ > 0) && (myX < XDIVS-1) ) {
        MPI_Send (peerCellList[5], ny, cellDataType, getRankfromCartesian(myX+1, myZ-1), 2500, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountList[5], ny, MPI_INT, getRankfromCartesian(myX+1, myZ-1), 2501, MPI_COMM_WORLD);
    }	
	  //////////////////// For List 6 ////////////////////
	  
    if ( (myX < (XDIVS-1) ) && (myZ > 0) ){				// all process except first row and last col

		MPI_Recv (temp_peerCellList[2], ny, cellDataType, getRankfromCartesian (myX+1, myZ-1), 2600, MPI_COMM_WORLD, &status);
		MPI_Recv (temp_peerParticleCountList[2], ny, MPI_INT, getRankfromCartesian (myX+1, myZ-1), 2601, MPI_COMM_WORLD, &status);

        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++) {
			int lIndex = getLocalIndex (grids[rank],grids[rank].ex-1, iy, grids[rank].sz);			      
			cnumPars_sub[lIndex] += (temp_peerParticleCountList[2][iy]-peerParticleCountListBak[6][iy]);		
			subCells[lIndex] = addCell(subCells[lIndex] , diffCell(temp_peerCellList[2][iy],peerCellListBak[6][iy],cnumPars_sub[lIndex]) ,cnumPars_sub[lIndex]);
		}  	
    }
    if ( (myX > 0) && (myZ < ZDIVS-1) ) {
        MPI_Send (peerCellList[6], ny, cellDataType, getRankfromCartesian(myX-1, myZ+1), 2600, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountList[6], ny, MPI_INT, getRankfromCartesian(myX-1, myZ+1), 2601, MPI_COMM_WORLD);

    }	  

	//////////////////// For List 7 ////////////////////  
    
	if ( (myZ >0 ) && (myX > 0) ){				// all process except first row and first col

		MPI_Recv (temp_peerCellList[2], ny, cellDataType, getRankfromCartesian (myX-1, myZ-1), 2700, MPI_COMM_WORLD, &status);
		MPI_Recv (temp_peerParticleCountList[2], ny, MPI_INT, getRankfromCartesian (myX-1, myZ-1), 2701, MPI_COMM_WORLD, &status);
        for (int iy = grids[rank].sy; iy < grids[rank].ey; iy ++){
			int lIndex = getLocalIndex (grids[rank],grids[rank].sx, iy, grids[rank].sz);
			cnumPars_sub[lIndex] += (temp_peerParticleCountList[2][iy]-peerParticleCountListBak[7][iy]);		
			subCells[lIndex] = addCell(subCells[lIndex] , diffCell(temp_peerCellList[2][iy],peerCellListBak[7][iy],cnumPars_sub[lIndex]) ,cnumPars_sub[lIndex]);
		}  	
    }
    if ( (myX < XDIVS-1) && (myZ < ZDIVS-1) ) {
        MPI_Send (peerCellList[7], ny, cellDataType, getRankfromCartesian(myX+1, myZ+1), 2700, MPI_COMM_WORLD);
        MPI_Send (peerParticleCountList[7], ny, MPI_INT, getRankfromCartesian(myX+1, myZ+1), 2701, MPI_COMM_WORLD);

    }
    MPI_Barrier (MPI_COMM_WORLD);
    for (int x = 0; x < 2; x ++) {
		delete temp_peerCellList [x];
		delete temp_peerParticleCountList[x];
    }
	
    delete temp_peerCellList;
    delete temp_peerParticleCountList;
    temp_peerCellList = NULL;
    temp_peerParticleCountList = NULL;

}

/**
 * Utility that copies cells.
 * Added by: Jahanzeb Maqbool
*/
void copyData(Cell* origionalcells,int* origionalnos,Cell* backupcells,int* backupnos){
	int iter=0;
	if (DEBUG)
		printf("In CopyData \n");
	for(iter=0;iter<chunkSize;iter++) {
		backupcells[iter] = origionalcells[iter];
		backupnos[iter] = origionalnos[iter];
	}
}

/**
 * Computing Densitites of tha particles.
 * Original Author: Christian Bienia
 * Added by: Jahanzeb Maqbool
*/
void ComputeDensitiesMT(int rank)
{
	int neighCells[27];
	Cell** peerCellList;
	Cell** peerCellListBak;
	int** peerParticleCountList;
	int** peerParticleCountListBak;
	int peerListSizeX, peerListSizeZ;
		
	fetchNeighborBorders (rank, peerCellList, peerParticleCountList, peerCellListBak, peerParticleCountListBak,peerListSizeX, peerListSizeZ); // to be changed...ok
	
	if (DEBUG)
		printf ("<%d>, In compute dendisties after fetching... \n", rank);  
		
		
	MPI_Barrier (MPI_COMM_WORLD);

	for(int iz = grids[rank].sz; iz < grids[rank].ez; ++iz)
		for(int iy = grids[rank].sy; iy < grids[rank].ey; ++iy)
			for(int ix = grids[rank].sx; ix < grids[rank].ex; ++ix)
			{
                int globalIndex = (iz*ny + iy)*nx + ix;
						
				int index = getLocalIndex (grids[rank], ix, iy, iz);
				int np = cnumPars_sub[index];
				if(np == 0)
				    continue;

				int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);
				Cell &cell = subCells[index];

				for(int j = 0; j < np; ++j)
					for(int inc = 0; inc < numNeighCells; ++inc) {
						int indexNeigh = neighCells[inc];
                        int cx, cy, cz;

                        calculate3dCellCoordinates (indexNeigh, cx, cy, cz);
						int numNeighPars;
						
 						if ( isLocalGridCell (grids[rank], cx, cy, cz)  ){// not to update locals when updating neighbours
														
                            int localIndexNeigh = getLocalIndex (grids[rank], cx, cy, cz);
                            Cell &neigh = subCells[localIndexNeigh];
						    numNeighPars = cnumPars_sub[localIndexNeigh];

	  				        for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
							if(&neigh.p[iparNeigh] < &cell.p[j])
							{
								float distSq = (cell.p[j] - neigh.p[iparNeigh]).GetLengthSq();
								if(distSq < hSq)
								{
									float t = hSq - distSq;
									float tc = t*t*t;
									cell.density[j] += tc;
									neigh.density[iparNeigh] += tc;
								}
							}
							
						}
						else if(!isLocalGridCell (grids[rank], cx, cy, cz)  ){         
	
							int neighRank = calculateGridIndex (cx, cy, cz);								
						    int listIndex = getNeighborListIndex (rank, neighRank);  // topology index...		
							
							if (listIndex == -1 ) {//  || listIndex > 3
								printf ("<%d> NeighRank [%d] and listIndex is %d\n", rank, neighRank, listIndex);
								continue;
							}
							
						    // calculate localIndexNeigh from 4 peer list of listIndex.
							int localIndexNeigh = calcLocalIndexNeigh(listIndex, neighRank, cx, cy, cz, peerListSizeX, peerListSizeZ);	
							
					        Cell &neigh = peerCellList [listIndex][localIndexNeigh];
						    numNeighPars = peerParticleCountList [listIndex][localIndexNeigh];

	  				        for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
							if(&neigh.p[iparNeigh] < &cell.p[j]) {
								float distSq = (cell.p[j] - neigh.p[iparNeigh]).GetLengthSq();
								if(distSq < hSq) {
									float t = hSq - distSq;
									float tc = t*t*t;
									cell.density[j] += tc;
									neigh.density[iparNeigh] += tc;
								}								
							}
						}
					}

	

			}  // end of for grids[rank]
			
			updateNeighborBorders (rank, peerCellList, peerParticleCountList, peerCellListBak, peerParticleCountListBak,peerListSizeX, peerListSizeZ);	///// TO BE CHANGED,,, OK
		
			if (DEBUG)
				printf ("<%d>, In compute dendisties after calculating... \n", rank);  
      
	  
	   // Updating Neighbours Data.
	   for (int i = 0; i < 4; i ++) {
	       delete peerCellList[i];	
	       delete peerParticleCountList[i];
	   }
	   
	   
	   delete peerCellList;
	   delete peerParticleCountList;
	   peerCellList = NULL;
	   peerParticleCountList = NULL;
	
}

/**
 * Original Compute Densities Method
*/

void ComputeDensities2MT(int rank)
{
	const float tc = hSq*hSq*hSq;
	for(int iz = grids[rank].sz; iz < grids[rank].ez; ++iz)
		for(int iy = grids[rank].sy; iy < grids[rank].ey; ++iy)
			for(int ix = grids[rank].sx; ix < grids[rank].ex; ++ix)
			{
                		//int index = (iz*ny + iy)*nx + ix;
  				int index = getLocalIndex (grids[rank], ix, iy, iz);
				Cell &cell = subCells[index];
				int np = cnumPars_sub[index];
				for(int j = 0; j < np; ++j)
				{
					cell.density[j] += tc;
					cell.density[j] *= densityCoeff;
				}
			}
}

/**
 * Computes the result of forces of one particle onto another. Uses modification to the original algorithm
 * original author: Chistian Bienia
 * Added by: Jahanzeb Maqbool
*/
void ComputeForcesMT(int rank)
{
	int neighCells[27];
	Cell** peerCellList;
	Cell** peerCellListBak;
	int** peerParticleCountList;
	int** peerParticleCountListBak;
	int peerListSizeX, peerListSizeZ;
	fetchNeighborBorders (rank, peerCellList, peerParticleCountList, peerCellListBak, peerParticleCountListBak, peerListSizeX, peerListSizeZ);
	Cell* backupcells = new Cell[chunkSize];
	int*  backupnos   = new int[chunkSize];	

	for(int raceResolver = 0; raceResolver < 2; ++raceResolver){
		if(raceResolver==0)
			copyData(subCells,cnumPars_sub,backupcells,backupnos);
		if(raceResolver==1)
			copyData(backupcells,backupnos,subCells,cnumPars_sub);
		for(int iz = grids[rank].sz; iz < grids[rank].ez; ++iz)
		  for(int iy = grids[rank].sy; iy < grids[rank].ey; ++iy)
			for(int ix = grids[rank].sx; ix < grids[rank].ex; ++ix)
			{
		        int index = getLocalIndex (grids[rank], ix, iy, iz); //(iz*ny + iy)*nx + ix;
				int np = cnumPars_sub[index];
				if(np == 0)
	                continue;

				int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

				Cell &cell = subCells[index];

				for(int j = 0; j < np; ++j)
					for(int inc = 0; inc < numNeighCells; ++inc)
					{
						int indexNeigh = neighCells[inc];
						int cx, cy, cz;
                        calculate3dCellCoordinates (indexNeigh, cx, cy, cz);//
						// Cell &neigh = subCells[indexNeigh];
						int numNeighPars;
					
						if ( isLocalGridCell (grids[rank], cx, cy, cz)  ) {// not to update locals when updating neighbours
						    int localIndexNeigh = getLocalIndex (grids[rank], cx, cy, cz);
                            Cell &neigh = subCells[localIndexNeigh];
						    numNeighPars = cnumPars_sub[localIndexNeigh];

  						    for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
							if(&neigh.p[iparNeigh] < &cell.p[j])
							{
								Vec3 disp = cell.p[j] - neigh.p[iparNeigh];
								float distSq = disp.GetLengthSq();
								if(distSq < hSq)
								{
									float dist = sqrtf(std::max(distSq, 1e-12f));
									float hmr = h - dist;

									Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) * (cell.density[j]+neigh.density[iparNeigh] - doubleRestDensity);
									acc += (neigh.v[iparNeigh] - cell.v[j]) * viscosityCoeff * hmr;
									acc /= cell.density[j] * neigh.density[iparNeigh];

									cell.a[j] += acc;
									neigh.a[iparNeigh] -= acc;
								}
							}
						} // if localgrid ends..

						else if(!isLocalGridCell (grids[rank], cx, cy, cz) && (raceResolver==0) ) {         
	           
                            int neighRank = calculateGridIndex (cx, cy, cz);										
						    int listIndex = getNeighborListIndex (rank, neighRank);  // topology index...		
						    // calculate localIndexNeigh from 4 peer list of listIndex.
						    int localIndexNeigh = calcLocalIndexNeigh(listIndex,neighRank , cx, cy, cz, peerListSizeX, peerListSizeZ);
																					
					        Cell &neigh = peerCellList [listIndex][localIndexNeigh];
						    numNeighPars = peerParticleCountList [listIndex][localIndexNeigh];

						    for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
							if(&neigh.p[iparNeigh] < &cell.p[j])
							{
								Vec3 disp = cell.p[j] - neigh.p[iparNeigh];
								float distSq = disp.GetLengthSq();
								if(distSq < hSq)
								{
									float dist = sqrtf(std::max(distSq, 1e-12f));
									float hmr = h - dist;

									Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) * (cell.density[j]+neigh.density[iparNeigh] - doubleRestDensity);
									acc += (neigh.v[iparNeigh] - cell.v[j]) * viscosityCoeff * hmr;
									acc /= cell.density[j] * neigh.density[iparNeigh];

									cell.a[j] += acc;
									neigh.a[iparNeigh] -= acc;
								}
							}
						}  // else neighbourgrids data....

					}
			}

           // Updating Neighbours Data. 
	   if(raceResolver==0){
				updateNeighborBorders (rank, peerCellList, peerParticleCountList, peerCellListBak, peerParticleCountListBak, peerListSizeX, peerListSizeZ);	///// TO BE CHANGED,,, OK
			}
		}// resolver loop ends here
	   
	   	   
	   for (int i = 0; i < 4; i ++) {
	       delete peerCellList[i];	
	       delete peerParticleCountList[i];
	   }

	    delete peerCellList;
	    delete peerParticleCountList;
	    peerCellList = NULL;
	    peerParticleCountList = NULL;

}



/**
 * Processes the collision of particles in the 3-D subgrid
 * Modified by: Jahanzeb Maqbool
*/

void ProcessCollisionsMT(int rank)
{
	const float parSize = 0.0002f;
	const float epsilon = 1e-10f;
	const float stiffness = 30000.f;
	const float damping = 128.f;

	for(int iz = grids[rank].sz; iz < grids[rank].ez; ++iz)
		for(int iy = grids[rank].sy; iy < grids[rank].ey; ++iy)
			for(int ix = grids[rank].sx; ix < grids[rank].ex; ++ix)
			{
                		//int index = (iz*ny + iy)*nx + ix;
				int index = getLocalIndex (grids[rank], ix, iy, iz);
				Cell &cell = subCells[index];
				int np = cnumPars_sub[index];
				for(int j = 0; j < np; ++j)
				{
					Vec3 pos = cell.p[j] + cell.hv[j] * timeStep;
			
					float diff = parSize - (pos.x - domainMin.x);
					if(diff > epsilon)
						cell.a[j].x += stiffness*diff - damping*cell.v[j].x;

					diff = parSize - (domainMax.x - pos.x);
					if(diff > epsilon)
						cell.a[j].x -= stiffness*diff + damping*cell.v[j].x;

					diff = parSize - (pos.y - domainMin.y);
					if(diff > epsilon)
						cell.a[j].y += stiffness*diff - damping*cell.v[j].y;

					diff = parSize - (domainMax.y - pos.y);
					if(diff > epsilon)
						cell.a[j].y -= stiffness*diff + damping*cell.v[j].y;

					diff = parSize - (pos.z - domainMin.z);
					if(diff > epsilon)
						cell.a[j].z += stiffness*diff - damping*cell.v[j].z;

					diff = parSize - (domainMax.z - pos.z);
					if(diff > epsilon)
						cell.a[j].z -= stiffness*diff + damping*cell.v[j].z;
				}
			}
}



/**
 * Updates the position of particles in the entire grid (whether they need to change their local grid under the effect of forces). 
 * Next frame of simulation will use these position as its input.
 * Modified by: Jahanzeb Maqbool
*/
void AdvanceParticlesMT(int rank)
{
	for(int iz = grids[rank].sz; iz < grids[rank].ez; ++iz)
		for(int iy = grids[rank].sy; iy < grids[rank].ey; ++iy)
			for(int ix = grids[rank].sx; ix < grids[rank].ex; ++ix)
			{
                		//int index = (iz*ny + iy)*nx + ix;
				int index = getLocalIndex (grids[rank], ix, iy, iz);
				Cell &cell = subCells[index];
				int np = cnumPars_sub[index];
				for(int j = 0; j < np; ++j)
				{
					Vec3 v_half = cell.hv[j] + cell.a[j]*timeStep;
					cell.p[j] += v_half * timeStep;
					cell.v[j] = cell.hv[j] + v_half;
					cell.v[j] *= 0.5f;
					cell.hv[j] = v_half;
				}
			}
}
	


/**
 * Runs the simulation
*/
void AdvanceFrame(int rank)
{
//   printf ("<%d>, in advance Frame ...\n", rank);
//   printf ("<%d>, Single Cell Data2: %f, %f, %f \n", rank, subCells2[0].hv[0].x, subCells2[0].hv[0].y, subCells2[0].hv[0].z);	

  		
	ClearParticlesMT(rank);
	
	//MPI_Barrier (MPI_COMM_WORLD);
	RebuildGridMT(rank);

	//MPI_Barrier (MPI_COMM_WORLD);
	InitDensitiesAndForcesMT(rank);

	//MPI_Barrier (MPI_COMM_WORLD);
	ComputeDensitiesMT(rank); 
	
	//MPI_Barrier (MPI_COMM_WORLD);
	ComputeDensities2MT(rank); 
	
	//MPI_Barrier (MPI_COMM_WORLD);
	ComputeForcesMT(rank); 
	
	//MPI_Barrier (MPI_COMM_WORLD);
	ProcessCollisionsMT(rank);
	
	//MPI_Barrier (MPI_COMM_WORLD);
	AdvanceParticlesMT(rank);
	
	MPI_Barrier (MPI_COMM_WORLD);
}

void RunSimulation(int rank)
{	
        //thread_args *targs = (thread_args *)args;
	for(int i = 0; i < NUM_FRAMES; ++i)
	   AdvanceFrame(rank);

}


/**
 * Application Main Program, I modified this to add more MPI datastructures and temporary lists to support particle communication.
*/

main(int argc, char **argv)
{
  int rank, size, status;
  char* inputFile = argv [1]; 
  char* outputFile = argv [2];	

  
    struct timeval fread_start, fread_end, malloc_start, malloc_end, computation_start, computation_end, fwrite_start, fwrite_end;
    long fread_time_usec, fread_time_second, fwrite_time_usec, fwrite_time_second, computation_time_usec, computation_time_second, malloc_time_usec, malloc_time_second;
	double averageCompuationTime;
  
  
  gettimeofday (&computation_start, NULL);
  
  
  // MPI Initialization...
  MPI_Init(&argc,&argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size); 
  
  /*Define new Data types for Cell, Grid and Vec3 .... */
  // defining own data type of gridDataType...
  MPI_Datatype oldtypes [1];
  int blockcounts [1];
  MPI_Aint offsets [1];
  // setting up description for 6-ints in gridDataType...
   offsets [0] = 0;
   oldtypes [0] = MPI_INT;
   blockcounts [0] = 6;
 
   MPI_Type_struct (1, blockcounts, offsets, oldtypes, &gridDataType);
   MPI_Type_commit (&gridDataType);
  
  // defining own data type of vecDataType...
   offsets [0] = 0;
   oldtypes [0] = MPI_FLOAT;
   blockcounts [0] = 3;
 
   MPI_Type_contiguous (3, MPI_FLOAT, &vecDataType);
   //MPI_Type_struct (1, blockcounts, offsets, oldtypes, &vecDataType);
   MPI_Type_commit (&vecDataType);

  // defining own data type of cellDataType...
  MPI_Datatype oldtypes_ [2];
  int blockcounts_ [2];
  MPI_Aint offsets_ [2], extent;
  // setting up description for 4*16*Vec3 in gridDataType...
   offsets_ [0] = 0;
   oldtypes_ [0] = vecDataType;
   blockcounts_ [0] = 64;
   MPI_Type_extent (vecDataType, &extent);
   //extent = sizeof (Vec3);
   offsets_ [1] = 64 * extent;
   oldtypes_ [1] = MPI_FLOAT;
   blockcounts_ [1] = 16;
   /* Now define structured type and commit it */
   MPI_Type_struct (2, blockcounts_, offsets_, oldtypes_, &cellDataType);
   MPI_Type_commit (&cellDataType);

//  int n = -1; 
  if(rank==0) {
  //  printf ("input file is %s\n",inputFile); 


	
    InitSim (inputFile, size);		
    //displayLoadedTestData ();
    
    GRIDS_SIZE = NUM_GRIDS;
 //   printf ("Grid Size %d and n %d\n", NUM_GRIDS, GRIDS_SIZE);

   }

   MPI_Bcast( &XDIVS, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast( &ZDIVS, 1, MPI_INT, 0, MPI_COMM_WORLD);
   // broadcasting GRIDS_SIZE = size of Grid Array...
   MPI_Bcast( &nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
   // broadcasting GRIDS_SIZE = size of Grid Array...
   MPI_Bcast( &ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
   // broadcasting GRIDS_SIZE = size of Grid Array...
   MPI_Bcast( &nz, 1, MPI_INT, 0, MPI_COMM_WORLD);   
   // broadcasting GRIDS_SIZE = size of Grid Array...
   MPI_Bcast( &GRIDS_SIZE, 1, MPI_INT, 0, MPI_COMM_WORLD);
   // broadcasting delta...
   MPI_Bcast( &delta, 1, vecDataType, 0, MPI_COMM_WORLD);
   // broadcasting numCells...
   MPI_Bcast (&numCells, 1, MPI_INT, 0, MPI_COMM_WORLD);
   // broadcasting numCells...
   MPI_Bcast (&restParticlesPerMeter, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // broadcasting numCells...
   MPI_Bcast (&h, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   // broadcasting numCells...
   MPI_Bcast (&hSq, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   // broadcasting numCells...
   MPI_Bcast (&densityCoeff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     // broadcasting numCells...
   MPI_Bcast (&pressureCoeff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // broadcasting numCells...
   MPI_Bcast (&viscosityCoeff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   // broadcastin numParticles...
   MPI_Bcast (&numParticles, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
   
   if (DEBUG)
    printf ("<%d>, nx, ny, nz = %d, %d, %d \n", rank, nx, ny, nz);
  
  
   if (rank != 0)
      border = new bool [numCells];
   // broadcasting border....
   MPI_Bcast (border, numCells, MPI_BYTE, 0, MPI_COMM_WORLD);
   
   // create swapList array... (array of Vector)....
   swapList = new std::vector <SingleCell>[GRIDS_SIZE];  
 
   if (rank != 0) {
     grids = new struct Grid[GRIDS_SIZE];
   }    
   // Master is going to send grids data to all the slaves.
   if (rank==0) {
      int tag = 111;
      int slaves;
      for (slaves = 1; slaves < GRIDS_SIZE; slaves++) 
	MPI_Send(grids, GRIDS_SIZE, gridDataType, slaves, tag, MPI_COMM_WORLD);
   }
   // All the slaves are going to receive the grids data.
   if (rank!=0) {
      int tag = 111;
      MPI_Status status;
      MPI_Recv( grids, GRIDS_SIZE, gridDataType, 0, tag, MPI_COMM_WORLD, &status );
   } 


	if (DEBUG) {
	   printf ("Printing grids data \n");
	   printf ("grids[%d].sx = %d and grids[%d].ex = %d\n", rank, grids[rank].sx, rank, grids[rank].ex);
	   printf ("grids[%d].sy = %d and grids[%d].ey = %d\n", rank, grids[rank].sy, rank, grids[rank].ey);
	   printf ("grids[%d].sz = %d and grids[%d].ez = %d\n", rank, grids[rank].sz, rank, grids[rank].ez);
	}
 
 
   // Data Distribution Phase....
   // Step-01: Calculating Chunk Size....
 
   chunkSize = (grids[rank].ex-grids[rank].sx)*(grids[rank].ey-grids[rank].sy)*(grids[rank].ez-grids[rank].sz); // uniform distribution so take any grid like 0,1,2 ...

   subCells2 = new Cell [chunkSize];
   cnumPars2_sub = new int [chunkSize];
   subCells  = new Cell [chunkSize];
   cnumPars_sub = new int [chunkSize];

   
  //memset(cnumPars_sub, 0, numCells*sizeof(int));
  //memset(subCells, 0, numCells*sizeof(Cell));
  
   // Allocate memory to send buffer...
      sBuff = new Cell [chunkSize];
      cnumPars2_s = new int [chunkSize];


   int iChunk = 0; 

   int checkNumParticles = 0;
   if (rank==0) {

   // Send Data to other nodes...   
      bool temp = true;

      for(int i = 0; i < NUM_GRIDS; ++i) {
	 for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
	    for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
		for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
		{
			int index = (iz*ny + iy)*nx + ix;    

  			if (i==0) {
		            subCells2[iChunk] = cells2[index];   // Master node: Just a local copy, no need to send.
                            cnumPars2_sub[iChunk] = cnumPars2[index];
			    checkNumParticles += cnumPars2[index];
			    
                        }
			else {
                          /*if (temp) {
              		    //printf ("In else <%d> %f, %f, %f \n",index, cells2[index].hv[0].x, cells2[index].hv[0].y, cells2[index].hv[0].z);	           
  			    temp = false;
  			  }*/
			//printf ("<%d>", index);	           
			    sBuff[iChunk] = cells2[index];
                            cnumPars2_s[iChunk] = cnumPars2[index];
			    checkNumParticles += cnumPars2[index];
							
							//printf (" cnumPars Main %d\n", cnumPars2_s[iChunk]);
                        }
			
		    iChunk ++;
         	}
         
		 if (DEBUG)
			printf ("Rank 0 sending to rank<%d>, iChunk = %d and checkParticles is %d \n", i, iChunk, checkNumParticles );
         if (i!=0) { 
             
	 // Send Data to All nodes....
               //printf ("Before send %f, %f, %f \n", sBuff[0].hv[0].x, sBuff[0].hv[0].y, sBuff[0].hv[0].z);	    
               MPI_Send( sBuff, iChunk, cellDataType, i, 500, MPI_COMM_WORLD );
               MPI_Send( cnumPars2_s, iChunk, MPI_INT, i, 600, MPI_COMM_WORLD );  
			   
			   
         }
  	  // printf (" chunk sssss %d\n",iChunk) ;	
        	iChunk = 0;
		checkNumParticles  = 0;
      } 

    // free up sender memory...
    //delete sBuff;
    //delete cnumPars2_s;

   } // rank=0 ends...
   else {
	if (DEBUG)
		printf ("rank<%d>, chunkSize = %d \n",rank, chunkSize);
		
      MPI_Status status;
//printf ("<%d>, start receive grids data \n", rank);


        MPI_Recv (subCells2, chunkSize, cellDataType, 0, 500, MPI_COMM_WORLD, &status); 
//printf ("<%d>, received \n", rank);
        MPI_Recv (cnumPars2_sub, chunkSize, MPI_INT, 0, 600, MPI_COMM_WORLD, &status); 
				
//	printf("Particles when recieved are <%d> at local index <%d> \n",cnumPars2_sub[33027],33027);
   }
   
   
//gettimeofday (&computation_start, NULL);
   RunSimulation (rank);
gettimeofday (&computation_end, NULL);
   
	computation_time_usec = (double)(computation_end.tv_usec - computation_start.tv_usec);  
    computation_time_second = computation_end.tv_sec - computation_start.tv_sec; 
    long total_exec_time_ms = (computation_time_second*1000) + (computation_time_usec/1000);	
   
   if (DEBUG)
	printf("<%d> fluidanimate Computation Time    : %ld msec\n", rank, total_exec_time_ms);
	   
	   
	long *sendbuff, *recvbuff;
	int count = 1;
	sendbuff = (long*) malloc (count*sizeof(long));
	recvbuff = (long*) malloc (count*sizeof(long));
	sendbuff [0] = total_exec_time_ms;

//printf("<%d> BlackScholes Computation Time    : %f msec\n",rank, (float)computation_time/1000);

	MPI_Reduce( sendbuff, recvbuff, count, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD );
   
   
   
   if (DEBUG)
	printf ("<%d>, After run \n", rank);                      // ERROR OCCURING..........

   /////////////////////////////////////////////////// Gathering Data //////////////////////////////////////////

if (rank==0) {

     for(int i = 0; i < NUM_GRIDS; ++i) {

        iChunk = 0;
	//printf ("iChunk = %d and chunkSize = %d\n", iChunk, chunkSize);
        if (i!=0) { 
	 // Send Data to All nodes....
               //printf ("Before send %f, %f, %f \n", sBuff[0].hv[0].x, sBuff[0].hv[0].y, sBuff[0].hv[0].z);	    
               MPI_Recv( sBuff, chunkSize, cellDataType, i, 500, MPI_COMM_WORLD,  NULL);
               MPI_Recv( cnumPars2_s, chunkSize, MPI_INT, i, 600, MPI_COMM_WORLD,  NULL );  
         }
  	 	 
		 checkNumParticles = 0;
		 
	 for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
	    for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
			for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
			{
				int index = (iz*ny + iy)*nx + ix;    

				if (i==0) {
					cells[index] = subCells[iChunk];   // Master node: Just a local copy, no need to send.
					cnumPars[index] = cnumPars_sub[iChunk];
					checkNumParticles += cnumPars_sub[iChunk];
				}
				else {      
					cells[index] = sBuff[iChunk] ;
					cnumPars[index] = cnumPars2_s[iChunk];
					checkNumParticles += cnumPars2_s[iChunk];
				}
				
				iChunk ++;
				
			 }
			 if (DEBUG)
				printf ("checnNumPars after = %d \n", checkNumParticles);
			
		    } 
 
   } // rank=0 ends...
   else {
        MPI_Status status;
        MPI_Send (subCells, chunkSize, cellDataType, 0, 500, MPI_COMM_WORLD); 
        MPI_Send (cnumPars_sub, chunkSize, MPI_INT, 0, 600, MPI_COMM_WORLD); 
   }

   if (DEBUG)
	printf ("<%d>, numParticles = %d\n", rank, numParticles);
   
   
    MPI_Barrier (MPI_COMM_WORLD);

   if (rank==0) {
	// saving file...
	SaveFile (outputFile);
	averageCompuationTime = (float)recvbuff [0]/size;
	printf ("Average Computation time of all processors is = %f msec \n", averageCompuationTime );		
	
   }


	MPI_Barrier (MPI_COMM_WORLD);
	
  MPI_Finalize();
}
