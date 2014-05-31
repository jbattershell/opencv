#include "CSharpFFTW.h"
#include <string.h>
#include <math.h>
#include <float.h>

using namespace CSharpFFTW;
using namespace Platform;

#define mag(x,y) sqrtf((x)*(x) + (y)*(y))
#define min(x,y) ((x) > (y) ? (y) : (x))

FFTW::FFTW( unsigned int N )
{
	this->N = N;
	this->inputBuffer = (float*) fftwf_malloc(sizeof(float)*N);
	this->outputBuffer = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*(N/2 + 1));

	this->plan = fftwf_plan_dft_r2c_1d( N, inputBuffer, outputBuffer, FFTW_MEASURE );
}

FFTW::~FFTW() {
	fftwf_destroy_plan( this->plan );
	fftwf_free( this->outputBuffer );
	fftwf_free( this->inputBuffer );
}

unsigned int FFTW::getLength() {
	return this->N;
}

Platform::Array<Complex>^ FFTW::fft( const Platform::Array<float>^ input ) {
	// First, we gotta copy input into inputBuffer.  We only copy as much as we can,
	// that is, min(N, input->Length).  If input->Length < N, then we zero-pad up to N
	memcpy( this->inputBuffer, input->Data, sizeof(float)*min(this->N, input->Length) );

	// Zero-pad
	if( this->N > input->Length )
		memset( this->inputBuffer + input->Length*sizeof(float), 0, sizeof(float)*(this->N - input->Length) );

	// Make fftw do its magic
	fftwf_execute( this->plan );

	// The resultant spectrum is in outputBuffer!
	Platform::Array<Complex>^ outputArray = ref new Platform::Array<Complex>((Complex *)this->outputBuffer, (this->N/2 + 1) );
	return outputArray;
}


Platform::Array<float>^ FFTW::fftMag( const Platform::Array<float>^ input ) {
	// First, we gotta copy input into inputBuffer.  :(
	memcpy( this->inputBuffer, input->Data, sizeof(float)*min(this->N,input->Length) );

	// Zero-pad
	if( this->N > input->Length )
		memset( this->inputBuffer + input->Length*sizeof(float), 0, sizeof(float)*(this->N - input->Length) );

	// Make fftw do its magic
	fftwf_execute( this->plan );

	// Allocate a chunk of memory N/2 + 1 samples long
	Platform::Array<float>^ outputArray = ref new Platform::Array<float>(this->N/2 + 1);

	// Calculate the magnitude of each FFT bin
	float * outputArrayData = outputArray->begin();
	for( unsigned int i=0; i<this->N/2 + 1; ++i ) {
		outputArrayData[i] = mag(outputBuffer[i][0], outputBuffer[i][1]);
	}

	return outputArray;
}

Platform::Array<float>^ FFTW::fftLogMag( const Platform::Array<float>^ input ) {
	// Make fftMag do most of the work for us,
	Platform::Array<float>^ data = fftMag( input );

	// Take the log of every element:
	float * dataPtr = data->begin();
	for( unsigned int i=0; i<data->Length; ++i )
		dataPtr[i] = logf(dataPtr[i]);

	return data;
}