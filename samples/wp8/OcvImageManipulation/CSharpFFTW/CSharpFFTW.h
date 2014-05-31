#pragma once
#include "include/fftw3.h"

namespace CSharpFFTW
{
	// This is a simple value struct that packs memory the same way as an fftwf_complex
	public value struct Complex {
		float real, imag;
	};




    public ref class FFTW sealed
    {
	private:
		// Internally managed buffers
		float * inputBuffer;
		fftwf_complex * outputBuffer;

		// Length of buffers
		unsigned int N;

		// The fftw plan we generate in the constructor
		fftwf_plan plan;

    public:
        FFTW( unsigned int N );
		virtual ~FFTW();

		// Returns the length this FFTW calculates for us
		unsigned int getLength();

		// Calculate the complex-valued FFT, returning it as an array.  Note that this FFT will now zero-pad for you!
		Platform::Array<Complex>^ fft( const Platform::Array<float>^ input );

		// Returns the magnitude-only FFT.  Note that this FFT will now zero-pad for you!
		Platform::Array<float>^ fftMag( const Platform::Array<float>^ input );

		// Outputs the log-magnitude of the fft, same logical length as fft(), but doesn't have two elements per bin
		// Note that this FFT will now zero-pad for you!
		Platform::Array<float>^ fftLogMag( const Platform::Array<float>^ input );
    };
}