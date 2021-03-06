#pragma once

#include "pch.h"
#include "BasicTimer.h"
#include "QuadRenderer.h"
#include <DrawingSurfaceNative.h>
#include <ppltasks.h>
#include <windows.storage.streams.h>
#include <memory>
#include <mutex>


#include <opencv2\imgproc\types_c.h>

using namespace Windows::Foundation;

namespace PhoneXamlDirect3DApp1Comp
{
  
public enum class OCVFilterType
{
	ePreview,
	eGray,
	eCanny,
	eBlur,
	eFindFeatures,
	eSepia,
	eMotion,
	eNumOCVFilterTypes
};

class CameraCapturePreviewSink;
class CameraCaptureSampleSink;

public delegate void RequestAdditionalFrameHandler();
public delegate void RecreateSynchronizedTextureHandler();
public delegate void CaptureFrameReadyEvent(const Platform::Array<int>^ data, int cols, int rows);
public delegate void FrameReadyEvent(const Platform::Array<float>^ bins);

[Windows::Foundation::Metadata::WebHostHidden]
public ref class Direct3DInterop sealed : public Windows::Phone::Input::Interop::IDrawingSurfaceManipulationHandler
{
public:
	Direct3DInterop();

	Windows::Phone::Graphics::Interop::IDrawingSurfaceContentProvider^ CreateContentProvider();

	// IDrawingSurfaceManipulationHandler
	virtual void SetManipulationHost(Windows::Phone::Input::Interop::DrawingSurfaceManipulationHost^ manipulationHost);

	event RequestAdditionalFrameHandler^ RequestAdditionalFrame;
	event RecreateSynchronizedTextureHandler^ RecreateSynchronizedTexture;

	property Windows::Foundation::Size WindowBounds;
	property Windows::Foundation::Size NativeResolution;
	property Windows::Foundation::Size RenderResolution
	{
		Windows::Foundation::Size get(){ return m_renderResolution; }
		void set(Windows::Foundation::Size renderResolution);
	}
    void SetAlgorithm(OCVFilterType type) { m_algorithm = type; };	//changes which filter is applied
    void UpdateFrame(byte* buffer, int width, int height);
	
	event CaptureFrameReadyEvent^ OnCaptureFrameReady;
	event FrameReadyEvent^ OnFrameReady;
	void SetCapture();
	void ResetCapture();
	bool MotionStatus();
	float LowMotionBins();
	Platform::Array<float>^ MotionBins();
	int GetNumberOfBins();
	void SetPixelThreshold(int thresh);
	void SetImageThreshold(int thresh);
	void SetBackground();
	void SetMotionDetected(bool motion);

	void pauseVideo();
	void resumeVideo();

	void viewFinderTurnOff();
	void viewFinderTurnOn();

protected:
	// Event Handlers
	void OnPointerPressed(Windows::Phone::Input::Interop::DrawingSurfaceManipulationHost^ sender, Windows::UI::Core::PointerEventArgs^ args);
	void OnPointerMoved(Windows::Phone::Input::Interop::DrawingSurfaceManipulationHost^ sender, Windows::UI::Core::PointerEventArgs^ args);
	void OnPointerReleased(Windows::Phone::Input::Interop::DrawingSurfaceManipulationHost^ sender, Windows::UI::Core::PointerEventArgs^ args);

internal:
	HRESULT STDMETHODCALLTYPE Connect(_In_ IDrawingSurfaceRuntimeHostNative* host);
	void STDMETHODCALLTYPE Disconnect();
	HRESULT STDMETHODCALLTYPE PrepareResources(_In_ const LARGE_INTEGER* presentTargetTime, _Out_ BOOL* contentDirty);
	HRESULT STDMETHODCALLTYPE GetTexture(_In_ const DrawingSurfaceSizeF* size, _Out_ IDrawingSurfaceSynchronizedTextureNative** synchronizedTexture, _Out_ DrawingSurfaceRectF* textureSubRectangle);
	ID3D11Texture2D* GetTexture();

private:
    void StartCamera();
    void ProcessFrame();
    bool SwapFrames();
	
	static const int NUMOFBINS = 15;
	float motionBins[NUMOFBINS];

	QuadRenderer^ m_renderer;
	Windows::Foundation::Size m_renderResolution;
    OCVFilterType m_algorithm;
    bool m_contentDirty;
    bool m_getBackground;
	bool m_captureFrame;
	bool motionDetected;
	bool pauseFrames;
	bool viewFinderOn;
	float bottombins;
    std::shared_ptr<cv::Mat> m_backFrame;
    std::shared_ptr<cv::Mat> m_frontFrame;
    std::shared_ptr<cv::Mat> m_frontMinus1Frame;
    std::shared_ptr<cv::Mat> m_frontMinus2Frame;
    std::shared_ptr<cv::Mat> m_diffFrame;
	std::shared_ptr<cv::Mat> m_backgroundFrame;
    std::mutex m_mutex;

	//Processing Thread Stuff
	IAsyncAction^ threadHandle;
	bool frameProcessingInProgress;
	bool frameRenderingInProgress;
	Direct3DInterop^ callingInstance;

	int pixelThreshold;
	int imageThreshold;
	//int bins;

	Windows::Phone::Media::Capture::AudioVideoCaptureDevice ^pAudioVideoCaptureDevice;
	ICameraCaptureDeviceNative* pCameraCaptureDeviceNative;
	IAudioVideoCaptureDeviceNative* pAudioVideoCaptureDeviceNative;
	CameraCapturePreviewSink* pCameraCapturePreviewSink;
	CameraCaptureSampleSink* pCameraCaptureSampleSink;


	void diffImg(cv::Mat* t0, cv::Mat* t1, cv::Mat* t2, cv::Mat* output);
	void diffImg(cv::Mat* t0, cv::Mat* t1, cv::Mat* output);
	void ResetTransparency(cv::Mat* mat);
	void ApplyGrayFilter(cv::Mat* mat);
	void ApplyCannyFilter(cv::Mat* mat);
	void ApplyBlurFilter(cv::Mat* mat);
	void ApplyFindFeaturesFilter(cv::Mat* mat);
	void ApplySepiaFilter(cv::Mat* mat); 
	void GetHist(cv::Mat* image, int bins, float binvals[]);
	void ShiftBackground(cv::Mat* newframe, cv::Mat* backFrame, double scale);


	void startProc();

	// This pauses the thread
	void stopProc();

	// This is the function that gets run in a new thread
	void thread( IAsyncAction^ operation );

	//Pass in the call to trigger a new frame
	void ProcessThisFrame();
};

class CameraCapturePreviewSink :
	public Microsoft::WRL::RuntimeClass<
		Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::RuntimeClassType::ClassicCom>,
		ICameraCapturePreviewSink>
{
public:
    void SetDelegate(Direct3DInterop^ delegate)
    {
        m_Direct3dInterop = delegate;
    }

	IFACEMETHODIMP_(void) OnFrameAvailable(
		DXGI_FORMAT format,
		UINT width,
		UINT height,
		BYTE* pixels);

private:
    Direct3DInterop^ m_Direct3dInterop;
};


class CameraCaptureSampleSink :
	public Microsoft::WRL::RuntimeClass<
		Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::RuntimeClassType::ClassicCom>,
		ICameraCaptureSampleSink>
{
public:
    void SetDelegate(Direct3DInterop^ delegate)
    {
        m_Direct3dInterop = delegate;
    }

    IFACEMETHODIMP_(void) OnSampleAvailable(
		    ULONGLONG hnsPresentationTime,
		    ULONGLONG hnsSampleDuration,
		    DWORD cbSample,
		    BYTE* pSample);

private:
    Direct3DInterop^ m_Direct3dInterop;
};

}
