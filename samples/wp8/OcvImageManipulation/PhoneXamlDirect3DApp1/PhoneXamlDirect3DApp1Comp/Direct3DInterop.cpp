#include "pch.h"
#include "Direct3DInterop.h"
#include "Direct3DContentProvider.h"
#include <windows.storage.streams.h>
#include <wrl.h>
#include <robuffer.h>
#include <opencv2\core\core.hpp>
#include <opencv2\core\core_c.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <algorithm>
#include "Frame.h"

using namespace Windows::Storage::Streams;
using namespace Microsoft::WRL;
using namespace Windows::Foundation;
using namespace Windows::UI::Core;
using namespace Microsoft::WRL;
using namespace Windows::Phone::Graphics::Interop;
using namespace Windows::Phone::Input::Interop;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::Phone::Media::Capture;

#if !defined(_M_ARM)
#pragma message("warning: Direct3DInterop.cpp: Windows Phone camera code does not run in the emulator.")
#pragma message("warning: Direct3DInterop.cpp: Please compile as an ARM build and run on a device.")
#endif

namespace PhoneXamlDirect3DApp1Comp
{
    // Called each time a preview frame is available
    void CameraCapturePreviewSink::OnFrameAvailable(
	    DXGI_FORMAT format,
	    UINT width,
	    UINT height,
	    BYTE* pixels
	    )
    {
		//passes data from camera to the function that processes and draws
        m_Direct3dInterop->UpdateFrame(pixels, width, height);		
    }

    // Called each time a captured frame is available	
    void CameraCaptureSampleSink::OnSampleAvailable(
	    ULONGLONG hnsPresentationTime,
	    ULONGLONG hnsSampleDuration,
	    DWORD cbSample,
	    BYTE* pSample)
    {


    }

    Direct3DInterop::Direct3DInterop() 
        : m_algorithm(OCVFilterType::ePreview)
        , m_contentDirty(false)
        , m_backFrame(nullptr)
        , m_frontFrame(nullptr)
        , m_frontMinus1Frame(nullptr)
        , m_frontMinus2Frame(nullptr)
        , m_diffFrame(nullptr)
    {
    }

    bool Direct3DInterop::SwapFrames()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if(m_backFrame != nullptr)
        {
			
            std::swap(m_frontMinus2Frame, m_frontMinus1Frame);	//oldest data goes to m_frontMinus2Frame
            std::swap(m_frontMinus1Frame, m_frontFrame);		//next oldest goes to m_frontMinus1Frame
            std::swap(m_backFrame, m_frontFrame);				//newest data goes to frontFrame
            return true;
        }
        return false;
    }

    void Direct3DInterop::UpdateFrame(byte* buffer,int width,int height)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if(m_backFrame == nullptr)
        {
			//Only goes in here the first time a frame is passed in.
            m_backFrame = std::shared_ptr<cv::Mat> (new cv::Mat(height, width, CV_8UC4));
            m_frontFrame = std::shared_ptr<cv::Mat> (new cv::Mat(height, width, CV_8UC4));
            m_frontMinus1Frame = std::shared_ptr<cv::Mat> (new cv::Mat(height, width, CV_8UC4));
            m_frontMinus2Frame = std::shared_ptr<cv::Mat> (new cv::Mat(height, width, CV_8UC4));
            m_diffFrame = std::shared_ptr<cv::Mat> (new cv::Mat(height, width, CV_8UC4));
        }

        memcpy(m_backFrame.get()->data, buffer, 4 * height*width);	//copies new frame to m_backFrame
        m_contentDirty = true;	//Marks that m_backFrame holds a new image
        RequestAdditionalFrame();	//Maybe a feedback mechanism to tell camera to send another frame? Just a guess
    }

    void Direct3DInterop::ProcessFrame()
    {
		if (SwapFrames())	//always returns true unless backframe is a nullptr. Puts most current into frontframe
		{
			if (m_renderer)
			{
				cv::Mat* mat = m_frontFrame.get();	//load in newest data
				cv::Mat* matOld = m_frontMinus1Frame.get();	//load in 1 frame old data
				cv::Mat* matOlder = m_frontMinus2Frame.get();	//load in 2 frame old data
				cv::Mat* matdiff = m_diffFrame.get();	//load in newest data
				//cv::Mat* matDiff = m_diffFrame.get();
		
				switch (m_algorithm)
				{
					case OCVFilterType::ePreview:
					{
						break;
					}

					case OCVFilterType::eGray:
					{
						ApplyGrayFilter(mat);
						break;
					}

					case OCVFilterType::eCanny:
					{
						ApplyCannyFilter(mat);
						break;
					}

					case OCVFilterType::eBlur:
					{
						ApplyBlurFilter(mat);
						break;
					}

					case OCVFilterType::eFindFeatures:
					{
						ApplyFindFeaturesFilter(mat);
						break;
					}

					case OCVFilterType::eSepia:
					{
						ApplySepiaFilter(mat);
						break;
					}
					case OCVFilterType::eMotion:
					{
						ApplyGrayFilter(mat);
						diffImg(matOlder, matOld, mat, matdiff);

						const int bins = 10;
						float binvals[bins];
						GetHist(matdiff,bins,binvals);
						int i=0;

					//	int dims = 1;
					//	const int histSize = 10;
					//	cv::Mat hist = cv::Mat(dims,histSize,CV_8UC4);
					//	int nimages = 1;
					//	const int channels = 0;
					//	const float varranges[] = {0, 256};
					//	const float* ranges = varranges;
					//	

					//		cv::calcHist(matdiff, nimages, &channels,
					//			cv::Mat(), hist, dims, &histSize,
     //              &ranges, true, false );



					//float binvals[histSize];
					//for (int i=0; i<histSize;i++)
					//{
					//	binvals[i]=hist.at<float>(i);
					//}

				   //cv::calcHist( const Mat* images, int nimages, const int* channels,
       //            InputArray _mask, OutputArray _hist, int dims, const int* histSize,
       //            const float** ranges, bool uniform, bool accumulate )

						break;	
					}
				}
				

				if(m_algorithm == OCVFilterType::eMotion)
					m_renderer->CreateTextureFromByte(matdiff->data, matdiff->cols, matdiff->rows);
				else
					m_renderer->CreateTextureFromByte(mat->data, mat->cols, mat->rows);
			}
		}
    }

	void Direct3DInterop::GetHist(cv::Mat* image, int bins, float binvals[])
	{
		int dims = 1;
		const int histSize = 10;
		cv::Mat hist = cv::Mat(dims,histSize,CV_8UC4);
		int nimages = 1;
		const int channels = 0;
		const float varranges[] = {0, 256};
		const float* ranges = varranges;
						

			cv::calcHist(image, nimages, &channels,
				cv::Mat(), hist, dims, &histSize,
		&ranges, true, false );

		//put results into array
		for (int i=0; i<histSize;i++)
		{
			binvals[i]=hist.at<float>(i);
		}
	}

	void Direct3DInterop::diffImg(cv::Mat* t0, cv::Mat* t1, cv::Mat* t2, cv::Mat* output)
	{
		
		cv::absdiff(*t0,*t1,*output);	//take difference of past two frames
		cv::absdiff(*t2,*t1,*t0);	//overwrites t0 (t0 will be overwritten in the next cycle anyway)

		cv::bitwise_and(*output,*t0,*output);
		ResetTransparency(output);	//only needed when I want to visualize the result
			
	}

	void Direct3DInterop::ResetTransparency(cv::Mat* mat){
		//for pointer method
		unsigned char *dataptr;
		unsigned int imgBytes = 4 * mat->rows * mat->cols;

		//for frame method
		Frame f = Frame(mat->cols, mat->rows, (uintptr_t)mat->data);
		unsigned int pixels = f.height() * f.width();
		
		dataptr = mat->data;

		for(unsigned int y=0; y<f.height();y++)
		{
			for(unsigned int x=0; x<f.width();x++)
			f(x,y).alpha = 255;
		}

		////transparency is last (4th) byte in each pixel
		//for(unsigned int i=3;i<imgBytes;i+=4)
		//{
		//	dataptr[i]=255;
		//}

		return;
	}


	void Direct3DInterop::ApplyGrayFilter(cv::Mat* mat)
	{
		cv::Mat intermediateMat;
		cv::cvtColor(*mat, intermediateMat, CV_RGBA2GRAY);
		cv::cvtColor(intermediateMat, *mat, CV_GRAY2BGRA);
	}

	void Direct3DInterop::ApplyCannyFilter(cv::Mat* mat)
	{
		cv::Mat intermediateMat;
		cv::Canny(*mat, intermediateMat, 80, 90);
		cv::cvtColor(intermediateMat, *mat, CV_GRAY2BGRA);
	}

	void Direct3DInterop::ApplyBlurFilter(cv::Mat* mat)
	{
		cv::Mat intermediateMat;
		//	cv::Blur(image, intermediateMat, 80, 90);
		cv::cvtColor(intermediateMat, *mat, CV_GRAY2BGRA);
	}

	void Direct3DInterop::ApplyFindFeaturesFilter(cv::Mat* mat)
	{
        cv::Mat intermediateMat;
        cv::FastFeatureDetector detector(50);
        std::vector<cv::KeyPoint> features;

        cv::cvtColor(*mat, intermediateMat, CV_RGBA2GRAY);
        detector.detect(intermediateMat, features);

        for( unsigned int i = 0; i < std::min(features.size(), (size_t)50); i++ )
        {
            const cv::KeyPoint& kp = features[i];
            cv::circle(*mat, cv::Point((int)kp.pt.x, (int)kp.pt.y), 10, cv::Scalar(255,0,0,255));
        }
	}

	void Direct3DInterop::ApplySepiaFilter(cv::Mat* mat)
	{
		const float SepiaKernelData[16] =
		{
			/* B */0.131f, 0.534f, 0.272f, 0.f,
			/* G */0.168f, 0.686f, 0.349f, 0.f,
			/* R */0.189f, 0.769f, 0.393f, 0.f,
			/* A */0.000f, 0.000f, 0.000f, 1.f
		};

		const cv::Mat SepiaKernel(4, 4, CV_32FC1, (void*)SepiaKernelData);
		//cv::transform(*mat, *mat, SepiaKernel);
			
		////////////////////////////////////
		//My little blurring function
		cv::Mat intermediateMat;
		cv::Size blurSize;
		blurSize.height=20;
		blurSize.width=20;

		cv::blur(*mat, intermediateMat, blurSize, cv::Point(-1,-1), 4);	//blur, store result in intermediate
		std::swap(intermediateMat,*mat);	//move blurred image to displayed image
		//intermediateMat.copyTo(*mat);
		///////////////////////////////////////
		
	}

    IDrawingSurfaceContentProvider^ Direct3DInterop::CreateContentProvider()
    {
        ComPtr<Direct3DContentProvider> provider = Make<Direct3DContentProvider>(this);
        return reinterpret_cast<IDrawingSurfaceContentProvider^>(provider.Detach());
    }

    // IDrawingSurfaceManipulationHandler
    void Direct3DInterop::SetManipulationHost(DrawingSurfaceManipulationHost^ manipulationHost)
    {
        manipulationHost->PointerPressed +=
            ref new TypedEventHandler<DrawingSurfaceManipulationHost^, PointerEventArgs^>(this, &Direct3DInterop::OnPointerPressed);

        manipulationHost->PointerMoved +=
            ref new TypedEventHandler<DrawingSurfaceManipulationHost^, PointerEventArgs^>(this, &Direct3DInterop::OnPointerMoved);

        manipulationHost->PointerReleased +=
            ref new TypedEventHandler<DrawingSurfaceManipulationHost^, PointerEventArgs^>(this, &Direct3DInterop::OnPointerReleased);
    }

    void Direct3DInterop::RenderResolution::set(Windows::Foundation::Size renderResolution)
    {
        if (renderResolution.Width  != m_renderResolution.Width ||
            renderResolution.Height != m_renderResolution.Height)
        {
            m_renderResolution = renderResolution;

            if (m_renderer)
            {
                m_renderer->UpdateForRenderResolutionChange(m_renderResolution.Width, m_renderResolution.Height);
                RecreateSynchronizedTexture();
            }
        }
    }

    // Event Handlers

    void Direct3DInterop::OnPointerPressed(DrawingSurfaceManipulationHost^ sender, PointerEventArgs^ args)
    {
        // Insert your code here.
    }

    void Direct3DInterop::OnPointerMoved(DrawingSurfaceManipulationHost^ sender, PointerEventArgs^ args)
    {
        // Insert your code here.
    }

    void Direct3DInterop::OnPointerReleased(DrawingSurfaceManipulationHost^ sender, PointerEventArgs^ args)
    {
        // Insert your code here.
    }

    void Direct3DInterop::StartCamera()
    {
        // Set the capture dimensions
	    Size captureDimensions;
	    captureDimensions.Width = 640;
	    captureDimensions.Height = 480;		

	    // Open the AudioVideoCaptureDevice for video only
	    IAsyncOperation<AudioVideoCaptureDevice^> ^openOperation = AudioVideoCaptureDevice::OpenForVideoOnlyAsync(CameraSensorLocation::Back, captureDimensions);

	    openOperation->Completed = ref new AsyncOperationCompletedHandler<AudioVideoCaptureDevice^>(
		    [this] (IAsyncOperation<AudioVideoCaptureDevice^> ^operation, Windows::Foundation::AsyncStatus status)
		    {
			    if (status == Windows::Foundation::AsyncStatus::Completed)
			    {
				    auto captureDevice = operation->GetResults();

				    // Save the reference to the opened video capture device
				    pAudioVideoCaptureDevice = captureDevice;
	
				    // Retrieve the native ICameraCaptureDeviceNative interface from the managed video capture device
				    ICameraCaptureDeviceNative *iCameraCaptureDeviceNative = NULL; 
				    HRESULT hr = reinterpret_cast<IUnknown*>(captureDevice)->QueryInterface(__uuidof(ICameraCaptureDeviceNative), (void**) &iCameraCaptureDeviceNative);

				    // Save the pointer to the native interface
				    pCameraCaptureDeviceNative = iCameraCaptureDeviceNative;

				    // Initialize the preview dimensions (see the accompanying article at )
				    // The aspect ratio of the capture and preview resolution must be equal,
				    // 4:3 for capture => 4:3 for preview, and 16:9 for capture => 16:9 for preview.
				    Size previewDimensions;
				    previewDimensions.Width = 640;
				    previewDimensions.Height = 480;					

				    IAsyncAction^ setPreviewResolutionAction = pAudioVideoCaptureDevice->SetPreviewResolutionAsync(previewDimensions);
				    setPreviewResolutionAction->Completed = ref new AsyncActionCompletedHandler(
					    [this](IAsyncAction^ action, Windows::Foundation::AsyncStatus status)
					    {
						    HResult hr = action->ErrorCode;

						    if (status == Windows::Foundation::AsyncStatus::Completed)
						    {
							    // Create the sink
							    MakeAndInitialize<CameraCapturePreviewSink>(&pCameraCapturePreviewSink);
                                pCameraCapturePreviewSink->SetDelegate(this);
							    pCameraCaptureDeviceNative->SetPreviewSink(pCameraCapturePreviewSink);

							    // Set the preview format
							    pCameraCaptureDeviceNative->SetPreviewFormat(DXGI_FORMAT::DXGI_FORMAT_B8G8R8A8_UNORM);						
						    }
					    }
				    );

				    // Retrieve IAudioVideoCaptureDeviceNative native interface from managed projection.
				    IAudioVideoCaptureDeviceNative *iAudioVideoCaptureDeviceNative = NULL;
				    hr = reinterpret_cast<IUnknown*>(captureDevice)->QueryInterface(__uuidof(IAudioVideoCaptureDeviceNative), (void**) &iAudioVideoCaptureDeviceNative);

				    // Save the pointer to the IAudioVideoCaptureDeviceNative native interface
				    pAudioVideoCaptureDeviceNative = iAudioVideoCaptureDeviceNative;

				    // Set sample encoding format to ARGB. See the documentation for further values.
				    pAudioVideoCaptureDevice->VideoEncodingFormat = CameraCaptureVideoFormat::Argb;

				    // Initialize and set the CameraCaptureSampleSink class as sink for captures samples
				    MakeAndInitialize<CameraCaptureSampleSink>(&pCameraCaptureSampleSink);
				    pAudioVideoCaptureDeviceNative->SetVideoSampleSink(pCameraCaptureSampleSink);

				    // Start recording (only way to receive samples using the ICameraCaptureSampleSink interface
				    pAudioVideoCaptureDevice->StartRecordingToSinkAsync();
			    }
		    }
	    );

    }
    // Interface With Direct3DContentProvider
    HRESULT Direct3DInterop::Connect(_In_ IDrawingSurfaceRuntimeHostNative* host)
    {
        m_renderer = ref new QuadRenderer();
        m_renderer->Initialize();
        m_renderer->UpdateForWindowSizeChange(WindowBounds.Width, WindowBounds.Height);
        m_renderer->UpdateForRenderResolutionChange(m_renderResolution.Width, m_renderResolution.Height);
        StartCamera();

        return S_OK;
    }

    void Direct3DInterop::Disconnect()
    {
        m_renderer = nullptr;
    }

    HRESULT Direct3DInterop::PrepareResources(_In_ const LARGE_INTEGER* presentTargetTime, _Out_ BOOL* contentDirty)
    {
        *contentDirty = m_contentDirty;
        if(m_contentDirty)		
        {
			//m_backFrame holds a new image, so go process it
            ProcessFrame();
        }
        m_contentDirty = false;
        return S_OK;
    }

    HRESULT Direct3DInterop::GetTexture(_In_ const DrawingSurfaceSizeF* size, _Out_ IDrawingSurfaceSynchronizedTextureNative** synchronizedTexture, _Out_ DrawingSurfaceRectF* textureSubRectangle)
    {
 		m_renderer->Update();
        m_renderer->Render();
        return S_OK;
    }

    ID3D11Texture2D* Direct3DInterop::GetTexture()
    {
        return m_renderer->GetTexture();
    }
}
