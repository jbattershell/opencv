using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Navigation;
using Microsoft.Phone.Controls;
using Microsoft.Phone.Shell;
using PhoneXamlDirect3DApp1Comp;
using Microsoft.Phone.Tasks;
using System.Windows.Media.Imaging;
using System.Threading;
using System.Windows.Resources;
using System.IO;
using  System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.Xna.Framework.Media;
using System.Windows.Threading;
using Microsoft.Phone.Info;

namespace PhoneXamlDirect3DApp1
{
    public partial class MainPage : PhoneApplicationPage
    {
        private Direct3DInterop m_d3dInterop = new Direct3DInterop();
        private DispatcherTimer m_timer;
        MediaLibrary library = new MediaLibrary();

        // Constructor
        public MainPage()
        {
            InitializeComponent();
            m_timer = new DispatcherTimer();
            m_timer.Interval = new TimeSpan(0, 0, 1);
            m_timer.Tick += new EventHandler(timer_Tick);
            m_timer.Start();
        }

        private void DrawingSurface_Loaded(object sender, RoutedEventArgs e)
        {
            // Set window bounds in dips
            m_d3dInterop.WindowBounds = new Windows.Foundation.Size(
                (float)DrawingSurface.ActualWidth,
                (float)DrawingSurface.ActualHeight
                );

            // Set native resolution in pixels
            m_d3dInterop.NativeResolution = new Windows.Foundation.Size(
                (float)Math.Floor(DrawingSurface.ActualWidth * Application.Current.Host.Content.ScaleFactor / 100.0f + 0.5f),
                (float)Math.Floor(DrawingSurface.ActualHeight * Application.Current.Host.Content.ScaleFactor / 100.0f + 0.5f)
                );

            // Set render resolution to the full native resolution
            m_d3dInterop.RenderResolution = m_d3dInterop.NativeResolution;

            // Hook-up native component to DrawingSurface
            DrawingSurface.SetContentProvider(m_d3dInterop.CreateContentProvider());
            DrawingSurface.SetManipulationHandler(m_d3dInterop);

            //Hookup capture frame
            m_d3dInterop.OnCaptureFrameReady += m_d3dInterop_OnCaptureFrameReady;
        }

        void m_d3dInterop_OnCaptureFrameReady(int[] data, int cols, int rows)
        {
            
            Dispatcher.BeginInvoke(() =>
            {
                WriteableBitmap wb = new WriteableBitmap(cols, rows);
                Array.Copy(data, wb.Pixels, data.Length);

                var fileStream = new MemoryStream();
                wb.SaveJpeg(fileStream, wb.PixelWidth, wb.PixelHeight, 100, 100);
                fileStream.Seek(0, SeekOrigin.Begin);

                library.SavePictureToCameraRoll("finalpicture", fileStream);
            });

        }

        private void RadioButton_Checked(object sender, RoutedEventArgs e)
        {
            RadioButton rb = sender as RadioButton;
            switch (rb.Name)
            {
                case "Normal":
                    m_d3dInterop.SetAlgorithm(OCVFilterType.ePreview);
                    break;

                case "Gray":
                    m_d3dInterop.SetAlgorithm(OCVFilterType.eGray);
                    break;

                case "Canny":
                    m_d3dInterop.SetAlgorithm(OCVFilterType.eCanny);
                    break;

                case "Motion":
                    m_d3dInterop.SetAlgorithm(OCVFilterType.eMotion);
                    break;

                case "Features":
                    //m_d3dInterop.SetAlgorithm(OCVFilterType.eFindFeatures);
                    m_d3dInterop.SetCapture();
                    break;
            }
        }

        private void timer_Tick(object sender, EventArgs e)
        {
            try
            {
                // These are TextBlock controls that are created in the page’s XAML file.  
                float value = DeviceStatus.ApplicationCurrentMemoryUsage / (1024.0f * 1024.0f) ;
                MemoryTextBlock.Text = value.ToString();
                value = DeviceStatus.ApplicationPeakMemoryUsage / (1024.0f * 1024.0f);
                PeakMemoryTextBlock.Text = value.ToString();
            }
            catch (Exception ex)
            {
                MemoryTextBlock.Text = ex.Message;
            }
        }
    }       
}