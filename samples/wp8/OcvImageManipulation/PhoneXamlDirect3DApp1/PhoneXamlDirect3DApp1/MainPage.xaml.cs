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
using System.Runtime.InteropServices.WindowsRuntime;
using libsvm;
using Microsoft.Xna.Framework.Media;
using System.Windows.Threading;
using Microsoft.Phone.Info;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Windows.Media;

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Audio;
using System.IO.IsolatedStorage;

using Microsoft.Live;
using Microsoft.Live.Controls;

using libflac_wrapper;
using libsound;
using LineGraph;
using CSharpFFTW;

using System.Threading.Tasks;
using System.Windows.Documents;
using Windows.Storage.Streams;

using Newtonsoft.Json;


namespace PhoneXamlDirect3DApp1
{
    public partial class MainPage : PhoneApplicationPage
    {
        #region Video Variables
        private Direct3DInterop m_d3dInterop = new Direct3DInterop();
        private DispatcherTimer m_timer;
        MediaLibrary library = new MediaLibrary();
        private bool motionCaptureEnabled;
        private string uploadPrefix = "Jon_";
        
        private int trainingMode;   //0=not training, 1=positive training, 2=negative training
            const int NOTTRAINING = 0;
            const int POSITIVETRAINING = 1;
            const int NEGATIVETRAINING = 2;
            const int SAMPLESTOTRAIN = 10;

        private int pendingTrainingMode;
        private int pendingCount;
            const int PENDINGCOUNTTRIGGER = 20;
        
        private bool pendingCaptureMode;


            private int TIMERPERIOD = 250;

        // This is the SVM model we will eventually train up
        svm_model model = null;
        // This is the buffer of audio data we will perform recognition on
        svm_node[] recogBuff = null;

        // These are the list of recording objects
        //ObservableCollection<float> posRecordings = new ObservableCollection<float>();
        //ObservableCollection<float> negRecordings = new ObservableCollection<float>();

        ObservableCollection<float[]> posRecordings = new ObservableCollection<float[]>();
        ObservableCollection<float[]> negRecordings = new ObservableCollection<float[]>();

        #endregion


        #region Audio Classes
        ///////////this is code for google stuff//////////////////
        // These classes made from http://json2csharp.com/
        public class Alternative
        {
            public string transcript { get; set; }
            public double confidence { get; set; }
        }

        public class Result
        {
            public List<Alternative> alternative { get; set; }
            public bool final { get; set; }
        }

        public class RecognitionResult
        {
            public List<Result> result { get; set; }
            public int result_index { get; set; }
        }
        /// ///////this is code for google stuff///////////////////
        #endregion


        #region Audio Variables
        //this is for C++ FFTW
        //SoundIO sio = null;
        AudioTool at = null;
        
        FFTW fftw = null;
        //this is for C++ FFTW

        private Microphone microphone = Microphone.Default;     // Object representing the physical microphone on the device
        private byte[] buffer;                                  // Dynamic buffer to retrieve audio data from the microphone
        private MemoryStream stream = new MemoryStream();       // Stores the audio data for later playback
        

        // we give our file a filename
        private string strSaveName;

        //Scale the volume
        byte volumeScale = 1;

        // SkyDrive session
        private LiveConnectClient client;

        //////This is code for google stuff//////////////////////////
        // This is the object we'll record data into
        private libFLAC lf = new libFLAC();
        private SoundIO sio = new SoundIO();

        // This is our list of float[] chunks that we're keeping track of
        private List<float[]> recordedAudio = new List<float[]>();

        // This is our flag as to whether or not we're currently recording
        private bool recording = false;
        private bool processing = false; 
        private bool loggedIn = false;
        private bool uploadComplete = true;

        string googleText;

        int quietCount = 0;

        
        //////This is code for google stuff//////////////////////////
        #endregion

        // Constructor
        public MainPage()
        {
            InitializeComponent();

            #region Video Init

            m_timer = new DispatcherTimer();
            m_timer.Interval = new TimeSpan(0, 0, 0, 0, TIMERPERIOD);   //trigger timer up to 4x per second
            m_timer.Tick += new EventHandler(timer_Tick);
            m_timer.Start();

            motionCaptureEnabled = false;
            trainingMode = NOTTRAINING;

            pendingTrainingMode = NOTTRAINING;
            pendingCount=0;
            
            pendingCaptureMode = false;
            
            #endregion

            #region Audio Init
            // Timer to simulate the XNA Framework game loop (Microphone is 
            // from the XNA Framework). We also use this timer to monitor the 
            // state of audio playback so we can update the UI appropriately.
            DispatcherTimer dt = new DispatcherTimer();
            dt.Interval = TimeSpan.FromMilliseconds(33);
            dt.Tick += new EventHandler(dt_Tick);
            dt.Start();

            // Event handler for getting audio data when the buffer is full
            microphone.BufferReady += new EventHandler<EventArgs>(microphone_BufferReady);

            //this is for C++ FFTW
            // Assuming that we don't know in general the number of output channels/samplerate,
            // we dynamically get those from SoundIO, and then must construct AudioTool in the constructor
            at = new AudioTool(sio.getOutputNumChannels(), sio.getOutputSampleRate());

            // Setup SoundIO right away
            sio.audioInEvent += sio_audioInEvent;
            //sio.start();

            //load audio files into isolated storage
            CopyToIsolatedStorage();

            #endregion

        }

        #region Video Drawing Hookup
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
            m_d3dInterop.OnFrameReady += m_d3dInterop_OnFrameReady;
        }
        #endregion

        #region Video Processing Frame Ready Events
        void m_d3dInterop_OnFrameReady(float[] bins)
        {
            //Learned thresh eval
            if (model != null)
            {
                // Copy in the latest data to recogBuff
                double[] featureData = computeFeatureVector(bins);
                if (recogBuff == null)
                {
                    recogBuff = new svm_node[featureData.Length];
                    for (int i = 0; i < recogBuff.Length; ++i)
                    {
                        recogBuff[i] = new svm_node();
                        recogBuff[i].index = i;
                    }
                }

                // Convert the feature vector over to the svm_node structure
                for (int i = 0; i < featureData.Length; ++i)
                {
                    recogBuff[i].value = featureData[i];
                }

                string resultStr = "";
                double val = evaluate(recogBuff, model, ref resultStr);

                if (val == 1.0)
                {
                    m_d3dInterop.SetMotionDetected(true);
                    Debug.WriteLine(bins[0] + " " + bins[1] + " " + bins[2] + " " + bins[3] + " " + bins[4]);
                }
                else
                    m_d3dInterop.SetMotionDetected(false);

            }
        }

        void m_d3dInterop_OnCaptureFrameReady(int[] data, int cols, int rows)
        {
            m_d3dInterop.ResetCapture();

            float[] bins = m_d3dInterop.MotionBins();
            int midBinNum = m_d3dInterop.GetNumberOfBins() * 4 / 15;

            if (bins[midBinNum] > 300)   //only send in if there is significant motion
            {
                Dispatcher.BeginInvoke(() =>
                {
                    WriteableBitmap wb = new WriteableBitmap(cols, rows);
                    Array.Copy(data, wb.Pixels, data.Length);

                    var fileStream = new MemoryStream();
                    wb.SaveJpeg(fileStream, wb.PixelWidth, wb.PixelHeight, 100, 100);
                    fileStream.Seek(0, SeekOrigin.Begin);

                    var rotatedStream = new MemoryStream();
                    rotatedStream = RotateStream(fileStream, 90);
                    rotatedStream.Seek(0, SeekOrigin.Begin);

                    //float[] bins = m_d3dInterop.MotionBins();
                    string motionoutput = bins[0].ToString() + "," + bins[1].ToString() + "," + bins[2].ToString() + "," + bins[3].ToString() + "," + bins[4].ToString();


                    string name = uploadPrefix + "motion " + DateTime.Now.ToString("yy_MM_dd_hh_mm_ss_fff") + "_" + motionoutput;

                    /*maybe can detect using SensorRotationInDegrees property in AudioVideoCaptureDevice class: Gets the number of degrees that the camera sensor is rotated relative to the screen
                      http://msdn.microsoft.com/en-us/library/windows/desktop/windows.phone.media.capture.audiovideocapturedevice
                     */

                    //library.SavePictureToCameraRoll(name, rotatedStream); //This saves a vertical orientation picture to photo reel
                    uploadFile(name, rotatedStream);   //This saves a vertical orientation picture to oneDrive

                    //library.SavePictureToCameraRoll(name, fileStream); //This saves a horizontal orientation picture to photo reel
                    //uploadFile(name, fileStream);   //This saves a horizontal orientation picture to oneDrive
                });
            }
        }
        #endregion

        #region Video UI Handlers
        private void Training_Checked(object sender, RoutedEventArgs e)
        { 
            RadioButton rb = sender as RadioButton;

            switch (rb.Name)
            {
                case "TrainingOff":
                    trainingMode = NOTTRAINING;

                    break;

                case "TrainingPos":
                    posRecordings.Clear();
                    pendingTrainingMode = POSITIVETRAINING;
                    //trainingMode = POSITIVETRAINING;

                    break;

                case "TrainingNeg":
                    negRecordings.Clear();
                    pendingTrainingMode = NEGATIVETRAINING;
                    //trainingMode = NEGATIVETRAINING;

                    break;
            }
        }

        private void RadioButton_Checked(object sender, RoutedEventArgs e)
        {
            RadioButton rb = sender as RadioButton;
            switch (rb.Name)
            {
               
                case "Motion":
                    //motionCaptureEnabled = true;
                    pendingCaptureMode = true;
                    break;

                case "MotionOff":
                    motionCaptureEnabled = false;
                    break;
            }
        }
        
        private void Background_Button_Click(object sender, RoutedEventArgs e)
        {
            m_d3dInterop.SetBackground();
        }
        #endregion

        private void timer_Tick(object sender, EventArgs e)
        {

            if (pendingCaptureMode) //Button pressed, waiting to settle
            {
                if (pendingCount < PENDINGCOUNTTRIGGER)
                {
                    pendingCount++;
                    Dispatcher.BeginInvoke(() =>
                    {
                        learnOutput.Text = ((int)((PENDINGCOUNTTRIGGER - pendingCount) * TIMERPERIOD / 1000)).ToString();
                    });
                }
                else
                {
                    pendingCount = 0;
                    motionCaptureEnabled = true;
                    pendingCaptureMode = false;

                    Dispatcher.BeginInvoke(() =>
                    {
                        learnOutput.Text = "Capture Started";
                    });
                }
            }

            if (pendingTrainingMode != NOTTRAINING) //Button pressed, waiting to settle
            {
                if (pendingCount < PENDINGCOUNTTRIGGER)
                { 
                    pendingCount++;
                    Dispatcher.BeginInvoke(() =>
                    {
                        learnOutput.Text = ((int)((PENDINGCOUNTTRIGGER - pendingCount) * TIMERPERIOD/1000)).ToString();
                    });
                }
                else
                { 
                    pendingCount = 0;
                    trainingMode = pendingTrainingMode;
                    pendingTrainingMode = NOTTRAINING;


                    Dispatcher.BeginInvoke(() =>
                    {
                        learnOutput.Text = "Training Started";
                    });
                }
            }

            //////////////////////////////////////////////////////////////////////////////

            //Capture training data
            //Capture a certain SAMPLESTOTRAIN samples
            if (negRecordings.Count >= SAMPLESTOTRAIN && trainingMode == NEGATIVETRAINING)
            {
                //training done
                trainingMode = NOTTRAINING;
                Dispatcher.BeginInvoke(() =>
                {
                    learnOutput.Text = "Negative Samples Recorded";
                });
            }
            else if (posRecordings.Count >= SAMPLESTOTRAIN && trainingMode == POSITIVETRAINING)
            {
                //training done
                trainingMode = NOTTRAINING;
                Dispatcher.BeginInvoke(() =>
                {
                    learnOutput.Text = "Positive Samples Recorded";
                });
            }
            else if (m_d3dInterop != null && posRecordings != null && negRecordings != null && trainingMode == POSITIVETRAINING && posRecordings.Count < SAMPLESTOTRAIN)
            {
                posRecordings.Add(m_d3dInterop.MotionBins());
            }
            else if (m_d3dInterop != null && posRecordings != null && negRecordings != null && trainingMode == NEGATIVETRAINING && negRecordings.Count < SAMPLESTOTRAIN)
            {
                //Only records a certain number of frames
                negRecordings.Add(m_d3dInterop.MotionBins());
            }

            ///////////////////////////

            try
            {
                // These are TextBlock controls that are created in the page’s XAML file.  
                float value = DeviceStatus.ApplicationCurrentMemoryUsage / (1024.0f * 1024.0f);
                MemoryTextBlock.Text = value.ToString();
                value = DeviceStatus.ApplicationPeakMemoryUsage / (1024.0f * 1024.0f);
                PeakMemoryTextBlock.Text = value.ToString();

                float[] bins = m_d3dInterop.MotionBins();
                string motionoutput = bins[0].ToString() + ", " + bins[1].ToString() + ", " + bins[2].ToString() + "\n" + bins[3].ToString() + ", " + bins[4].ToString();
                MotionOutput.Text = motionoutput;
                //MotionOutput.Text = m_d3dInterop.LowMotionBins().ToString();
                LearnedOutput.Text = m_d3dInterop.MotionStatus().ToString();

            }
            catch (Exception ex)
            {
                MemoryTextBlock.Text = ex.Message;
            }

            //if (ImageThresh != null)    //update text
            //{
            //    ImageThresVal.Text = ((int)ImageThresh.Value).ToString();

            //}

            if (motionCaptureEnabled)
            {
                m_d3dInterop.SetCapture();
            }
        }

        private MemoryStream RotateStream(MemoryStream stream, int angle)
          {
              stream.Position = 0;
              if (angle % 90 != 0 || angle < 0) throw new ArgumentException();
              if (angle % 360 == 0) return stream;
   
              BitmapImage bitmap = new BitmapImage();
              bitmap.SetSource(stream);
              WriteableBitmap wbSource = new WriteableBitmap(bitmap);
   
              WriteableBitmap wbTarget = null;
              if (angle % 180 == 0)
              {
                  wbTarget = new WriteableBitmap(wbSource.PixelWidth, wbSource.PixelHeight);
              }
              else
              {
                  wbTarget = new WriteableBitmap(wbSource.PixelHeight, wbSource.PixelWidth);
              }
   
              for (int x = 0; x < wbSource.PixelWidth; x++)
              {
                  for (int y = 0; y < wbSource.PixelHeight; y++)
                  {
                      switch (angle % 360)
                      {
                          case 90:
                              wbTarget.Pixels[(wbSource.PixelHeight - y - 1) + x * wbTarget.PixelWidth] = wbSource.Pixels[x + y * wbSource.PixelWidth];
                              break;
                          case 180:
                              wbTarget.Pixels[(wbSource.PixelWidth - x - 1) + (wbSource.PixelHeight - y - 1) * wbSource.PixelWidth] = wbSource.Pixels[x + y * wbSource.PixelWidth];
                              break;
                          case 270:
                              wbTarget.Pixels[y + (wbSource.PixelWidth - x - 1) * wbTarget.PixelWidth] = wbSource.Pixels[x + y * wbSource.PixelWidth];
                              break;
                      }
                  }
              }
              MemoryStream targetStream = new MemoryStream();
              wbTarget.SaveJpeg(targetStream, wbTarget.PixelWidth, wbTarget.PixelHeight, 0, 100);
              return targetStream;
          }


        #region Machine Learning Stuffs
        // This is what starts all the fun!
        private void learnButton_Click(object sender, RoutedEventArgs e)
        {
            learnOutput.Text = "Learning";
            
            // We're going to assemble all positive and negative feature vectors
            List<double[]> featureVecs = new List<double[]>();

            // First, throw in the positive feature vectors
            featureVecs.AddRange(posRecordings.Select(x => computeFeatureVector(x)));

            // Next, throw in the negative feature vectors:
            featureVecs.AddRange(negRecordings.Select(x => computeFeatureVector(x)));

            // Now, create labels, and assign them the proper values:
            double[] labels = new double[featureVecs.Count];
            for (int i = 0; i < posRecordings.Count; ++i)
            {
                labels[i] = 1;
            }
            for (int i = posRecordings.Count; i < featureVecs.Count; ++i)
            {
                labels[i] = 0;
            }

            m_d3dInterop.pauseVideo();
            // Train up our model!
            model = svmTrain(labels, featureVecs.ToArray());
            m_d3dInterop.resumeVideo();
        }

        // This computes a feature vector out of some floating point data
        double[] computeFeatureVector(float data)
        { 
            //floating point data is number of bins over threshold
            //Also add in bin numbers
            double[] featureVec = { data, m_d3dInterop.GetNumberOfBins() };

            return featureVec;
        }

        // This computes a feature vector out of some floating point data
        double[] computeFeatureVector(float[] data)
        {
            //floating point data is number of bins over threshold
            //Also add in bin numbers
            double[] featureVec = { data[0], data[1], data[2], data[3], data[4] };

            return featureVec;
        }

        // This takes in a vector of "labels", (e.g. 0 for a negative example, 1 for a positive example), and a 2-d array
        // of feature vectors, where each vector in featureVecs is labeled by the corresponding value in labels
        private svm_model svmTrain(double[] labels, double[][] featureVecs)
        {
            DateTime start = DateTime.Now;

            // Create our "problem" formulation
            svm_problem prob = new svm_problem();
            int dataCount = featureVecs.Length;

            // y represents the label assigned to each feature vector
            prob.y = labels;
            // l is number of feature vectors
            prob.l = dataCount;
            // x represents the feature vectors
            prob.x = new svm_node[dataCount][];

            // Convert each feature vector into an array of svm_node structures, and assign them to this problem
            for (int i = 0; i < dataCount; i++)
            {
                // Convert a single feature vector into an array of svm_node structures
                prob.x[i] = featureVecs[i].Select((element, idx) =>
                {
                    var node = new svm_node();
                    node.index = idx;
                    node.value = element;
                    return node;
                }).ToArray();
            }

            // Setup initial guess parameters
            svm_parameter param = new svm_parameter();
            param.probability = 1;
            param.gamma = 0.5;
            param.nu = 0.5;
            param.C = 1;
            param.svm_type = svm_parameter.C_SVC;
            param.kernel_type = svm_parameter.LINEAR;
            param.cache_size = 20000;
            param.eps = 0.001;

            // Use cross-validation to find best parameters.  Targets is the calculated
            // label for the given training vectors, from which we glean accuracy
            int bestPow = -5;
            double bestAccuracy = 0.0;
            for (int pow = -2; pow < 10; ++pow)
            {
                param.C = Math.Pow(2.0, pow);
                double[] targets = new double[labels.Length];
                svm.svm_cross_validation(prob, param, 4, targets);

                double currAccuracy = 0;
                for (int i = 0; i < labels.Length; ++i)
                {
                    if (labels[i] == targets[i])
                    {
                        currAccuracy += 1;
                    }
                }
                if (bestAccuracy < currAccuracy)
                {
                    Debug.WriteLine("Got " + currAccuracy / labels.Length + " with pow = " + pow);
                    bestAccuracy = currAccuracy;
                    bestPow = pow;
                }
            }

            param.C = Math.Pow(2.0, bestPow);

            // Train up the model!
            svm_model model = svm.svm_train(prob, param);

            learnOutput.Text = "Training finished in " + (DateTime.Now - start);
            return model;
        }

        public double evaluate(svm_node[] nodes, svm_model model, ref string resultStr)
        {
            // We're hardcoding in the number of classes here
            int totalClasses = 2;
            int[] labels = new int[totalClasses];

            // Get the labels (in this case, [0,1])
            svm.svm_get_labels(model, labels);

            // This stores the probability estimates for each class
            double[] prob_estimates = new double[totalClasses];

            // Actually get the probabilities
            double v = svm.svm_predict_probability(model, nodes, prob_estimates);

            // For each class, print out the probability that it was that class
            for (int i = 0; i < totalClasses; i++)
            {
                resultStr += "  (" + labels[i] + ": " + prob_estimates[i] + ")\n";
            }
            resultStr += "(Predicted Label:" + v + ")\n";

            // Return the most likely class
            return v;
        }
        #endregion



        //new version of this function for images
        private async void uploadFile(string strSaveName, Stream fileStream)
        {

            try
            {
                //LiveOperationResult uploadOperation = await client.BackgroundUploadAsync("me/skydrive", new Uri("/shared/transfers/" + strSaveName, UriKind.Relative), OverwriteOption.Overwrite);
                LiveOperationResult uploadOperation = await this.client.UploadAsync("me/skydrive", strSaveName, fileStream, OverwriteOption.Overwrite);
                //LiveOperationResult uploadResult = await uploadOperation.StartAsync();
                textOutput.Text = "File " + strSaveName + " uploaded";
            }

            catch (Exception ex)
            {
                textOutput.Text = "Error uploading photo: " + ex.Message;
            }

        }

        #region Audio Functions
        
        /// <summary>
        /// Updates the XNA FrameworkDispatcher and checks to see if a sound is playing.
        /// If sound has stopped playing, it updates the UI.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        void dt_Tick(object sender, EventArgs e)
        {
            try { FrameworkDispatcher.Update(); }
            catch { }
        }

        /// <summary>
        /// The Microphone.BufferReady event handler.
        /// Gets the audio data from the microphone and stores it in a buffer,
        /// then writes that buffer to a stream for later playback.
        /// Any action in this event handler should be quick!
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        void microphone_BufferReady(object sender, EventArgs e)
        {
            // Retrieve audio data
            microphone.GetData(buffer);

            //scale the audio up
            var tempArray = buffer;
            for (int i = 0; i < tempArray.Length; i++)
            {
                byte tempOverflow = (byte)((int)tempArray[i] * (int)volumeScale);
                if (tempArray[i] < tempOverflow)
                {
                    tempArray[i] = tempOverflow;
                }
            }

            // Store the audio data in a stream
            //stream.Write(buffer, 0, buffer.Length);
            stream.Write(tempArray, 0, tempArray.Length);

        }

        private void SaveToIsolatedStorage()
        {
            // first, we grab the current apps isolated storage handle
            IsolatedStorageFile isf = IsolatedStorageFile.GetUserStoreForApplication();

            // we give our file a filename
            strSaveName = uploadPrefix+"audio_" + DateTime.Now.ToString("yy_MM_dd_hh_mm_ss") + ".wav";

            // if that file exists... 
            if (isf.FileExists(strSaveName))
            {
                // then delete it
                isf.DeleteFile(strSaveName);
            }

            // now we set up an isolated storage stream to point to store our data
            IsolatedStorageFileStream isfStream =
                     new IsolatedStorageFileStream(strSaveName,
                     FileMode.Create, IsolatedStorageFile.GetUserStoreForApplication());

            isfStream.Write(stream.ToArray(), 0, stream.ToArray().Length);

            // ok, done with isolated storage... so close it
            isfStream.Close();
        }

        public void UpdateWavHeader(Stream stream)
        {
            if (!stream.CanSeek) throw new Exception("Can't seek stream to update wav header");

            var oldPos = stream.Position;

            // ChunkSize 36 + SubChunk2Size
            stream.Seek(4, SeekOrigin.Begin);
            stream.Write(BitConverter.GetBytes((int)stream.Length - 8), 0, 4);

            // Subchunk2Size == NumSamples * NumChannels * BitsPerSample/8 This is the number of bytes in the data.
            stream.Seek(40, SeekOrigin.Begin);
            stream.Write(BitConverter.GetBytes((int)stream.Length - 44), 0, 4);

            stream.Seek(oldPos, SeekOrigin.Begin);
        }

        public void WriteWavHeader(Stream stream, int sampleRate)
        {
            const int bitsPerSample = 16;
            const int bytesPerSample = bitsPerSample / 8;
            var encoding = System.Text.Encoding.UTF8;

            // ChunkID Contains the letters "RIFF" in ASCII form (0x52494646 big-endian form).
            stream.Write(encoding.GetBytes("RIFF"), 0, 4);

            // NOTE this will be filled in later
            stream.Write(BitConverter.GetBytes(0), 0, 4);

            // Format Contains the letters "WAVE"(0x57415645 big-endian form).
            stream.Write(encoding.GetBytes("WAVE"), 0, 4);

            // Subchunk1ID Contains the letters "fmt " (0x666d7420 big-endian form).
            stream.Write(encoding.GetBytes("fmt "), 0, 4);

            // Subchunk1Size 16 for PCM. This is the size of therest of the Subchunk which follows this number.
            stream.Write(BitConverter.GetBytes(16), 0, 4);

            // AudioFormat PCM = 1 (i.e. Linear quantization) Values other than 1 indicate some form of compression.
            stream.Write(BitConverter.GetBytes((short)1), 0, 2);

            // NumChannels Mono = 1, Stereo = 2, etc.
            stream.Write(BitConverter.GetBytes((short)1), 0, 2);

            // SampleRate 8000, 44100, etc.
            stream.Write(BitConverter.GetBytes(sampleRate), 0, 4);

            // ByteRate = SampleRate * NumChannels * BitsPerSample/8
            stream.Write(BitConverter.GetBytes(sampleRate * bytesPerSample), 0, 4);

            // BlockAlign NumChannels * BitsPerSample/8 The number of bytes for one sample including all channels.
            stream.Write(BitConverter.GetBytes((short)(bytesPerSample)), 0, 2);

            // BitsPerSample 8 bits = 8, 16 bits = 16, etc.
            stream.Write(BitConverter.GetBytes((short)(bitsPerSample)), 0, 2);

            // Subchunk2ID Contains the letters "data" (0x64617461 big-endian form).
            stream.Write(encoding.GetBytes("data"), 0, 4);

            // NOTE to be filled in later
            stream.Write(BitConverter.GetBytes(0), 0, 4);
        }

        private void skydrive_SessionChanged(object sender, LiveConnectSessionChangedEventArgs e)
        {

            if (e != null && e.Status == LiveConnectSessionStatus.Connected)
            {
                this.client = new LiveConnectClient(e.Session);
                this.GetAccountInformations();

                // small files but via 3G and on Battery
                this.client.BackgroundTransferPreferences = BackgroundTransferPreferences.AllowCellularAndBattery;

            }
            else
            {
                this.client = null;
                textOutput.Text = e.Error != null ? e.Error.ToString() : string.Empty;
            }

        }

        private async void GetAccountInformations()
        {
            try
            {
                LiveOperationResult operationResult = await this.client.GetAsync("me");
                var jsonResult = operationResult.Result as dynamic;
                string firstName = jsonResult.first_name ?? string.Empty;
                string lastName = jsonResult.last_name ?? string.Empty;
                textOutTranscript.Text = "Welcome " + firstName + " " + lastName;
                loggedIn = true;
            }
            catch (Exception e)
            {
                textOutTranscript.Text = e.ToString();
            }
        }

        private async void uploadFile()
        {

            using (IsolatedStorageFile store = IsolatedStorageFile.GetUserStoreForApplication())
            {
                using (var fileStream = store.OpenFile(strSaveName, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    try
                    {
                        uploadComplete = false;

                        //LiveOperationResult uploadOperation = await client.BackgroundUploadAsync("me/skydrive", new Uri("/shared/transfers/" + strSaveName, UriKind.Relative), OverwriteOption.Overwrite);
                        LiveOperationResult uploadOperation = await this.client.UploadAsync("me/skydrive", strSaveName, fileStream, OverwriteOption.Overwrite);
                        //LiveOperationResult uploadResult = await uploadOperation.StartAsync();
                        if (uploadOperation.Result != null)
                        {
                            uploadComplete = true;
                        }
                        Dispatcher.BeginInvoke(() =>
                        {
                            this.textOutput.Text = "File " + strSaveName + " uploaded";
                        });

                    }

                    catch (Exception ex)
                    {

                        Dispatcher.BeginInvoke(() =>
                        {
                            this.textOutput.Text = "Error uploading photo: " + ex.Message;
                        });

                    }
                }
            }
        }

        ///////this is the google code part
        protected override void OnNavigatedTo(NavigationEventArgs e)
        {
            base.OnNavigatedTo(e);

            // Setup SoundIO right away
            sio.audioInEvent += sio_audioInEvent;
            sio.start();
        }

        void sio_audioInEvent(float[] data)
        {
            // Only do something if we're recording right now
            if (this.recording)
            {
                // If we are recording, throw our data into our recordedAudio list
                recordedAudio.Add(data);

                // If we're reached our maximum recording limit....
                if (recordedAudio.Count == 10000) //10 sec worth
                {
                    // We stop ourselves! :P
                    stopRecording();
                }
            }

            // If we haven't already initialized our fftw object, do so with the length of the data we will be analyzing
            if (fftw == null)
                fftw = new FFTW((uint)data.Length);

            //Find the Frequency from the FFT result bins
            //freqMax = bin * sampleRate / numSamplesToWrite;
            uint binLow = 400 * (uint)data.Length / 48000; //low frequency of interest is 400
            uint binHigh = 2000 * (uint)data.Length / 48000; //high frequency of interest is 2000

            // Calculate a Complex array!  What fun!
            Complex[] DATA = fftw.fft(data);

            //if (this.recording == false && this.processing == false && loggedIn == true && uploadComplete == true)
            if (this.recording == false && loggedIn == true)
            {
                for (uint i = binLow; i < binHigh; i++)
                {
                    if ((DATA[i].real * DATA[i].real + DATA[i].imag * DATA[i].imag) > 60)
                    {

                        playAudio();

                        Dispatcher.BeginInvoke(() =>
                        {
                            textOutput.Text = "Trigger Freq: " + i * 48000 / (uint)data.Length;
                        });
                        startRecording();
                        break;
                    }
                }
            }

            if (this.recording == true)
            {
                for (uint i = binLow; i < binHigh; i++)
                {
                    if (((DATA[i].real * DATA[i].real + DATA[i].imag * DATA[i].imag) > 60))
                    {
                        quietCount = 0;
                        break;
                    }
                }

                quietCount++;

                if (quietCount > 1000)
                {
                    quietCount = 0;
                    Dispatcher.BeginInvoke(() =>
                    {
                        textOutput.Text = "Audio no longer detected";
                    });

                    stopRecording();
                }

            }

        }

        private void playAudio()
        {

            // Play the audio in a new thread so the UI can update.
            Thread soundThread = new Thread(new ThreadStart(playSound));
            soundThread.Start();
        }

        private void playSound()
        {
            // Play audio using SoundEffectInstance so we can monitor it's State 
            // and update the UI in the dt_Tick handler when it is done playing.         
            //SoundEffect sound = new SoundEffect(info, microphone.SampleRate, AudioChannels.Mono);
            SoundEffect sound;
            //sound = Content.Load<SoundEffect>("Hellooo");
            //sound.Play();
        }


        // This gets called when the button gets pressed while it says "Go"
        private void startRecording()
        {
            //m_d3dInterop.pauseVideo();

            // Get audio data in 1/2 second chunks
            microphone.BufferDuration = TimeSpan.FromMilliseconds(500);

            // Allocate memory to hold the audio data
            buffer = new byte[microphone.GetSampleSizeInBytes(microphone.BufferDuration)];

            // Set the stream back to zero in case there is already something in it
            stream.SetLength(0);

            WriteWavHeader(stream, microphone.SampleRate);

            // Start recording
            microphone.Start();

            this.recording = true;

            // Do this in a Dispatcher.BeginInvoke since we're not certain which thread is calling this function!
            Dispatcher.BeginInvoke(() =>
            {
                this.textOutTranscript.Text = "Recording...";

            });
        }

        // This gets called when the button gets pressed while it says "Stop" or when we reach
        // our maximum buffer amount (set to 10 seconds right now)
        private void stopRecording()
        {
            microphone.Stop();
            UpdateWavHeader(stream);
            SaveToIsolatedStorage();

            Dispatcher.BeginInvoke(() =>
            {
                this.textOutTranscript.Text = "Uploading File...";
            });
            uploadFile();
            //recordedAudio.Clear();

            this.recording = false;

            // Do this in a Dispatcher.BeginInvoke since we're not certain which thread is calling this function!
            Dispatcher.BeginInvoke(() =>
            {
                this.textOutput.Text = "Processing...";
                processData();
            });

        }

        private void CopyToIsolatedStorage()
        {
            return; //I don't have hello.wav right now
            using (IsolatedStorageFile storage = IsolatedStorageFile.GetUserStoreForApplication())
            {
                string[] files = new string[] { "Hellooo.wav" };

                foreach (var _fileName in files)
                {
                    if (!storage.FileExists(_fileName))
                    {
                        string _filePath = "Audio/" + _fileName;
                        StreamResourceInfo resource = Application.GetResourceStream(new Uri(_filePath, UriKind.Relative));

                        using (IsolatedStorageFileStream file = storage.CreateFile(_fileName))
                        {
                            int chunkSize = 4096;
                            byte[] bytes = new byte[chunkSize];
                            int byteCount;

                            while ((byteCount = resource.Stream.Read(bytes, 0, chunkSize)) > 0)
                            {
                                file.Write(bytes, 0, byteCount);
                            }
                        }
                    }
                }
            }
        }




        // This is a utility to take a list of arrays and mash them all together into one large array
        private T[] flattenList<T>(List<T[]> list)
        {
            // Calculate total size
            int size = 0;
            foreach (var el in list)
            {
                size += el.Length;
            }

            // Allocate the returning array
            T[] ret = new T[size];


            // Copy each chunk into this new array
            int idx = 0;
            foreach (var el in list)
            {
                el.CopyTo(ret, idx);
                idx += el.Length;
            }


            // Return the "flattened" array
            return ret;
        }

        private async void processData()
        {
            this.processing = true;

            // First, convert our list of audio chunks into a flattened single array
            float[] rawData = flattenList(recordedAudio);

            // Once we've done that, we can clear this out no problem
            recordedAudio.Clear();

            // Next, convert the data into FLAC:
            byte[] flacData = null;
            flacData = lf.compressAudio(rawData, sio.getInputSampleRate(), sio.getInputNumChannels());

            // Upload it to the server and get a response!
            RecognitionResult result = await recognizeSpeech(flacData, sio.getInputSampleRate());


            if (result.result != null)
            {
                // Check to make sure everything went okay, if it didn't, check the debug log!
                if (result.result.Count != 0)
                {
                    // This is just some fancy code to display each hypothesis as sone text that gets redder
                    // as our confidence goes down; note that I've never managed to get multiple hypotheses
                    this.textOutTranscript.Inlines.Clear();
                    foreach (var alternative in result.result[0].alternative)
                    {
                        Run run = new Run();
                        run.Text = alternative.transcript + "\n\n";
                        byte bg = (byte)(alternative.confidence * 255);
                        run.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 255, bg, bg));
                        textOutTranscript.Inlines.Add(run);
                        if (alternative.confidence * 100 + 1 > 0)
                        {
                            googleText = alternative.transcript;
                            IsolatedStorageFile store = IsolatedStorageFile.GetUserStoreForApplication();
                            // we give our file a filename
                            string strSaveNameTemp = uploadPrefix + "transcript_" + DateTime.Now.ToString("yy_MM_dd_hh_mm_ss") + ".txt";

                            // Declare a new StreamWriter.
                            StreamWriter writer = null;

                            // Assign the writer to the store and the file TestStore.
                            writer = new StreamWriter(new IsolatedStorageFileStream(
                                strSaveNameTemp, FileMode.CreateNew, store));

                            // Have the writer write "Hello Isolated Storage" to the store.
                            writer.WriteLine(googleText);

                            writer.Close();

                            var fileStreamTemp = store.OpenFile(strSaveNameTemp, FileMode.Open, FileAccess.Read, FileShare.Read);

                            uploadFile(strSaveNameTemp, fileStreamTemp);
                            textOutTranscript.Text = "Transcript Uploaded";
                            break;
                        }
                    }

                }
                else
                {
                    textOutTranscript.Text = "Errored out!";
                }
            }
            else
            {
                textOutTranscript.Text = "Google return bad!";
            }

            this.processing = false;

        }

        private async Task<RecognitionResult> recognizeSpeech(byte[] flacData, uint sampleRate)
        {
            try
            {
                // Construct our HTTP request to the server
                string url = "https://www.google.com/speech-api/v2/recognize?output=json&lang=en-us&key=AIzaSyC-YKuxG4Pe5Xg1veSXtPPt3S3aKfzXDTM";
                HttpWebRequest request = WebRequest.CreateHttp(url);

                // Make sure we tell it what kind of data we're sending
                request.ContentType = "audio/x-flac; rate=" + sampleRate;
                request.Method = "POST";

                // Actually write the data out to the stream!
                using (var stream = await Task.Factory.FromAsync<Stream>(request.BeginGetRequestStream, request.EndGetRequestStream, null))
                {
                    await stream.WriteAsync(flacData, 0, flacData.Length);
                }

                // We are going to store our json response into this RecognitionResult:
                RecognitionResult root = null;

                // Now, we wait for a response and read it in:
                using (var response = await Task.Factory.FromAsync<WebResponse>(request.BeginGetResponse, request.EndGetResponse, null))
                {
                    // Construct a datareader so we can read everything in as a string
                    DataReader dr = new DataReader(response.GetResponseStream().AsInputStream());

                    dr.InputStreamOptions = InputStreamOptions.Partial;

                    uint datalen = await dr.LoadAsync(1024 * 1024);
                    string responseStringsCombined = dr.ReadString(datalen);

                    // Split this response string by its newlines
                    var responseStrings = responseStringsCombined.Split(new string[] { "\r\n", "\n" }, StringSplitOptions.None);

                    // Now, inspect the JSON of each string
                    foreach (var responseString in responseStrings)
                    {
                        root = JsonConvert.DeserializeObject<RecognitionResult>(responseString);

                        // If this is a good result
                        if (root.result.Count != 0)
                        {
                            //return it!
                            return root;
                        }
                    }
                }

                // Aaaaand, return the root object!
                return root;
            }
            catch (Exception e)
            {
                Debug.WriteLine("Error detected!  Exception thrown: " + e.Message);
            }

            // Otherwise, something failed, and we don't know what!
            return new RecognitionResult();
        }


        #endregion
    }       
}