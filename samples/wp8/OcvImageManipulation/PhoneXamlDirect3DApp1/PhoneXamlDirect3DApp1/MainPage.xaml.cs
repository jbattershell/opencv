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

namespace PhoneXamlDirect3DApp1
{
    public partial class MainPage : PhoneApplicationPage
    {
        private Direct3DInterop m_d3dInterop = new Direct3DInterop();
        private DispatcherTimer m_timer;
        MediaLibrary library = new MediaLibrary();
        private bool motionCaptureEnabled;
        
        private int trainingMode;   //0=not training, 1=positive training, 2=negative training
            const int NOTTRAINING = 0;
            const int POSITIVETRAINING = 1;
            const int NEGATIVETRAINING = 2;
            const int SAMPLESTOTRAIN = 10;


        // This is the SVM model we will eventually train up
        svm_model model = null;
        // This is the buffer of audio data we will perform recognition on
        svm_node[] recogBuff = null;

        // These are the list of recording objects
        ObservableCollection<float> posRecordings = new ObservableCollection<float>();
        ObservableCollection<float> negRecordings = new ObservableCollection<float>();

        // Constructor
        public MainPage()
        {
            InitializeComponent();
            m_timer = new DispatcherTimer();
            m_timer.Interval = new TimeSpan(0, 0, 0, 0, 250);   //trigger timer up to 4x per second
            m_timer.Tick += new EventHandler(timer_Tick);
            m_timer.Start();

            motionCaptureEnabled = false;
            trainingMode = NOTTRAINING;
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
            m_d3dInterop.OnFrameReady += m_d3dInterop_OnFrameReady;
        }

        void m_d3dInterop_OnFrameReady(float lowbin)
        {
            //Learned thresh eval
            if (model != null)
            {
                // Copy in the latest data to recogBuff
                double[] featureData = computeFeatureVector(lowbin);
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

                // Let's try our hand at some recognition!
                DateTime startTime = DateTime.Now;

                string resultStr = "";
                double val = evaluate(recogBuff, model, ref resultStr);

                if (val == 1.0)
                    m_d3dInterop.SetMotionDetected(true);
                else
                    m_d3dInterop.SetMotionDetected(false);

            }
        }

        void m_d3dInterop_OnCaptureFrameReady(int[] data, int cols, int rows)
        {
            m_d3dInterop.ResetCapture();
            Dispatcher.BeginInvoke(() =>
            {
                WriteableBitmap wb = new WriteableBitmap(cols, rows);
                Array.Copy(data, wb.Pixels, data.Length);

                var fileStream = new MemoryStream();
                wb.SaveJpeg(fileStream, wb.PixelWidth, wb.PixelHeight, 100, 100);
                fileStream.Seek(0, SeekOrigin.Begin);

                string name = "motion "+ DateTime.Now.ToString("yy_MM_dd_hh_mm_ss_fff");

               library.SavePictureToCameraRoll(name, fileStream);
            });

        }


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
                    trainingMode = POSITIVETRAINING;

                    break;

                case "TrainingNeg":
                    negRecordings.Clear();
                    trainingMode = NEGATIVETRAINING;

                    break;
            }
        }

        private void RadioButton_Checked(object sender, RoutedEventArgs e)
        {
            RadioButton rb = sender as RadioButton;
            switch (rb.Name)
            {
               
                case "Motion":
                    motionCaptureEnabled = true;
                    break;

                case "MotionOff":
                    motionCaptureEnabled = false;
                    break;
            }
        }

        private void timer_Tick(object sender, EventArgs e)
        {

            //////////////////////////////////
            //Capture training data

            if (negRecordings.Count >= SAMPLESTOTRAIN && trainingMode == NEGATIVETRAINING)
            {
                //training done
                Dispatcher.BeginInvoke(() =>
                {
                    trainingMode = NOTTRAINING;
                    learnOutput.Text = "Negative Samples Recorded";

                });
            }
            else if (posRecordings.Count >= SAMPLESTOTRAIN && trainingMode == POSITIVETRAINING)
            {
                //training done
                Dispatcher.BeginInvoke(() =>
                {
                    trainingMode = NOTTRAINING;
                    learnOutput.Text = "Positive Samples Recorded";

                });
            }
            else if (m_d3dInterop != null && posRecordings != null && negRecordings != null && trainingMode == POSITIVETRAINING && posRecordings.Count < SAMPLESTOTRAIN)
            {
                posRecordings.Add(m_d3dInterop.LowMotionBins());
            }
            else if (m_d3dInterop != null && posRecordings != null && negRecordings != null && trainingMode == NEGATIVETRAINING && negRecordings.Count < SAMPLESTOTRAIN)
            {
                //Only records a certain number of frames
                negRecordings.Add(m_d3dInterop.LowMotionBins());
            }

            ///////////////////////////

            try
            {
                // These are TextBlock controls that are created in the page’s XAML file.  
                float value = DeviceStatus.ApplicationCurrentMemoryUsage / (1024.0f * 1024.0f);
                MemoryTextBlock.Text = value.ToString();
                value = DeviceStatus.ApplicationPeakMemoryUsage / (1024.0f * 1024.0f);
                PeakMemoryTextBlock.Text = value.ToString();

                MotionOutput.Text = m_d3dInterop.LowMotionBins().ToString();
                LearnedOutput.Text = m_d3dInterop.MotionStatus().ToString();

            }
            catch (Exception ex)
            {
                MemoryTextBlock.Text = ex.Message;
            }

            if (ImageThresh != null)    //update text
            {
                ImageThresVal.Text = ((int)ImageThresh.Value).ToString();

            }

            if (motionCaptureEnabled)
            {
                m_d3dInterop.SetCapture();
            }
        }

        private void ImageThresh_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (ImageThresh != null)
            {
                m_d3dInterop.SetImageThreshold((int)ImageThresh.Value);
                ImageThresVal.Text = ((int)ImageThresh.Value).ToString();
            }
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

    }       
}