using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using System.Dynamic;
using System.Diagnostics;
using RealTimePatternRec.DataLogging;
using System.Threading;

using Accord.Controls;
using Accord.IO;
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Accord.Statistics.Analysis;

using Microsoft.ML;
using Microsoft.ML.Data;

// useful: https://dotnetfiddle.net/
// potential library: http://accord-framework.net/

namespace RealTimePatternRec.PatternRec
{
    /// <summary>
    /// class to hold all information from a data recording session
    /// </summary>
    public class Data
    {
        // Class to hold all information related to the data being used by the pattern rec class
        public List<double> timestamps = new List<double>();
        public List<List<double>> inputs = new List<List<double>>();    // raw input values
        public List<int> outputs = new List<int>(); // raw output values
        public List<int> input_types = new List<int>();     // types of input signals (0 = generic, 1 = emg)
        public List<List<double>> features = new List<List<double>>();  // inputs after mapping to features
        public List<int> feature_outputs = new List<int>(); // outputs trimmed to account for windowing
        public List<bool> input_active_flags = new List<bool>();    //  indicating which inputs are active in model
        public List<string> output_labels = new List<string>(); // list of output label names
        public List<string> input_labels = new List<string>();  // list of input label names
        public int input_num;   // number of inputs in data
        public int output_num;  // number of outputs in data
        public int freq;    // frequency data was recorded at
        public int collection_cycles;   // number of times outputs were iterated over during data collection
        public int contraction_time;    // number of seconds per contraction during data collection
        public int relaxation_time; // number of seconds for relaxation during data collection
        public string json_filepath;  // path to json data configuration file
        public string csv_filepath;     // path to csv data file

        public void Clear()
        {
            timestamps.Clear();
            inputs.Clear();
            outputs.Clear();
            features.Clear();
            feature_outputs.Clear();
        }

        /// <summary>
        /// sets the input types based on specified list of tuples
        /// </summary>
        /// <param name="map"> A list of tuples that map input titles to values</param>
        public void SetInputTypes(List<Tuple<string, int>> map)
        {
            input_types.Clear();
            for (int i=0; i<input_labels.Count; i++)
            {
                int value = 0;
                foreach (Tuple<string, int> pair in map)
                {
                    if (input_labels[i].ToLower().Contains(pair.Item1))
                    {
                        value = pair.Item2;
                    }
                }
                input_types.Add(value);
            }
        }

        public void SetInputActiveFlags(bool flag)
        {
            input_active_flags.Clear();
            for (int i = 0; i < input_labels.Count; i++)
            {
                input_active_flags.Add(flag);
            }
        }

        /// <summary>
        /// loads data file into inputs and outputs fields
        /// </summary>
        /// <param name="filepath"></param>
        /// <returns>0 if successful, 1 if not successful</returns>
        public int LoadFileToListCols(string filepath)
        {
            string[] lines_arr = System.IO.File.ReadAllLines(filepath);
            List<string> lines = lines_arr.ToList();

            // if number of inputs in data file does not match set number of inputs
            if (input_num != lines[0].Split(',').Length - 2)
            {
                return 1;
            }

            // clear inputs/output lists and preallocate inputs
            Clear();
            for (int i = 0; i < input_num; i++)
            {
                inputs.Add(new List<double>());
            }

            foreach (string line in lines)
            {
                // parse data lines
                List<double> vals = Array.ConvertAll(line.Split(','), Double.Parse).ToList();
                outputs.Add((int)vals.Last());
                vals.RemoveAt(vals.Count - 1);

                timestamps.Add(vals.First());
                vals.RemoveAt(0);

                for (int i = 0; i < vals.Count; i++)
                {
                    inputs[i].Add(vals[i]);
                }
            }

            return 0;
        }

        /// <summary>
        /// shuffles training data features and outputs (i.e. inputs/outputs passed by reference)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        public static void shuffle_training_data<T>(List<List<T>> inputs, List<int> outputs)
        {

            Random rng = new Random();
            int n = outputs.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);

                // swap all data.inputs
                for (int i = 0; i < inputs.Count; i++)
                {
                    T value_T = inputs[i][k];
                    inputs[i][k] = inputs[i][n];
                    inputs[i][n] = value_T;
                }

                int value = outputs[k];
                outputs[k] = outputs[n];
                outputs[n] = value;
            }

            return;
        }

        /// <summary>
        /// Transposes a list of lists (i.e. switches rows/columns)
        /// </summary>
        /// <typeparam name="T"> any primitive data type</typeparam>
        /// <param name="list"> list of lists to transpose</param>
        /// <returns></returns>
        public static List<List<T>> transpose_list_list<T>(List<List<T>> list)
        {
            // https://stackoverflow.com/questions/39484996/rotate-transposing-a-listliststring-using-linq-c-sharp
            List<List<T>> inverted_list = list
                        .SelectMany(inner => inner.Select((item, index) => new { item, index }))
                        .GroupBy(i => i.index, i => i.item)
                        .Select(g => g.ToList())
                        .ToList();            // *cries in LINQ*

            return inverted_list;
        }
    }

    public static class Fiters
    {
        static public List<double> MinMaxScaling(List<double> raw_values)
        {
            List<double> filtered_values = new List<double>();

            double min_value = raw_values.Min();
            double max_value = raw_values.Max();

            for (int i = 0; i < raw_values.Count; i++)
            {
                double new_value = (raw_values[i] - min_value) / (max_value - min_value);
                filtered_values.Add(new_value);
            }
            return filtered_values;
        }

        static public List<double> ZeroCentering(List<double> raw_values)
        {
            List<double> filtered_values = new List<double>();

            double mean_value = raw_values.Average();

            for (int i = 0; i < raw_values.Count; i++)
            {
                double new_value = raw_values[i] - mean_value;
                filtered_values.Add(new_value);
            }
            return filtered_values;
        }
    }

    /// <summary>
    /// static class of EMG and other feature mapping methods for time series data structured as Lists
    /// </summary>
    /// 
    // list of sources:
    // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7250028/
    // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3821366/
    // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5017469/
    // https://doi.org/10.1016/j.eswa.2012.01.102

    public static class Features
    {

        /// <summary>
        /// Returns the windowed raw value (downsamples to match other windowed features)
        /// </summary>
        /// <param name="raw_values"></param>
        /// <param name="window_size_n"></param>
        /// <param name="window_overlap_n"></param>
        /// <returns></returns>
        static public List<double> RAW(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                filtered_values.Add(raw_values[i]);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed mean value
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns></returns>
        static public List<double> MV(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);
                filtered_values.Add(sub_window.Sum() / window_size_n);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed integrated emg value
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> IEMG(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double sum = 0;
                for (int j = 0; j < window_size_n; j++)
                {
                    sum += Math.Abs(sub_window[j]);
                }
                filtered_values.Add(sum);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed mean absolute value
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> MAV(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double sum = 0;
                for (int j = 0; j < sub_window.Count; j++)
                {
                    sum += Math.Abs(sub_window[j]);
                }
                filtered_values.Add(sum / window_size_n);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed modified mean absolute value 1
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> MMAV1(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double sum = 0;
                for (int j = 0; j < window_size_n; j++)
                {
                    double w;
                    if (j >= 0.25 * window_size_n && j <= 0.75 * window_size_n)
                    {
                        w = 1;
                    }
                    else
                    {
                        w = 0.5;
                    }
                    sum += (w * Math.Abs(sub_window[j]));
                }
                filtered_values.Add(sum / window_size_n);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed modified mean absolute value 2
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> MMAV2(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double sum = 0;
                for (int j = 0; j < sub_window.Count; j++)
                {
                    double w;
                    if (j < 0.25 * window_size_n)
                    {
                        w = 4 * j / window_size_n;
                    }
                    else if (j >= 0.25 * window_size_n && j <= 0.75 * window_size_n)
                    {
                        w = 1;
                    }
                    else
                    {
                        w = 4*(j-window_size_n)/window_size_n;
                    }
                    sum += (w * Math.Abs(sub_window[j]));
                }
                filtered_values.Add(sum / window_size_n);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the winowed simple square integral value
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> SSI(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double sum = 0;
                for (int j = 0; j < window_size_n; j++)
                {
                    sum += Math.Pow(sub_window[j], 2);
                }
                filtered_values.Add(sum);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed variance
        /// </summary>
        /// <param name="raw_values"></param>
        /// <param name="window_size_n"></param>
        /// <param name="window_overlap_n"></param>
        /// <returns></returns>
        static public List<double> VAR(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();
            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double variance = 0;
                for (int j = 0; j < window_size_n; j++)
                {
                    variance += Math.Pow(sub_window[j], 2); ;
                }
                filtered_values.Add(variance / (window_size_n - 1));
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed number of zero crossings
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns></returns>
        static public List<double> ZC(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);
                int crossing_count = 0;
                bool positive_flag = (sub_window[0] >= 0);    // true for 
                for (int j = 0; j < sub_window.Count; j++)
                {
                    if ((sub_window[j] >= 0) ^ positive_flag)
                    {
                        crossing_count++;
                        positive_flag = !positive_flag;
                    }
                }
                filtered_values.Add(crossing_count);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed number of slope sign changes
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns></returns>
        static public List<double> SSC(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> difference_values = raw_values.Zip(raw_values.Skip(1), (x, y) => y - x).ToList();
            difference_values.Add(difference_values.Last());    // append final value since one is lost during difference calculation, keeps feature sizes consistent
            List<double> filtered_values = ZC(difference_values, window_size_n, window_overlap_n);
            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed waveform length
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns></returns>
        static public List<double> WL(List<double> raw_values, int window_size_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();
            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);
                double length = 0;
                for (int j=1; j<sub_window.Count; j++)
                {
                    length += Math.Abs(sub_window[j] - sub_window[j-1]);
                }
                filtered_values.Add(length);
            }
            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed wilson amplitude
        /// </summary>
        /// <param name="raw_values"></param>
        /// <param name="window_size_n"></param>
        /// <param name="window_overlap_n"></param>
        /// <param name="threshold"></param>
        /// <returns></returns>
        static public List<double> WAMP(List<double> raw_values, int window_size_n, int window_overlap_n, double threshold)
        {
            List<double> filtered_values = new List<double>();
            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double wilson_amplitude = 0;
                for (int j = 1; j < sub_window.Count; j++)
                {
                    if (sub_window[j] > threshold)
                    {
                        wilson_amplitude += Math.Abs(sub_window[j] - sub_window[j - 1]);
                    }
                }
                filtered_values.Add(wilson_amplitude);
            }

            return filtered_values;
        }

    }

    /// <summary>
    /// class to hold all pattern recognition model information and capabilities
    /// </summary>
    public class Model
    {
        public Data data;
        public dataLogger logger;

        public int window_time;
        public int window_overlap; 
        public int window_size_n;   // number of timesteps per window
        public int window_overlap_n;    // number of timesteps to overalp
        public int window_n;    // number of windows per model input
        public double train_test_split;
        public double accuracy;

        //public bool modelFlag = false;
        public bool realtimeFlag = false;
        //public string model_type;
        public Dictionary<string, string> model_params;

        public delegate List<double> pipeline_func(List<double> data_);

        public List<pipeline_func> generic_feature_pipeline = new List<pipeline_func>();
        public List<pipeline_func> emg_feature_pipeline = new List<pipeline_func>();
        public List<string> generic_feature_pipeline_titles = new List<string>();
        public List<string> emg_feature_pipeline_titles = new List<string>();

        public List<pipeline_func> generic_filter_pipeline = new List<pipeline_func>();
        public List<pipeline_func> emg_filter_pipeline = new List<pipeline_func>();
        public List<string> generic_filter_pipeline_titles = new List<string>();
        public List<string> emg_filter_pipeline_titles = new List<string>();

        public string model_save_filepath;
        public dynamic learner;

        public IPredictor model;

        public Model()
        {
            data = new Data();
        }

        public List<List<double>> map_features(List<List<double>> temp_inputs, List<int> input_types, List<bool> input_active_flags)
        {
            // map inputs "temp_inputs" to features using the set "emg_feature_pipeline" and "generic_feature_pipeline"
            // function lists.  Also requires that "data.input_types" are set and "data.input_active_flags" are set

            List<List<double>> temp_features = new List<List<double>>();

            for (int i = 0; i < input_types.Count; i++)
            {
                if (input_types[i] == 0 && input_active_flags[i] == true)
                {
                    foreach (Model.pipeline_func f in generic_feature_pipeline)
                    {
                        temp_features.Add(f(temp_inputs[i]));
                    }
                }
                else if (input_types[i] == 1 && input_active_flags[i] == true)
                {
                    foreach (Model.pipeline_func f in emg_feature_pipeline)
                    {
                        temp_features.Add(f(temp_inputs[i]));
                    }
                }
            }

            return temp_features;
        }  

        public void map_features_training()
        {
            data.features.Clear();
            data.feature_outputs.Clear();

            // get indices of output changes
            List<int> output_change_indices = new List<int>();
            output_change_indices.Add(0);
            int cur_output = data.outputs[0];
            for (int i = 1; i < data.outputs.Count; i++)
            {
                if (cur_output != data.outputs[i])
                {
                    output_change_indices.Add(i);
                    cur_output = data.outputs[i];
                }
            }
            output_change_indices.Add(data.outputs.Count);

            int start_ind = output_change_indices[0];
            for (int i = 1; i < output_change_indices.Count; i++)
            {
                int output_value = data.outputs[output_change_indices[i - 1]];
                int end_ind = output_change_indices[i];

                List<List<double>> temp_input = Data.transpose_list_list(Data.transpose_list_list(data.inputs).GetRange(start_ind, end_ind - start_ind));
                List<List<double>> temp_features = map_features(temp_input, data.input_types, data.input_active_flags);

                if (data.features.Count == 0)
                {
                    for (int j = 0; j < temp_features.Count; j++)
                    {
                        data.features.Add(new List<double>());
                    }
                }
                for (int j = 0; j < temp_features.Count; j++)
                {
                    data.features[j].AddRange(temp_features[j]);
                }

                data.feature_outputs.AddRange(Enumerable.Repeat(output_value, temp_features[0].Count));
                start_ind = end_ind;
            }
        }

        public void train_model()
        {
            // map inputs to features and shuffle
            map_features_training();
            Data.shuffle_training_data(data.features, data.feature_outputs);

            // split to test/train set
            List<List<double>> features_rows = Data.transpose_list_list(data.features);
            int N_train = (int)(features_rows.Count * (1 - train_test_split));

            List<List<double>> training_features = features_rows.GetRange(0, N_train);
            List<int> training_outputs = data.feature_outputs.GetRange(0, N_train);

            List<List<double>> testing_features = features_rows.GetRange(N_train, features_rows.Count - N_train);
            List<int> testing_outputs = data.feature_outputs.GetRange(N_train, features_rows.Count - N_train);

            // train model
            model.train(training_features.Select(a => a.ToArray()).ToArray(), training_outputs.ToArray());

            // assess model
            int[] answers = new int[testing_features.Count];
            for (int i=0; i<testing_features.Count; i++)
            {
                double[] scores = model.predict(testing_features[i].ToArray());
                answers[i] = scores.IndexOf(scores.Max());
            }
            bool[] correct = answers.Zip(testing_outputs.ToArray(), (x, y) => x == y).ToArray<bool>();
            accuracy = (double)correct.Sum() / correct.Length;
        }
    }

    public interface IPredictor
    {
        string model_type { get; set; }
        bool is_trained { get; set; }
        double[] predict(double[] input);
        void train(double[][] inputs, int[] outputs);
        void save(string filepath);
        void load(string filepath);
    }

    public class AccordSVMGaussianModel:IPredictor
    {
        public MulticlassSupportVectorLearning<Gaussian> teacher;
        public MulticlassSupportVectorMachine<Gaussian> learner;
        public string model_type { get; set; } = "svmgaussian";
        public double gamma;
        public double complexity;
        public bool autoestimate = true;
        public bool is_trained { get; set; } = false;

        public AccordSVMGaussianModel()
        {
            return;
        }

        public void train(double[][] inputs, int[] outputs)
        {

            if (autoestimate)
            {
                teacher = new MulticlassSupportVectorLearning<Gaussian>()
                {
                    Learner = (p) => new SequentialMinimalOptimization<Gaussian>()
                    {
                        UseKernelEstimation = true,
                        UseComplexityHeuristic = true,  // automatically sets complexity  
                    }
                };
            }
            else
            {
                teacher = new MulticlassSupportVectorLearning<Gaussian>()
                {
                    Learner = (p) => new SequentialMinimalOptimization<Gaussian>()
                    {
                        Complexity = complexity,
                        Kernel = Gaussian.FromGamma(gamma)
                    }
                };
            }

            learner = teacher.Learn(inputs, outputs);

            is_trained = true;
        }

        public double[] predict(double[] input)
        {
            double[] results = learner.Scores(input);
            return results;
        }


        public void save(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";
            Serializer.Save(learner, new_filepath);

            teacher = null;
            learner = null;
        }

        public void load(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";

            Serializer.Load(new_filepath, out learner);
        }
    }

    public class AccordSVMLinearModel : IPredictor
    {
        public MulticlassSupportVectorLearning<Linear> teacher;
        public MulticlassSupportVectorMachine<Linear> learner;
        public string model_type { get; set; } = "svmlinear";
        public double complexity;
        public bool autoestimate = true;
        public bool is_trained { get; set; } = false;

        public AccordSVMLinearModel()
        {
            return;
        }

        public void train(double[][] inputs, int[] outputs)
        {
            if (autoestimate)
            {
                teacher = new MulticlassSupportVectorLearning<Linear>()
                {
                    Learner = (p) => new LinearDualCoordinateDescent()
                    {
                        UseComplexityHeuristic = true  // automatically sets complexity  
                    }
                };
            }
            else
            {
                teacher = new MulticlassSupportVectorLearning<Linear>()
                {
                    Learner = (p) => new LinearDualCoordinateDescent()
                    {
                        Complexity = complexity
                    }
                };
            }

            learner = teacher.Learn(inputs, outputs);
            is_trained = true;
        }

        public double[] predict(double[] input)
        {
            double[] results = learner.Scores(input);
            return results;
        }

        public void save(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";
            Serializer.Save(learner, new_filepath);

            teacher = null;
            learner = null;
        }

        public void load(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";

            Serializer.Load(new_filepath, out learner);
        }
    }

    public class AccordLDAModel : IPredictor
    {
        public LinearDiscriminantAnalysis teacher;
        public LinearDiscriminantAnalysis.Pipeline learner;
        public string model_type { get; set; } = "lda";
        public double complexity;
        public bool autoestimate = true;
        public bool is_trained { get; set; } = false;

        public AccordLDAModel()
        {
            return;
        }

        public void train(double[][] inputs, int[] outputs)
        {
            teacher = new LinearDiscriminantAnalysis();
            learner = teacher.Learn(inputs, outputs);
            is_trained = true;
        }

        public double[] predict(double[] input)
        {
            double[] results = learner.Scores(input);
            return results;
        }

        public void save(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";
            Serializer.Save(learner, new_filepath);

            teacher = null;
            learner = null;
        }

        public void load(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";

            Serializer.Load(new_filepath, out learner);
        }
    }

    /// <summary>
    /// holds a pre-trained Open Neural Network eXchange model and provides some simple functionality to manipulate input and output data
    /// <para>
    /// ML.NET complicates this process by requiring all data entering ML.NET to have "IDataView" interface implemented within it's own user defined
    /// data class.  This makes it quite difficult to have a flexible data class which can handle arbitrary input size depending on the model fed to it.
    /// Right now, the primitive_type variable must match both the input and output layer primitive data type.
    /// </para>
    /// </summary>
    public class ONNXModel:IPredictor
    {
        public string model_type { get; set; } = "onnx";

        public string filepath;
        public int input_num;
        public int output_num;
        public string input_layer_name;
        public string output_layer_name;
        public string primitive_type;

        SchemaDefinition input_schemadef;
        SchemaDefinition output_schemadef;
        Microsoft.ML.Transforms.Onnx.OnnxTransformer transformer;
        MLContext mlContext = new MLContext();
        public bool is_trained { get; set; } = false;

        public ONNXModel()
        {
            return;
        }

        /// <summary>
        /// ML.Net requires a user defined class for input data
        /// </summary>
        /// <typeparam name="T"></typeparam>
        public class DynamicInputType<T>
        {
            public T[] input { get; set; }
        }

        /// <summary>
        /// ML.net requires a user defined class for output data"
        /// </summary>
        /// <typeparam name="T"></typeparam>
        public class DynamicOutputType<T>
        {
            public T[] output { get; set; }
        }

        /// <summary>
        /// Load the ONNX model
        /// </summary>
        /// <typeparam name="Tinput">primitive data type (e.g. float) of the expected input layer</typeparam>
        /// <typeparam name="Toutput">primitive data type (e.g. float) of the expected output layer</typeparam>
        /// <param name="filepath_">"absolute path to the .onnx model"</param>
        /// <param name="input_num_">"number of inputs in flattenned input layer"</param>
        /// <param name="output_num_">"number of outputs in output layer</param>
        /// <param name="input_layer_name_">"name of input layer expected by .onnx model"</param>
        /// <param name="output_layer_name_">"name of output layer expected by .onnx model</param>
        public void load_onnx<Tinput, Toutput>(string filepath_, int input_num_, int output_num_, string input_layer_name_, string output_layer_name_)
        {
            filepath = filepath_;
            input_num = input_num_;
            output_num = output_num_;
            input_layer_name = input_layer_name_;
            output_layer_name = output_layer_name_;

            // define input schema
            input_schemadef = SchemaDefinition.Create(typeof(DynamicInputType<Tinput>));
            var vectorItemType = ((VectorDataViewType)input_schemadef[0].ColumnType).ItemType;
            input_schemadef[0].ColumnType = new VectorDataViewType(vectorItemType, input_num); // adjust size of input schema to match .onnx model
            input_schemadef[0].ColumnName = input_layer_name; // adjust name of input schema to match .onnx model

            // define output schema
            output_schemadef = SchemaDefinition.Create(typeof(DynamicOutputType<Toutput>)); 
            vectorItemType = ((VectorDataViewType)output_schemadef[0].ColumnType).ItemType; 
            output_schemadef[0].ColumnType = new VectorDataViewType(vectorItemType, output_num);   // adjust size of output schema to match .onnx model
            output_schemadef[0].ColumnName = output_layer_name;   // adjust name of output schema to match .onnx model

            // create dummy data
            Tinput[] dummydata = new Tinput[input_num];
            IDataView dummydv = convert_input_to_dataview<Tinput>(dummydata);

            // create transformer
            var pipeline = mlContext.Transforms.ApplyOnnxModel(filepath);
            transformer = pipeline.Fit(dummydv);

            is_trained = true;
        }

        /// <summary>
        /// Transforms primitive type input array to dataview matching the required input schema definition
        /// </summary>
        /// <typeparam name="Tinput"></typeparam>
        /// <param name="data"></param>
        /// <returns></returns>
        private IDataView convert_input_to_dataview<Tinput>(Tinput[] data)
        {
            IEnumerable<DynamicInputType<Tinput>> enumerable_data = new DynamicInputType<Tinput>[]
            {
                new DynamicInputType<Tinput> {input = data},
            };
            IDataView dv = mlContext.Data.LoadFromEnumerable(enumerable_data, input_schemadef);
            return dv;
        }

        /// <summary>
        /// Transforms output dataview matching the required output schema definition to primitive type output array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dv"></param>
        /// <returns></returns>
        private T[] convert_dataview_to_output<T>(IDataView dv)
        {
            List<DynamicOutputType<T>> output_list = mlContext.Data.CreateEnumerable<DynamicOutputType<T>>(dv, reuseRowObject: false, schemaDefinition: output_schemadef).ToList();
            return output_list[0].output;
        }

        /// <summary>
        /// transforms double type input array to double type output array scores
        /// </summary>
        /// <param name="input"> flattened input array</param>
        /// <returns></returns>
        public double[] predict(double[] input)
        {

            if (primitive_type == "float")
            {
                // convert double input to float
                float[] floatArray = new float[input.Length];
                for (int i = 0; i < input.Length; i++)
                {
                    floatArray[i] = (float)input[i];
                }

                IDataView dv = convert_input_to_dataview<float>(floatArray);
                IDataView output_dv = transformer.Transform(dv);
                float[] outputs = convert_dataview_to_output<float>(output_dv);

                // convert float output to double
                double[] outputs_double = new double[outputs.Length];
                for (int i = 0; i < outputs.Length; i++)
                {
                    outputs_double[i] = (double)outputs[i];
                }
                return outputs_double;
            }
            else
            {
                IDataView dv = convert_input_to_dataview<double>(input);
                IDataView output_dv = transformer.Transform(dv);
                double[] outputs = convert_dataview_to_output<double>(output_dv);
                return outputs;
            }
        }

        // onnx models are preloaded, and don't need to be trained.  This function exists to satisfy interface requirements
        public void train(double[][] inputs, int[] outputs)
        {
            return;
        }

        /// <summary>
        /// required by predictor interface, copies .onnx file to new location
        /// </summary>
        /// <param name="new_filepath"></param>
        public void save(string new_filepath)
        {
            int ind = new_filepath.LastIndexOf('.');
            new_filepath = new_filepath.Substring(0, ind);
            new_filepath = new_filepath + ".onnx";

            if (File.Exists(filepath)){
                File.Copy(filepath, new_filepath, true);
            }
            transformer = null;
        }

        /// <summary>
        /// required by predictor interface, loads onnx file
        /// </summary>
        /// <param name="load_filepath"></param>
        public void load(string load_filepath)
        {
            int ind = load_filepath.LastIndexOf('.');
            string model_filepath = load_filepath.Substring(0, ind);
            model_filepath = model_filepath + ".onnx";
            filepath = model_filepath;

            if (primitive_type == "float")
            {
                load_onnx<float, float>(model_filepath, input_num, output_num, input_layer_name, output_layer_name);
            }
            else
            {
                load_onnx<double, double>(model_filepath, input_num, output_num, input_layer_name, output_layer_name);
            }
        }
}

    /// <summary>
    /// class used to hold random tests I ran during development, will delete after
    /// </summary>
    public static class Tests
    {
        public static void ONNXTest()
        {
            Console.WriteLine("Testing ONNX model");
        
            string onnxfilepath = @"C:\Users\Rico\Desktop\garbage data\generalized_classifier_v3.onnx";
            ONNXModel onnxmodel = new ONNXModel();
            onnxmodel.primitive_type = "float";
            onnxmodel.input_num = 256;
            onnxmodel.output_num = 5;
            onnxmodel.input_layer_name = "input";
            onnxmodel.output_layer_name = "softmax";
            onnxmodel.load(onnxfilepath);

            // trial1, position1, emg1, rest, 1
            //double[] input = new double[] {0.0334534935281074,0.0155259186303412,-0.00988409236361692,-0.0108421774637829,-0.0211299283338090,-0.0242005834224005,0.0298945136657577,0.0240544283134423,0.0113635604694395,-0.0532899578437731,0.0401483231298973,0.00417206509488551,-0.0581217757496608,0.0442983477496809,-0.00859965590939239,0.0121959065640328,0.0182280504588846,-0.0453626761156503,-0.0129707943654723,0.0246754528699610,0.0230689602799590,0.0218181280180796,-0.0315637994370069,-0.0334857389074689,0.00681954827632905,0.0363738663877772,-0.0121080635918214,-0.0352708415138627,0.0385137837747015,0.00645001084846835,-0.00453204568863537,-0.00891359184945941,0.00283703289316430,
            //                            0.136942004565118,0.00718972940694145,0.0395023568088694,-0.0365030259293970,-0.0343176077831806,-0.101420257183788,0.0679201293834360,-0.0188800114358607,0.0585633826704930,-0.0191680373904450,0.101426604688434,-0.0318976484265562,-0.216293134948889,0.115312576244564,0.0350851307267493,0.0453707514721065,0.0444341865324249,-0.0995513883271743,-0.0433481541616588,0.0185380935851493,-0.0115726333301834,0.104761263389277,-0.0560243929336888,-0.0226980488211303,0.00422564271750316,0.0580327896519274,0.0351144109015937,-0.185852418698800,0.128196797924903,-0.0448458311935805,0.0201381762034269,0.0326521660119680,0.00795174220457468,
            //                            0.0607861260493341,-0.0747070459396507,-0.0508400745004207,0.00268113823664208,0.162966016387372,-0.219406069071175,0.0919198136739963,0.0747699105159269,-0.00120713011398509,-0.131649639920524,0.128991946690426,-0.107061791541931,-0.109289560392209,0.236532654237869,-0.0391781688158264,-0.00773080439144117,0.0195663846714481,-0.0880547739404220,0.0636405174321386,0.0251903947489786,-0.226197763410636,0.126482500641501,0.114922678483873,-0.112182962724439,0.0587853842777102,0.114834265195710,-0.0317982736131296,-0.232633774452608,0.127635863275860,-0.0616812976228820,0.0873170099371822,0.00753585669358226,-0.00813604523017314,
            //                            -0.0245831640587528,0.0256240474426312,0.00510498325840561,-0.0377072191355123,0.0109034143224457,0.0173487127897220,-0.00356428355115142,-0.00546951342179072,-0.0110207522664264,0.0238091607897799,-0.00690393659012362,-0.00226370435070930,-0.00489695876167369,-0.00161575494472939,-0.00360026385756105,-0.00276156872598658,0.0142007624452021,0.0173977408541908,-0.00385668867380239,-0.0208309583751508,-0.0113561567276561,-0.00973175236679421,0.0301749396130723,-0.0103411102357885,0.0154040951726480,0.00665594418319313,-0.00537020529440146,-0.0196852579650128,-0.00315694814193763,-0.00217955290669444,0.0125856668675496,0.00602925537643900,0.00129907338067300,
            //                            -0.00489226308250357,-0.0235987564920497,0.0158687620092543,-0.0155484064479956,0.0134707985919329,0.00366855049983604,-0.000373422239774472,-0.00321501747835489,0.00676079794559293,-0.00480201382905979,-0.00324738114889573,0.00321846828600702,-0.00783407901408358,-0.00582775252496097,0.00111021073808346,0.00220974263518378,0.0343705396772304,0.00852086717143132,-0.0327228174380032,-0.0227764034583787,-0.00327486897337390,0.0216176839103971,0.00804419253287389,-0.00239716886859300,-0.00964235172610403,0.0230755136295660,-0.0108308783666030,-0.00291789188360758,-0.00167635304803612,-0.0123456202493081,0.00502185204319605,0.00273291528932613,0.00946923220728409,
            //                            -0.0118198095365368,-0.00371543644955323,0.0106034924900177,-0.0239184549647216,0.0217800872258353,-0.0125783415433848,0.0167583215837360,0.00424819175080609,-0.0123660307875563,0.00378114025057254,-0.0142228205134030,0.0260140144821076,-0.0132705066723757,-0.0266667032486651,0.0160211522808966,0.00904134103901767,0.0160132673412711,0.00461976707579188,-0.0177299881316452,-0.00683858554321316,-0.00974619761370399,-0.00483487618769967,0.0218489283919260,-0.00826955736419858,0.0129967770718533,0.00964338400280889,-0.00662216913993211,-0.0138328945648146,-0.0127746989553591,0.000961218310699067,0.00959669051861310,0.0135962990045395,-0.000451207363221301,
            //                            0.00182023409294589,0.198127551142678,-0.232202126230322,-0.130649137363206,0.115401506096051,0.0437442114775829,0.0411181545881474,-0.0490202382532104,-0.0116934271772505,-0.0660284360016179,0.0745403758139587,0.0116345455682097,0.0356911450296018,0.0318722150346920,-0.0544735861194019,-0.433762253765705,0.456069273332087,0.258342346250226,-0.264075561622713,-0.0892963481537417,-0.149768121180977,0.199191396052974,0.0721195066606695,-0.100086424680931,0.0106941201974265,0.109104279100965,-0.139594648357677,0.00241612311370273,0.0138049717426850,0.0421390610206025,-0.00799240194472854,0.0122447960760187,-0.0240246362541800,
            //                            0.000469164718398144,0.0714409435984345,-0.0423140285044708,0.0301360549145895,-0.0403588804685305,0.0442081053884080,-0.00960887539009872,0.0419007759860559,-0.0289610297110550,-0.0780127252978121,0.0156250122720084,0.0648450371072024,-0.0374142773669149,0.0529216155751062,-0.0288806156645492,-0.0400369485200208,0.0540327044034317,-0.0138858145113757,-0.0208983838984417,0.0159751931822853,-0.0390662228341190,0.0339410185233548,0.0192446047919084,-0.00732704437632403,-0.0118444222803623,0.0227748418163459,-0.0432041474352850,0.00437652385679135,0.00274759585217781,0.0290552665823262,-0.00574345675926528,-0.000707867160478249,-0.00603938754197877f};
            
            // trial1, position1, emg1, rest, 2
            double[] input = new double[] {0.0334534935281074,0.0155259186303412,-0.00988409236361692,-0.0108421774637829,-0.0211299283338090,-0.0242005834224005,0.0298945136657577,0.0240544283134423,0.0113635604694395,-0.0532899578437731,0.0401483231298973,0.00417206509488551,-0.0581217757496608,0.0442983477496809,-0.00859965590939239,0.0121959065640328,0.0182280504588846,-0.0453626761156503,-0.0129707943654723,0.0246754528699610,0.0230689602799590,0.0218181280180796,-0.0315637994370069,-0.0334857389074689,0.00681954827632905,0.0363738663877772,-0.0121080635918214,-0.0352708415138627,0.0385137837747015,0.00645001084846835,-0.00453204568863537,-0.00891359184945941,
                                            0.136942004565118,0.00718972940694145,0.0395023568088694,-0.0365030259293970,-0.0343176077831806,-0.101420257183788,0.0679201293834360,-0.0188800114358607,0.0585633826704930,-0.0191680373904450,0.101426604688434,-0.0318976484265562,-0.216293134948889,0.115312576244564,0.0350851307267493,0.0453707514721065,0.0444341865324249,-0.0995513883271743,-0.0433481541616588,0.0185380935851493,-0.0115726333301834,0.104761263389277,-0.0560243929336888,-0.0226980488211303,0.00422564271750316,0.0580327896519274,0.0351144109015937,-0.185852418698800,0.128196797924903,-0.0448458311935805,0.0201381762034269,0.0326521660119680,
                                            0.0607861260493341,-0.0747070459396507,-0.0508400745004207,0.00268113823664208,0.162966016387372,-0.219406069071175,0.0919198136739963,0.0747699105159269,-0.00120713011398509,-0.131649639920524,0.128991946690426,-0.107061791541931,-0.109289560392209,0.236532654237869,-0.0391781688158264,-0.00773080439144117,0.0195663846714481,-0.0880547739404220,0.0636405174321386,0.0251903947489786,-0.226197763410636,0.126482500641501,0.114922678483873,-0.112182962724439,0.0587853842777102,0.114834265195710,-0.0317982736131296,-0.232633774452608,0.127635863275860,-0.0616812976228820,0.0873170099371822,0.00753585669358226,
                                            -0.0245831640587528,0.0256240474426312,0.00510498325840561,-0.0377072191355123,0.0109034143224457,0.0173487127897220,-0.00356428355115142,-0.00546951342179072,-0.0110207522664264,0.0238091607897799,-0.00690393659012362,-0.00226370435070930,-0.00489695876167369,-0.00161575494472939,-0.00360026385756105,-0.00276156872598658,0.0142007624452021,0.0173977408541908,-0.00385668867380239,-0.0208309583751508,-0.0113561567276561,-0.00973175236679421,0.0301749396130723,-0.0103411102357885,0.0154040951726480,0.00665594418319313,-0.00537020529440146,-0.0196852579650128,-0.00315694814193763,-0.00217955290669444,0.0125856668675496,0.00602925537643900,
                                            -0.00489226308250357,-0.0235987564920497,0.0158687620092543,-0.0155484064479956,0.0134707985919329,0.00366855049983604,-0.000373422239774472,-0.00321501747835489,0.00676079794559293,-0.00480201382905979,-0.00324738114889573,0.00321846828600702,-0.00783407901408358,-0.00582775252496097,0.00111021073808346,0.00220974263518378,0.0343705396772304,0.00852086717143132,-0.0327228174380032,-0.0227764034583787,-0.00327486897337390,0.0216176839103971,0.00804419253287389,-0.00239716886859300,-0.00964235172610403,0.0230755136295660,-0.0108308783666030,-0.00291789188360758,-0.00167635304803612,-0.0123456202493081,0.00502185204319605,0.00273291528932613,
                                            -0.0118198095365368,-0.00371543644955323,0.0106034924900177,-0.0239184549647216,0.0217800872258353,-0.0125783415433848,0.0167583215837360,0.00424819175080609,-0.0123660307875563,0.00378114025057254,-0.0142228205134030,0.0260140144821076,-0.0132705066723757,-0.0266667032486651,0.0160211522808966,0.00904134103901767,0.0160132673412711,0.00461976707579188,-0.0177299881316452,-0.00683858554321316,-0.00974619761370399,-0.00483487618769967,0.0218489283919260,-0.00826955736419858,0.0129967770718533,0.00964338400280889,-0.00662216913993211,-0.0138328945648146,-0.0127746989553591,0.000961218310699067,0.00959669051861310,0.0135962990045395,
                                            0.00182023409294589,0.198127551142678,-0.232202126230322,-0.130649137363206,0.115401506096051,0.0437442114775829,0.0411181545881474,-0.0490202382532104,-0.0116934271772505,-0.0660284360016179,0.0745403758139587,0.0116345455682097,0.0356911450296018,0.0318722150346920,-0.0544735861194019,-0.433762253765705,0.456069273332087,0.258342346250226,-0.264075561622713,-0.0892963481537417,-0.149768121180977,0.199191396052974,0.0721195066606695,-0.100086424680931,0.0106941201974265,0.109104279100965,-0.139594648357677,0.00241612311370273,0.0138049717426850,0.0421390610206025,-0.00799240194472854,0.0122447960760187,
                                            0.000469164718398144,0.0714409435984345,-0.0423140285044708,0.0301360549145895,-0.0403588804685305,0.0442081053884080,-0.00960887539009872,0.0419007759860559,-0.0289610297110550,-0.0780127252978121,0.0156250122720084,0.0648450371072024,-0.0374142773669149,0.0529216155751062,-0.0288806156645492,-0.0400369485200208,0.0540327044034317,-0.0138858145113757,-0.0208983838984417,0.0159751931822853,-0.0390662228341190,0.0339410185233548,0.0192446047919084,-0.00732704437632403,-0.0118444222803623,0.0227748418163459,-0.0432041474352850,0.00437652385679135,0.00274759585217781,0.0290552665823262,-0.00574345675926528,-0.000707867160478249};

            double[] outputs = onnxmodel.predict(input);
            foreach (float output in outputs)
            {
                Console.WriteLine(output);
            }


        }

        public static void ONNXSettingsTest()
        {
            Console.WriteLine("Testing loading of ONNX model settings");

            string settings_filepath = @"C:\Users\Rico\Desktop\garbage data\generalized_classifier_v3.json";
            string onnxfilepath = @"C:\Users\Rico\Desktop\garbage data\generalized_classifier_v3.onnx";

            Model model = ObjLogger.loadObjJson<Model>(settings_filepath);
            model.model.load(onnxfilepath);

            // input data from rest class (fourth output)
            double[] input = new double[] {0.0334534935281074,0.0155259186303412,-0.00988409236361692,-0.0108421774637829,-0.0211299283338090,-0.0242005834224005,0.0298945136657577,0.0240544283134423,0.0113635604694395,-0.0532899578437731,0.0401483231298973,0.00417206509488551,-0.0581217757496608,0.0442983477496809,-0.00859965590939239,0.0121959065640328,0.0182280504588846,-0.0453626761156503,-0.0129707943654723,0.0246754528699610,0.0230689602799590,0.0218181280180796,-0.0315637994370069,-0.0334857389074689,0.00681954827632905,0.0363738663877772,-0.0121080635918214,-0.0352708415138627,0.0385137837747015,0.00645001084846835,-0.00453204568863537,-0.00891359184945941,
                                            0.136942004565118,0.00718972940694145,0.0395023568088694,-0.0365030259293970,-0.0343176077831806,-0.101420257183788,0.0679201293834360,-0.0188800114358607,0.0585633826704930,-0.0191680373904450,0.101426604688434,-0.0318976484265562,-0.216293134948889,0.115312576244564,0.0350851307267493,0.0453707514721065,0.0444341865324249,-0.0995513883271743,-0.0433481541616588,0.0185380935851493,-0.0115726333301834,0.104761263389277,-0.0560243929336888,-0.0226980488211303,0.00422564271750316,0.0580327896519274,0.0351144109015937,-0.185852418698800,0.128196797924903,-0.0448458311935805,0.0201381762034269,0.0326521660119680,
                                            0.0607861260493341,-0.0747070459396507,-0.0508400745004207,0.00268113823664208,0.162966016387372,-0.219406069071175,0.0919198136739963,0.0747699105159269,-0.00120713011398509,-0.131649639920524,0.128991946690426,-0.107061791541931,-0.109289560392209,0.236532654237869,-0.0391781688158264,-0.00773080439144117,0.0195663846714481,-0.0880547739404220,0.0636405174321386,0.0251903947489786,-0.226197763410636,0.126482500641501,0.114922678483873,-0.112182962724439,0.0587853842777102,0.114834265195710,-0.0317982736131296,-0.232633774452608,0.127635863275860,-0.0616812976228820,0.0873170099371822,0.00753585669358226,
                                            -0.0245831640587528,0.0256240474426312,0.00510498325840561,-0.0377072191355123,0.0109034143224457,0.0173487127897220,-0.00356428355115142,-0.00546951342179072,-0.0110207522664264,0.0238091607897799,-0.00690393659012362,-0.00226370435070930,-0.00489695876167369,-0.00161575494472939,-0.00360026385756105,-0.00276156872598658,0.0142007624452021,0.0173977408541908,-0.00385668867380239,-0.0208309583751508,-0.0113561567276561,-0.00973175236679421,0.0301749396130723,-0.0103411102357885,0.0154040951726480,0.00665594418319313,-0.00537020529440146,-0.0196852579650128,-0.00315694814193763,-0.00217955290669444,0.0125856668675496,0.00602925537643900,
                                            -0.00489226308250357,-0.0235987564920497,0.0158687620092543,-0.0155484064479956,0.0134707985919329,0.00366855049983604,-0.000373422239774472,-0.00321501747835489,0.00676079794559293,-0.00480201382905979,-0.00324738114889573,0.00321846828600702,-0.00783407901408358,-0.00582775252496097,0.00111021073808346,0.00220974263518378,0.0343705396772304,0.00852086717143132,-0.0327228174380032,-0.0227764034583787,-0.00327486897337390,0.0216176839103971,0.00804419253287389,-0.00239716886859300,-0.00964235172610403,0.0230755136295660,-0.0108308783666030,-0.00291789188360758,-0.00167635304803612,-0.0123456202493081,0.00502185204319605,0.00273291528932613,
                                            -0.0118198095365368,-0.00371543644955323,0.0106034924900177,-0.0239184549647216,0.0217800872258353,-0.0125783415433848,0.0167583215837360,0.00424819175080609,-0.0123660307875563,0.00378114025057254,-0.0142228205134030,0.0260140144821076,-0.0132705066723757,-0.0266667032486651,0.0160211522808966,0.00904134103901767,0.0160132673412711,0.00461976707579188,-0.0177299881316452,-0.00683858554321316,-0.00974619761370399,-0.00483487618769967,0.0218489283919260,-0.00826955736419858,0.0129967770718533,0.00964338400280889,-0.00662216913993211,-0.0138328945648146,-0.0127746989553591,0.000961218310699067,0.00959669051861310,0.0135962990045395,
                                            0.00182023409294589,0.198127551142678,-0.232202126230322,-0.130649137363206,0.115401506096051,0.0437442114775829,0.0411181545881474,-0.0490202382532104,-0.0116934271772505,-0.0660284360016179,0.0745403758139587,0.0116345455682097,0.0356911450296018,0.0318722150346920,-0.0544735861194019,-0.433762253765705,0.456069273332087,0.258342346250226,-0.264075561622713,-0.0892963481537417,-0.149768121180977,0.199191396052974,0.0721195066606695,-0.100086424680931,0.0106941201974265,0.109104279100965,-0.139594648357677,0.00241612311370273,0.0138049717426850,0.0421390610206025,-0.00799240194472854,0.0122447960760187,
                                            0.000469164718398144,0.0714409435984345,-0.0423140285044708,0.0301360549145895,-0.0403588804685305,0.0442081053884080,-0.00960887539009872,0.0419007759860559,-0.0289610297110550,-0.0780127252978121,0.0156250122720084,0.0648450371072024,-0.0374142773669149,0.0529216155751062,-0.0288806156645492,-0.0400369485200208,0.0540327044034317,-0.0138858145113757,-0.0208983838984417,0.0159751931822853,-0.0390662228341190,0.0339410185233548,0.0192446047919084,-0.00732704437632403,-0.0118444222803623,0.0227748418163459,-0.0432041474352850,0.00437652385679135,0.00274759585217781,0.0290552665823262,-0.00574345675926528,-0.000707867160478249};

            double[] outputs = model.model.predict(input);
            foreach (float output in outputs)
            {
                Console.WriteLine(output);
            }
        }

        public static void AccordSVMTest()
        {
            double[][] inputs =
            {
                //               input         output
                new double[] { 0, 1, 1, 0 }, //  0 
                new double[] { 0, 1, 0, 0 }, //  0
                new double[] { 0, 0, 1, 0 }, //  0
                new double[] { 0, 1, 1, 0 }, //  0
                new double[] { 0, 1, 0, 0 }, //  0
                new double[] { 1, 0, 0, 0 }, //  1
                new double[] { 1, 0, 0, 0 }, //  1
                new double[] { 1, 0, 0, 1 }, //  1
                new double[] { 0, 0, 0, 1 }, //  1
                new double[] { 0, 0, 0, 1 }, //  1
                new double[] { 1, 1, 1, 1 }, //  2
                new double[] { 1, 0, 1, 1 }, //  2
                new double[] { 1, 1, 0, 1 }, //  2
                new double[] { 0, 1, 1, 1 }, //  2
                new double[] { 1, 1, 1, 1 }, //  2
            };

            int[] outputs = // those are the class labels
            {
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
            };
            double complexity = 1;
            double gamma = 1;
            var teacher = new MulticlassSupportVectorLearning<Gaussian>()
            {
                Learner = (p) => new SequentialMinimalOptimization<Gaussian>()
                {
                    Complexity = Convert.ToDouble(complexity),
                    Kernel = Gaussian.FromGamma(Convert.ToDouble(gamma))
                }
            };

            MulticlassSupportVectorMachine<Gaussian> learner = teacher.Learn(inputs, outputs);
            double[] scores = learner.Score(inputs);
        }


        public static void generate_random_data(string filepath, int input_num, int num_lines)
        {

            List<List<double>> dummy_data = new List<List<double>>(); ;
            Random random = new Random();
            for (int i = 0; i < num_lines; i++)
            {
                List<double> temp = new List<double>();
                for (int j = 0; j < input_num; j++)
                {
                    temp.Add(random.NextDouble());
                }
                dummy_data.Add(temp);
            }

            StreamWriter file = new StreamWriter(filepath);
            file.Write("h,timestamp,");
            for (int i = 1; i < input_num; i++)
            {
                file.Write("ch" + i.ToString() + ",");
            }
            file.Write("output\n");

            for (int i = 0; i < num_lines; i++)
            {
                file.Write("d");
                for (int j = 0; j < input_num; j++)
                {
                    file.Write(',');
                    file.Write(dummy_data[i][j].ToString());
                }
                file.Write(",0"); file.Write('\n');
            }

            file.Flush();
            file.Close();
        }
    }
}

