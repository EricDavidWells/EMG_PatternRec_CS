using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using RealTimePatternRec.DataLogging;
using System.Threading;

using Accord.Controls;
using Accord.IO;
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;

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

        public bool LoadFileToListCols(string filepath)
        {
            string[] lines_arr = System.IO.File.ReadAllLines(filepath);
            List<string> lines = lines_arr.ToList();

            // if number of inputs in data file does not match set number of inputs
            if (input_num != lines[0].Split(',').Length - 3)
            {
                return false;
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
                if (line[0] == 'd')
                {
                    char[] chars_to_trim = { 'd', ',' };
                    List<double> vals = Array.ConvertAll(line.TrimStart(chars_to_trim).Split(','), Double.Parse).ToList();
                    outputs.Add((int)vals.Last());
                    vals.RemoveAt(vals.Count - 1);

                    timestamps.Add(vals.First());
                    vals.RemoveAt(0);

                    for (int i = 0; i < vals.Count; i++)
                    {
                        inputs[i].Add(vals[i]);
                    }
                }
            }

            return true;
        }

        public bool LoadFileToListCols2(string filepath)
        {
            // read data file into lists data.inputs and data.outputs, return false if unsuccessful

            string[] lines_arr = System.IO.File.ReadAllLines(filepath);
            List<string> lines = lines_arr.ToList();

            Clear();

            input_num = lines[0].Split(',').Length - 3;

            // preallocate data.inputs
            for (int i = 0; i < input_num; i++)
            {
                inputs.Add(new List<double>());
            }

            foreach (string line in lines)
            {
                // parse header line
                if (line[0] == 'h')
                {
                    char[] chars_to_trim = { 'h', ',' };
                    input_labels = line.TrimStart(chars_to_trim).Split(',').ToList<string>();

                    if (input_labels.Last() != "output")
                    {
                        return false;  // return if there is no output column
                    }
                    else
                    {
                        input_labels.RemoveAt(input_labels.Count - 1);
                        input_labels.RemoveAt(0);

                        for (int i = 0; i < input_labels.Count; i++)
                        {
                            input_active_flags.Add(false);

                            if (input_labels[i].ToLower().Contains("ch"))
                            {
                                input_types.Add(1);
                            }
                            else
                            {
                                input_types.Add(0);
                            }
                        }
                    }
                }
                // parse data lines
                else if (line[0] == 'd')
                {
                    char[] chars_to_trim = { 'd', ',' };
                    List<double> vals = Array.ConvertAll(line.TrimStart(chars_to_trim).Split(','), Double.Parse).ToList();
                    outputs.Add((int)vals.Last());
                    vals.RemoveAt(vals.Count - 1);

                    timestamps.Add(vals.First());
                    vals.RemoveAt(0);

                    for (int i = 0; i < vals.Count; i++)
                    {
                        inputs[i].Add(vals[i]);
                    }
                }
                else
                {
                    return false; // return if there is no leading character (i.e. not a proper data file)
                }
            }

            // find frequency used for logging
            output_num = outputs.Max() + 1;
            //freq = (int)Math.Round(timestamps.Count * 1000 / (timestamps.Last() - timestamps.First()));

            return true;
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
        /// <param name="window_n"></param>
        /// <param name="window_overlap_n"></param>
        /// <returns></returns>
        static public List<double> RAW(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                filtered_values.Add(raw_values[i]);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed mean value
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns></returns>
        static public List<double> MV(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);
                filtered_values.Add(sub_window.Sum() / window_n);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed integrated emg value
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> IEMG(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);

                double sum = 0;
                for (int j = 0; j < window_n; j++)
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
        /// <param name="window_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> MAV(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);

                double sum = 0;
                for (int j = 0; j < sub_window.Count; j++)
                {
                    sum += Math.Abs(sub_window[j]);
                }
                filtered_values.Add(sum / window_n);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed modified mean absolute value 1
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> MMAV1(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);

                double sum = 0;
                for (int j = 0; j < window_n; j++)
                {
                    double w;
                    if (j >= 0.25 * window_n && j <= 0.75 * window_n)
                    {
                        w = 1;
                    }
                    else
                    {
                        w = 0.5;
                    }
                    sum += (w * Math.Abs(sub_window[j]));
                }
                filtered_values.Add(sum / window_n);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed modified mean absolute value 2
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> MMAV2(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);

                double sum = 0;
                for (int j = 0; j < sub_window.Count; j++)
                {
                    double w;
                    if (j < 0.25 * window_n)
                    {
                        w = 4 * j / window_n;
                    }
                    else if (j >= 0.25 * window_n && j <= 0.75 * window_n)
                    {
                        w = 1;
                    }
                    else
                    {
                        w = 4*(j-window_n)/window_n;
                    }
                    sum += (w * Math.Abs(sub_window[j]));
                }
                filtered_values.Add(sum / window_n);
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the winowed simple square integral value
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns>"windowed mean absolute value"</returns>
        static public List<double> SSI(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);

                double sum = 0;
                for (int j = 0; j < window_n; j++)
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
        /// <param name="window_n"></param>
        /// <param name="window_overlap_n"></param>
        /// <returns></returns>
        static public List<double> VAR(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();
            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);

                double variance = 0;
                for (int j = 0; j < window_n; j++)
                {
                    variance += Math.Pow(sub_window[j], 2); ;
                }
                filtered_values.Add(variance / (window_n - 1));
            }

            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed number of zero crossings
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns></returns>
        static public List<double> ZC(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);
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
        /// <param name="window_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns></returns>
        static public List<double> SSC(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> difference_values = raw_values.Zip(raw_values.Skip(1), (x, y) => y - x).ToList();
            difference_values.Add(difference_values.Last());    // append final value since one is lost during difference calculation, keeps feature sizes consistent
            List<double> filtered_values = ZC(difference_values, window_n, window_overlap_n);
            return filtered_values;
        }

        /// <summary>
        /// Returns the windowed waveform length
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns></returns>
        static public List<double> WL(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();
            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);
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
        /// <param name="window_n"></param>
        /// <param name="window_overlap_n"></param>
        /// <param name="threshold"></param>
        /// <returns></returns>
        static public List<double> WAMP(List<double> raw_values, int window_n, int window_overlap_n, double threshold)
        {
            List<double> filtered_values = new List<double>();
            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);

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
        public dataLogger logger;    // is this a good idea???c 

        public int window_time;
        public int window_overlap; 
        public int window_n;
        public int window_overlap_n;
        public double train_test_split;
        public double accuracy;

        public bool modelFlag = false;
        public bool realtimeFlag = false;
        public string model_type;
        public Dictionary<string, string> model_params;

        public delegate List<double> pipeline_func(List<double> data_);

        public List<pipeline_func> generic_pipeline = new List<pipeline_func>();
        public List<pipeline_func> emg_pipeline = new List<pipeline_func>();
        public List<string> generic_pipeline_titles = new List<string>();
        public List<string> emg_pipeline_titles = new List<string>();

        public dynamic accord_model = new System.Dynamic.ExpandoObject();
        public string model_save_filepath;
        public dynamic learner;

        public Model()
        {
            data = new Data();
            accord_model.learner = null;
        }

        public List<List<double>> map_features(List<List<double>> temp_inputs, List<int> input_types, List<bool> input_active_flags)
        {
            // map inputs "temp_inputs" to features using the set "emg_pipeline" and "generic_pipeline"
            // function lists.  Also requires that "data.input_types" are set and "data.input_active_flags" are set

            List<List<double>> temp_features = new List<List<double>>();

            for (int i = 0; i < input_types.Count; i++)
            {
                if (input_types[i] == 0 && input_active_flags[i] == true)
                {
                    foreach (Model.pipeline_func f in generic_pipeline)
                    {
                        temp_features.Add(f(temp_inputs[i]));
                    }
                }
                else if (input_types[i] == 1 && input_active_flags[i] == true)
                {
                    foreach (Model.pipeline_func f in emg_pipeline)
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

        public void train_model_Accord_list()
        {

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
            accord_model.learner = accord_model.teacher.Learn(training_features.Select(a => a.ToArray()).ToArray(), training_outputs.ToArray());

            // Compute the machine answers for the data.inputs
            int[] answers = accord_model.learner.Decide(testing_features.Select(a => a.ToArray()).ToArray());
            bool[] correct = answers.Zip(testing_outputs.ToArray(), (x, y) => x == y).ToArray<bool>();

            accuracy = (double)correct.Sum() / correct.Length;
            modelFlag = true; 
        }

        public double predict(List<List<double>> rawdata)
        {
            double result = 0;
            List<List<double>> features_temp = map_features(rawdata, data.input_types, data.input_active_flags);
            double[] temp = Data.transpose_list_list(features_temp)[0].ToArray();
            result = (double)accord_model.learner.Decide(temp);

            return result;
        }

        public void save_model(string filepath)
        {
            model_save_filepath = filepath;
            if (modelFlag)
            {
                Serializer.Save(accord_model.learner, model_save_filepath);
            }
        }

        public void load_model()
        {
            Serializer.Load(model_save_filepath, out learner);
            accord_model.learner = learner;
        }
    }

    /// <summary>
    /// holds a pre-trained Open Neural Network eXchange model and provides some simple functionality to manipulate input and output data
    /// </summary>
    public class ONNXModel
    {
        public string filepath;
        public int num_inputs;
        public int num_outputs;
        public Type inputType;
        public Type outputType;

        public SchemaDefinition schemadef;
        Microsoft.ML.Transforms.Onnx.OnnxTransformer transformer;
        MLContext mlContext = new MLContext();

        public ONNXModel()
        {
            //mlContext = new MLContext();
            return;
        }

        /// <summary>
        /// ML.Net requires a user defined class for input data.  The name of the variable being used must match that expected by the onxx model.
        /// This class allows for various input feature sizes, as they are set in the load_model method
        /// </summary>
        /// <typeparam name="T"></typeparam>
        public class DynamicDataType<T>
        {
            public T[] input { get; set; }
        }

        /// <summary>
        /// ML.net requires a user defined class for output data.  The name of the variable being used must match that expected by the onxx model.
        /// This class will work with a neural network output layer defined "softmax"
        /// </summary>
        /// <typeparam name="T"></typeparam>
        public class Prediction_softmax<T>
        {
            public T[] softmax { get; set; }
        }

        /// <summary>
        /// Load the ONNX model, resize user-defined input data class
        /// </summary>
        /// <typeparam name="Tinput"></typeparam>
        /// <typeparam name="Toutput"></typeparam>
        /// <param name="filepath_"></param>
        /// <param name="num_inputs_"></param>
        /// <param name="num_outputs_"></param>
        public void load_model<Tinput>(string filepath_, int num_inputs_, int num_outputs_)
        {
            filepath = filepath_;
            num_inputs = num_inputs_;
            num_outputs = num_outputs_;

            // define schema and adjust size for number of inputs
            schemadef = SchemaDefinition.Create(typeof(DynamicDataType<Tinput>));
            var vectorItemType = ((VectorDataViewType)schemadef[0].ColumnType).ItemType;
            schemadef[0].ColumnType = new VectorDataViewType(vectorItemType, num_inputs);

            // create dummy data
            Tinput[] dummydata = new Tinput[num_inputs];
            IDataView dummydv = convert_input_to_dataview<Tinput>(dummydata);

            // create transformer
            var pipeline = mlContext.Transforms.ApplyOnnxModel(filepath);
            transformer = pipeline.Fit(dummydv);
        }

        /// <summary>
        /// transforms array input to a dataview for ML.net
        /// </summary>
        /// <typeparam name="Tinput"></typeparam>
        /// <param name="data"></param>
        /// <returns></returns>
        public IDataView convert_input_to_dataview<Tinput>(Tinput[] data)
        {
            IEnumerable<DynamicDataType<Tinput>> enumerable_data = new DynamicDataType<Tinput>[]
            {
                new DynamicDataType<Tinput> {input = data},
            };
            IDataView dv = mlContext.Data.LoadFromEnumerable(enumerable_data, schemadef);
            return dv;
        }

        /// <summary>
        /// converts a dataview output from ML.net to the user-defined class specified to match the ONNX model
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dv"></param>
        /// <returns></returns>
        public T convert_dataview_to_output_userdefined_class<T>(IDataView dv) where T : class, new()
        {
            // must put into a list temporarily in order to convert from a dataview, only return the one output
            List<T> output_list = mlContext.Data.CreateEnumerable<T>(dv, reuseRowObject: false).ToList();
            return output_list[0];
        }

        /// <summary>
        /// transforms input into a user-defined class output specified to match the ONNX model
        /// </summary>
        /// <typeparam name="Tinput"> primitive data type</typeparam>
        /// <typeparam name="Toutput"> user-defined data type</typeparam>
        /// <param name="input"> array of specified primitive data type</param>
        /// <returns></returns>
        public Toutput predict<Tinput, Toutput>(Tinput[] input) where Toutput : class, new()
        {
            IDataView dv = convert_input_to_dataview<Tinput>(input);
            IDataView output_dv = transformer.Transform(dv);
            Toutput outputs = convert_dataview_to_output_userdefined_class<Toutput>(output_dv);
            return outputs;
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

            string onnxfilepath = @"C:\Users\Rico\OneDrive\Documents\School\BLINC\Project\Development\BrachIOplexus Classifier\Heather ONNX\generalized_classifier.onnx";
            ONNXModel onnxmodel = new ONNXModel();
            onnxmodel.load_model<float>(onnxfilepath, 8 * 33, 5);

            // input data from rest class (fourth output)
            float[] input = new float[] {0.0334534935281074f,0.0155259186303412f,-0.00988409236361692f,-0.0108421774637829f,-0.0211299283338090f,-0.0242005834224005f,0.0298945136657577f,0.0240544283134423f,0.0113635604694395f,-0.0532899578437731f,0.0401483231298973f,0.00417206509488551f,-0.0581217757496608f,0.0442983477496809f,-0.00859965590939239f,0.0121959065640328f,0.0182280504588846f,-0.0453626761156503f,-0.0129707943654723f,0.0246754528699610f,0.0230689602799590f,0.0218181280180796f,-0.0315637994370069f,-0.0334857389074689f,0.00681954827632905f,0.0363738663877772f,-0.0121080635918214f,-0.0352708415138627f,0.0385137837747015f,0.00645001084846835f,-0.00453204568863537f,-0.00891359184945941f,0.00283703289316430f,
                                        0.136942004565118f,0.00718972940694145f,0.0395023568088694f,-0.0365030259293970f,-0.0343176077831806f,-0.101420257183788f,0.0679201293834360f,-0.0188800114358607f,0.0585633826704930f,-0.0191680373904450f,0.101426604688434f,-0.0318976484265562f,-0.216293134948889f,0.115312576244564f,0.0350851307267493f,0.0453707514721065f,0.0444341865324249f,-0.0995513883271743f,-0.0433481541616588f,0.0185380935851493f,-0.0115726333301834f,0.104761263389277f,-0.0560243929336888f,-0.0226980488211303f,0.00422564271750316f,0.0580327896519274f,0.0351144109015937f,-0.185852418698800f,0.128196797924903f,-0.0448458311935805f,0.0201381762034269f,0.0326521660119680f,0.00795174220457468f,
                                        0.0607861260493341f,-0.0747070459396507f,-0.0508400745004207f,0.00268113823664208f,0.162966016387372f,-0.219406069071175f,0.0919198136739963f,0.0747699105159269f,-0.00120713011398509f,-0.131649639920524f,0.128991946690426f,-0.107061791541931f,-0.109289560392209f,0.236532654237869f,-0.0391781688158264f,-0.00773080439144117f,0.0195663846714481f,-0.0880547739404220f,0.0636405174321386f,0.0251903947489786f,-0.226197763410636f,0.126482500641501f,0.114922678483873f,-0.112182962724439f,0.0587853842777102f,0.114834265195710f,-0.0317982736131296f,-0.232633774452608f,0.127635863275860f,-0.0616812976228820f,0.0873170099371822f,0.00753585669358226f,-0.00813604523017314f,
                                        -0.0245831640587528f,0.0256240474426312f,0.00510498325840561f,-0.0377072191355123f,0.0109034143224457f,0.0173487127897220f,-0.00356428355115142f,-0.00546951342179072f,-0.0110207522664264f,0.0238091607897799f,-0.00690393659012362f,-0.00226370435070930f,-0.00489695876167369f,-0.00161575494472939f,-0.00360026385756105f,-0.00276156872598658f,0.0142007624452021f,0.0173977408541908f,-0.00385668867380239f,-0.0208309583751508f,-0.0113561567276561f,-0.00973175236679421f,0.0301749396130723f,-0.0103411102357885f,0.0154040951726480f,0.00665594418319313f,-0.00537020529440146f,-0.0196852579650128f,-0.00315694814193763f,-0.00217955290669444f,0.0125856668675496f,0.00602925537643900f,0.00129907338067300f,
                                        -0.00489226308250357f,-0.0235987564920497f,0.0158687620092543f,-0.0155484064479956f,0.0134707985919329f,0.00366855049983604f,-0.000373422239774472f,-0.00321501747835489f,0.00676079794559293f,-0.00480201382905979f,-0.00324738114889573f,0.00321846828600702f,-0.00783407901408358f,-0.00582775252496097f,0.00111021073808346f,0.00220974263518378f,0.0343705396772304f,0.00852086717143132f,-0.0327228174380032f,-0.0227764034583787f,-0.00327486897337390f,0.0216176839103971f,0.00804419253287389f,-0.00239716886859300f,-0.00964235172610403f,0.0230755136295660f,-0.0108308783666030f,-0.00291789188360758f,-0.00167635304803612f,-0.0123456202493081f,0.00502185204319605f,0.00273291528932613f,0.00946923220728409f,
                                        -0.0118198095365368f,-0.00371543644955323f,0.0106034924900177f,-0.0239184549647216f,0.0217800872258353f,-0.0125783415433848f,0.0167583215837360f,0.00424819175080609f,-0.0123660307875563f,0.00378114025057254f,-0.0142228205134030f,0.0260140144821076f,-0.0132705066723757f,-0.0266667032486651f,0.0160211522808966f,0.00904134103901767f,0.0160132673412711f,0.00461976707579188f,-0.0177299881316452f,-0.00683858554321316f,-0.00974619761370399f,-0.00483487618769967f,0.0218489283919260f,-0.00826955736419858f,0.0129967770718533f,0.00964338400280889f,-0.00662216913993211f,-0.0138328945648146f,-0.0127746989553591f,0.000961218310699067f,0.00959669051861310f,0.0135962990045395f,-0.000451207363221301f,
                                        0.00182023409294589f,0.198127551142678f,-0.232202126230322f,-0.130649137363206f,0.115401506096051f,0.0437442114775829f,0.0411181545881474f,-0.0490202382532104f,-0.0116934271772505f,-0.0660284360016179f,0.0745403758139587f,0.0116345455682097f,0.0356911450296018f,0.0318722150346920f,-0.0544735861194019f,-0.433762253765705f,0.456069273332087f,0.258342346250226f,-0.264075561622713f,-0.0892963481537417f,-0.149768121180977f,0.199191396052974f,0.0721195066606695f,-0.100086424680931f,0.0106941201974265f,0.109104279100965f,-0.139594648357677f,0.00241612311370273f,0.0138049717426850f,0.0421390610206025f,-0.00799240194472854f,0.0122447960760187f,-0.0240246362541800f,
                                        0.000469164718398144f,0.0714409435984345f,-0.0423140285044708f,0.0301360549145895f,-0.0403588804685305f,0.0442081053884080f,-0.00960887539009872f,0.0419007759860559f,-0.0289610297110550f,-0.0780127252978121f,0.0156250122720084f,0.0648450371072024f,-0.0374142773669149f,0.0529216155751062f,-0.0288806156645492f,-0.0400369485200208f,0.0540327044034317f,-0.0138858145113757f,-0.0208983838984417f,0.0159751931822853f,-0.0390662228341190f,0.0339410185233548f,0.0192446047919084f,-0.00732704437632403f,-0.0118444222803623f,0.0227748418163459f,-0.0432041474352850f,0.00437652385679135f,0.00274759585217781f,0.0290552665823262f,-0.00574345675926528f,-0.000707867160478249f,-0.00603938754197877f};

            ONNXModel.Prediction_softmax<float> output_list = onnxmodel.predict<float, ONNXModel.Prediction_softmax<float>>(input);

            int n_outputs = output_list.softmax.Length;
            float[] outputs = new float[n_outputs];
            for (int i = 0; i < n_outputs; i++)
            {
                outputs[i] = output_list.softmax[i];
            }

            foreach (float output in outputs)
            {
                Console.WriteLine(output);
            }
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

