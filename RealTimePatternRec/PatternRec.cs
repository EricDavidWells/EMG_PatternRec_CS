using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using System.Dynamic;
using System.Diagnostics;
using RealTimePatternRec.DataLogging;
using System.Threading;

using Accord.IO;
using Accord.Math;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Accord.Statistics.Analysis;

using NWaves.Filters;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace RealTimePatternRec.PatternRec
{
    /// <summary>
    /// class to hold all information from a data recording session
    /// </summary>
    public class Data
    {
        /// <summary>timestamp data</summary>
        public List<double> timestamps = new List<double>();
        /// <summary>raw input values</summary>
        public List<List<double>> inputs = new List<List<double>>();
        /// <summary>raw output values</summary>
        public List<int> outputs = new List<int>();
        /// <summary>types of input signals (0 = generic, 1 = emg)</summary>
        public List<int> input_types = new List<int>();
        /// <summary>inputs after mapping to features</summary>
        public List<List<double>> features = new List<List<double>>();
        /// <summary>outputs trimmed to account for windowing</summary>
        public List<int> feature_outputs = new List<int>();
        /// <summary>indicating which inputs are active in model</summary>
        public List<bool> input_active_flags = new List<bool>();
        /// <summary>output label names</summary>
        public List<string> output_labels = new List<string>();
        /// <summary>list of input label names</summary>
        public List<string> input_labels = new List<string>();
        /// <summary>number of inputs in data</summary>
        public int input_num;
        /// <summary>number of outputs in data</summary>
        public int output_num;
        /// <summary>frequency data was recorded at</summary>
        public int freq;
        /// <summary>number of times outputs were iterated over during data collection</summary>
        public int collection_cycles;
        /// <summary>number of seconds per contraction during data collection</summary>
        public int contraction_time;
        /// <summary>number of seconds for relaxation during data collection</summary>
        public int relaxation_time;

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
            for (int i = 0; i < input_labels.Count; i++)
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

        /// <summary>
        /// sets all input active flags
        /// </summary>
        /// <param name="flag"></param>
        public void SetAllInputActiveFlags(bool flag)
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

    /// <summary>
    /// static class of window based feature mapping methods for time series data structured as Lists
    /// </summary>
    /// <para>
    /// list of sources:
    /// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7250028/
    /// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3821366/
    /// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5017469/
    /// https://doi.org/10.1016/j.eswa.2012.01.102
    /// </para>
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
                        w = 4 * (j - window_size_n) / window_size_n;
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
                for (int j = 1; j < sub_window.Count; j++)
                {
                    length += Math.Abs(sub_window[j] - sub_window[j - 1]);
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
    /// static class of various scaling techniques for preprocessing data.  These functions are designed for use with the "scaler_pipeline_func" delegate.
    /// since the information required(min_values, max_values, and mean_values) will change after each function is performed, only one of these functions should be called
    /// </summary>
    public static class Scalers
    {
        /// <summary>
        /// performs both MinMaxScaling and ZeroCentering simultaneously to avoid the need to recalculate each channel's min/max/mean values after each scaling function independently.
        /// (i.e. the min, max, and mean values change after zero shifting, or after min max scaling)
        /// </summary>
        /// <param name="raw_values"></param>
        /// <param name="min_values"></param>
        /// <param name="max_values"></param>
        /// <param name="mean_values"></param>
        /// <param name="channel_num"></param>
        /// <returns></returns>
        static public List<double> MinMaxZeroCenter(List<double> raw_values, List<double> min_values, List<double> max_values, List<double> mean_values, int channel_num)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i < raw_values.Count; i++)
            {
                double new_value = ((raw_values[i] - min_values[channel_num]) / (max_values[channel_num] - min_values[channel_num])) - 0.5;
                filtered_values.Add(new_value);
            }

            return filtered_values;
        }

        /// <summary>
        /// scales all inputs to a value between 0 (minimum) and 1 (maximum)
        /// </summary>
        /// <param name="raw_values"></param>
        /// <param name="min_values">list of minimum values for each input signal</param>
        /// <param name="max_values">list of maximum values for each input signal</param>
        /// <param name="channel_num">input signal number, used to select which min_value and which max_value to use for scaling</param>
        /// <returns></returns>
        static public List<double> MinMaxScaling(List<double> raw_values, List<double> min_values, List<double> max_values, int channel_num)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i < raw_values.Count; i++)
            {
                double new_value = (raw_values[i] - min_values[channel_num]) / (max_values[channel_num] - min_values[channel_num]);
                filtered_values.Add(new_value);
            }

            return filtered_values;
        }

        /// <summary>
        /// centers all inputs to have a mean value of 0
        /// </summary>
        /// <param name="raw_values"></param>
        /// <param name="mean_values">list of mean values for each input signal</param>
        /// <param name="channel_num">input signal number, used to select which min_value and which max_value to use for scaling</param>
        /// <returns></returns>
        static public List<double> ZeroCentering(List<double> raw_values, List<double> mean_values, int channel_num)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i < raw_values.Count; i++)
            {
                double new_value = raw_values[i] - mean_values[channel_num];
                filtered_values.Add(new_value);
            }

            return filtered_values;
        }
    }

    /// <summary>
    /// static class of various filter techniques using the NWaves library for preprocessing signals.
    /// </summary>
    public static class Filters
    {

        public static NWaves.Filters.BiQuad.NotchFilter create_notch_filter(double f_notch, double fs)
        {
            var notchFilter = new NWaves.Filters.BiQuad.NotchFilter(f_notch / fs);
            return notchFilter;
        }

        public static NWaves.Filters.Butterworth.LowPassFilter create_lowpass_butterworth_filter(double fc, double fs, int order)
        {
            var butterLPfilter = new NWaves.Filters.Butterworth.LowPassFilter(fc / fs, order);
            return butterLPfilter;
        }

        public static NWaves.Filters.Butterworth.HighPassFilter create_highpass_butterworth_filter(double fc, double fs, int order)
        {
            var butterHPfilter = new NWaves.Filters.Butterworth.HighPassFilter(fc / fs, order);
            return butterHPfilter;
        }

        public static NWaves.Filters.MovingAverageFilter create_movingaverage_filter(int order)
        {
            var maFilter = new MovingAverageFilter(order);
            return maFilter;
        }

        public static List<double> apply_filter<T>(T filter, List<double> raw_values) where T: NWaves.Filters.Base.IOnlineFilter
        {
            List<double> filtered_values = new List<double>();
            for (int i=0; i<raw_values.Count; i++)
            {
                filtered_values.Add(filter.Process((float)raw_values[i]));
            }

            return filtered_values;
        }

        public static List<double> apply_filter(NWaves.Filters.Base.IOnlineFilter filter, List<double> raw_values) 
        {
            List<double> filtered_values = new List<double>();
            for (int i = 0; i < raw_values.Count; i++)
            {
                filtered_values.Add(filter.Process((float)raw_values[i]));
            }

            return filtered_values;
        }

    }
    
    /// <summary>
    /// utilizes Scalers, Features, and Filters functions to create a pipeline mapping raw input signals to features
    /// </summary>
    public class Mapper
    {
        /// <summary>number of timesteps per window</summary>
        public int window_size_n;
        /// <summary>number of timesteps to overlap</summary>
        public int window_overlap_n;    
        /// <summary>number of windows to use per model input</summary>
        public int window_n;

        /// <summary>
        /// delegate for Feature functions
        /// </summary>
        /// <param name="data_">input signal</param>
        /// <returns>features from input signal</returns>
        public delegate List<double> feature_pipeline_func(List<double> data_);

        /// <summary>list of Feature functions pipeline will iterate through for generic signals</summary>
        public List<feature_pipeline_func> generic_feature_pipeline = new List<feature_pipeline_func>();
        /// <summary>list of Feature functions pipeline will iterate through for emg signals</summary>
        public List<feature_pipeline_func> emg_feature_pipeline = new List<feature_pipeline_func>();
        /// <summary>list of Feature function keywords for generic signals</summary>
        public List<string> generic_feature_pipeline_titles = new List<string>();
        /// <summary>list of Feature function keywords for emg signals</summary>
        public List<string> emg_feature_pipeline_titles = new List<string>();

        /// <summary>
        /// delegate for Scaler functions
        /// </summary>
        /// <param name="data_">input signal</param>
        /// <param name="channel_num_">input channel num, used for selecting which min/max/mean value to use fo scaling</param>
        /// <returns></returns>
        public delegate List<double> scaler_pipeline_func(List<double> data_, int channel_num_);

        /// <summary>list of scaler functions pipeline will iterate through for generic signals</summary>
        public List<scaler_pipeline_func> generic_scaler_pipeline = new List<scaler_pipeline_func>();
        /// <summary>list of scaler functions pipeline will iterate through for emg signals</summary>
        public List<scaler_pipeline_func> emg_scaler_pipeline = new List<scaler_pipeline_func>();
        /// <summary>list of Scaler function keywords for generic signals</summary>
        public List<string> generic_scaler_pipeline_titles = new List<string>();
        /// <summary>list of Scaler function keywords for emg signals</summary>
        public List<string> emg_scaler_pipeline_titles = new List<string>();

        /// <summary>
        /// delegate for Filter functions
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="filter"></param>
        /// <param name="raw_values"></param>
        /// <returns></returns>
        public delegate List<double> filter_pipeline_func(List<double> raw_values);

        /// <summary>list of filter functions pipeline will iterate through for generic signals</summary>
        public List<filter_pipeline_func> generic_filter_pipeline = new List<filter_pipeline_func>();
        /// <summary>list of filter functions pipeline will iterate through for emg signals</summary>
        public List<filter_pipeline_func> emg_filter_pipeline = new List<filter_pipeline_func>();
        /// <summary>list of filter function keywords for generic signals</summary>
        public List<string> generic_filter_pipeline_titles = new List<string>();
        /// <summary>list of filter function keywords for emg signals</summary>
        public List<string> emg_filter_pipeline_titles = new List<string>();


        // variables required for Scalers
        public List<double> max_values;
        public List<double> min_values;
        public List<double> mean_values;

        public Mapper()
        {
            return;
        }

        /// <summary>
        /// Apply all Scaler functions in scaler pipelines to both generic and emg signals
        /// </summary>
        /// <param name="raw_inputs">raw signals</param>
        /// <param name="input_types">signal types (0 for generic, 1 for emg)</param>
        /// <param name="input_active_flags">indicates whether each signal is being used (true/false)</param>
        /// <returns></returns>
        public List<List<double>> scale_signals(List<List<double>> raw_inputs, List<int> input_types, List<bool> input_active_flags)
        {
            List<List<double>> scaled_values = new List<List<double>>(raw_inputs);

            for (int i = 0; i < input_types.Count; i++)
            {
                if (input_types[i] == 0 && input_active_flags[i] == true)
                {
                    foreach (scaler_pipeline_func f in generic_scaler_pipeline)
                    {
                        scaled_values[i] = (f(scaled_values[i], i));
                    }
                }
                else if (input_types[i] == 1 && input_active_flags[i] == true)
                {
                    foreach (scaler_pipeline_func f in emg_scaler_pipeline)
                    {
                        scaled_values[i] = (f(scaled_values[i], i));
                    }
                }
            }
            return scaled_values;
        }

        /// <summary>
        /// Apply all Scaler functions in scaler pipelines to both generic and emg signals
        /// </summary>
        /// <param name="raw_inputs">raw signals</param>
        /// <param name="input_types">signal types (0 for generic, 1 for emg)</param>
        /// <param name="input_active_flags">indicates whether each signal is being used (true/false)</param>
        /// <returns></returns>
        public List<List<double>> filter_signals(List<List<double>> raw_inputs, List<int> input_types, List<bool> input_active_flags)
        {
            List<List<double>> filtered_values = new List<List<double>>(raw_inputs);

            for (int i = 0; i < input_types.Count; i++)
            {
                if (input_types[i] == 0 && input_active_flags[i] == true)
                {
                    foreach (filter_pipeline_func f in generic_filter_pipeline)
                    {
                        filtered_values[i] = (f(filtered_values[i]));
                    }
                }
                else if (input_types[i] == 1 && input_active_flags[i] == true)
                {
                    foreach (filter_pipeline_func f in emg_filter_pipeline)
                    {
                        filtered_values[i] = (f(filtered_values[i]));
                    }
                }
            }
            return filtered_values;
        }


        /// <summary>
        /// Apply all Feature functions in feature pipelines to both generic and emg signals
        /// </summary>
        /// <param name="raw_inputs">raw signals</param>
        /// <param name="input_types">signal types (0 for generic, 1 for emg)</param>
        /// <param name="input_active_flags">indicates whether each signal is being used (true/false)</param>
        /// <returns></returns>
        public List<List<double>> map_features(List<List<double>> raw_inputs, List<int> input_types, List<bool> input_active_flags)
        {

            List<List<double>> temp_features = new List<List<double>>();

            for (int i = 0; i < input_types.Count; i++)
            {
                if (input_types[i] == 0 && input_active_flags[i] == true)
                {
                    foreach (feature_pipeline_func f in generic_feature_pipeline)
                    {
                        temp_features.Add(f(raw_inputs[i]));
                    }
                }
                else if (input_types[i] == 1 && input_active_flags[i] == true)
                {
                    foreach (feature_pipeline_func f in emg_feature_pipeline)
                    {
                        temp_features.Add(f(raw_inputs[i]));
                    }
                }
            }

            return temp_features;
        }

        public List<List<double>> map_all(List<List<double>> raw_inputs, List<int> input_types, List<bool> input_active_flags)
        {
            List<List<double>> scaled_inputs = scale_signals(raw_inputs, input_types, input_active_flags);
            List<List<double>> filtered_inputs = filter_signals(scaled_inputs, input_types, input_active_flags);
            List<List<double>> features = map_features(filtered_inputs, input_types, input_active_flags);

            return features;
        }
    }

    /// <summary>
    /// Predictor interface to standardize implemented predictors
    /// </summary>
    public interface IPredictor
    {
        string model_type { get; set; }
        bool is_trained { get; set; }
        double[] predict(double[] input);
        void train(double[][] inputs, int[] outputs);
        void save(string filepath);
        void load(string filepath);
    }

    /// <summary>
    /// SVM predictor with gaussian kernel using model from ACCORD.NET library
    /// </summary>
    public class AccordSVMGaussianModel : IPredictor
    {
        /// <summary>ACCORD.NET model</summary>
        public MulticlassSupportVectorMachine<Gaussian> learner;
        /// <summary>type of model</summary>
        public string model_type { get; set; } = "svmgaussian";
        /// <summary>SVM hyperparameter</summary>
        public double gamma;
        /// <summary>SVM hyperparameter</summary>
        public double complexity;
        /// <summary>if true ACCORD.NET will autoestimate gamma and complexity</summary>
        public bool autoestimate = true;
        /// <summary>indicates model has been trained</summary>
        public bool is_trained { get; set; } = false;

        public AccordSVMGaussianModel()
        {
            return;
        }

        /// <summary>
        /// trains model
        /// </summary>
        /// <param name="inputs">input features</param>
        /// <param name="outputs">ground truth classes</param>
        public void train(double[][] inputs, int[] outputs)
        {
            MulticlassSupportVectorLearning<Gaussian> teacher;

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

        /// <summary>
        /// predicts output
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] predict(double[] input)
        {
            double[] results = learner.Scores(input);
            return results;
        }

        /// <summary>
        /// serializes model to a .bin file and deletes model
        /// </summary>
        /// <param name="filepath"></param>
        public void save(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";
            Serializer.Save(learner, new_filepath);

            learner = null;
        }

        /// <summary>
        /// deserializes .bin file into model
        /// </summary>
        /// <param name="filepath"></param>
        public void load(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";

            Serializer.Load(new_filepath, out learner);
        }
    }

    /// <summary>
    /// SVM predictor with gaussian kernel using model from ACCORD.NET library
    /// </summary>
    public class AccordSVMLinearModel : IPredictor
    {
        /// <summary>ACCORD.NET model</summary>
        public MulticlassSupportVectorMachine<Linear> learner;
        /// <summary>type of model</summary>
        public string model_type { get; set; } = "svmlinear";
        /// <summary>SVM hyperparameter</summary>
        public double complexity;
        /// <summary>if true ACCORD.NET will autoestimate gamma and complexity</summary>
        public bool autoestimate = true;
        /// <summary>indicates model has been trained</summary>
        public bool is_trained { get; set; } = false;

        public AccordSVMLinearModel()
        {
            return;
        }

        /// <summary>
        /// trains model
        /// </summary>
        /// <param name="inputs">input features</param>
        /// <param name="outputs">ground truth classes</param>
        public void train(double[][] inputs, int[] outputs)
        {
            MulticlassSupportVectorLearning<Linear> teacher;

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

        /// <summary>
        /// predicts output
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] predict(double[] input)
        {
            double[] results = learner.Scores(input);
            return results;
        }

        /// <summary>
        /// serializes model to a .bin file and deletes model
        /// </summary>
        /// <param name="filepath"></param>
        public void save(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";
            Serializer.Save(learner, new_filepath);

            learner = null;
        }

        /// <summary>
        /// deserializes .bin file into model
        /// </summary>
        /// <param name="filepath"></param>
        public void load(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";

            Serializer.Load(new_filepath, out learner);
        }
    }

    /// <summary>
    /// LDA predictor using model from ACCORD.NET library
    /// </summary>
    public class AccordLDAModel : IPredictor
    {
        /// <summary>ACCORD.NET model</summary>
        public LinearDiscriminantAnalysis.Pipeline learner;
        /// <summary>type of model</summary>
        public string model_type { get; set; } = "lda";
        /// <summary>indicates model has been trained</summary>
        public bool is_trained { get; set; } = false;

        public AccordLDAModel()
        {
            return;
        }

        /// <summary>
        /// trains model
        /// </summary>
        /// <param name="inputs">input features</param>
        /// <param name="outputs">ground truth classes</param>
        public void train(double[][] inputs, int[] outputs)
        {
            LinearDiscriminantAnalysis teacher = new LinearDiscriminantAnalysis();
            learner = teacher.Learn(inputs, outputs);
            is_trained = true;
        }

        /// <summary>
        /// predicts output
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] predict(double[] input)
        {
            double[] results = learner.Scores(input);
            return results;
        }

        /// <summary>
        /// serializes model to a .bin file and deletes model
        /// </summary>
        /// <param name="filepath"></param>
        public void save(string filepath)
        {
            int ind = filepath.LastIndexOf('.');
            string new_filepath = filepath.Substring(0, ind) + ".bin";
            Serializer.Save(learner, new_filepath);

            learner = null;
        }

        /// <summary>
        /// deserializes .bin file into model
        /// </summary>
        /// <param name="filepath"></param>
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
    public class ONNXModel : IPredictor
    {
        /// <summary>type of model</summary>
        public string model_type { get; set; } = "onnx";
        /// <summary>path to onnx file</summary>
        public string filepath;
        /// <summary>number of inputs into onnx model</summary>
        public int input_num;
        /// <summary>number of outputs from onnx model</summary>
        public int output_num;
        /// <summary>name of input layer name, see netron.app to find out</summary>
        public string input_layer_name;
        /// <summary>name of output layer name, see netron.app to find out</summary>
        public string output_layer_name;
        /// <summary>name of input/output primitive type, see netron.app to find out</summary>
        public string primitive_type;
        /// <summary>indicates model has been trained</summary>
        public bool is_trained { get; set; } = false;

        private SchemaDefinition input_schemadef;
        private SchemaDefinition output_schemadef;
        private Microsoft.ML.Transforms.Onnx.OnnxTransformer transformer;
        private MLContext mlContext = new MLContext();

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

        /// <summary>
        /// onnx models are preloaded, and don't need to be trained.  This function exists to satisfy interface requirements
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
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

            if (File.Exists(filepath))
            {
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
    /// class to hold all pattern recognition model information and capabilities
    /// </summary>
    public class Model
    {
        public Data data;
        public DataLogger logger;
        public Mapper mapper;
        public IPredictor model;
        public PostProcessor postprocessor;

        /// <summary>percentage of total data to use for testing data</summary>
        public double train_test_split;
        /// <summary>accuracy of trained model</summary>
        public double accuracy;
        /// <summary>flag to indicate whether real-time is enabled</summary>
        public bool realtimeFlag = false;

        public Model()
        {
            data = new Data();
            mapper = new Mapper();
            logger = new DataLogger();
            postprocessor = new PostProcessor();
        }

        /// <summary>
        /// runs all of the training data through the Mapper pipeline and stores features and outputs in the Data object
        /// </summary>
        public void map_trainingdata_to_features()
        {
            // clear features
            data.features.Clear();
            data.feature_outputs.Clear();

            // get indices of where the output changes to split up data, since windowing technique will overlap into various classes if data is not split up
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

            // put all inputs through mapper in sections corresponding to output changes
            for (int i = 1; i < output_change_indices.Count; i++)
            {
                int output_value = data.outputs[output_change_indices[i - 1]];
                int end_ind = output_change_indices[i];

                List<List<double>> temp_input = Data.transpose_list_list(Data.transpose_list_list(data.inputs).GetRange(start_ind, end_ind - start_ind));
                //List<List<double>> temp_input_scaled = mapper.scale_signals(temp_input, data.input_types, data.input_active_flags);
                //List<List<double>> temp_features = mapper.map_features(temp_input_scaled, data.input_types, data.input_active_flags);
                List<List<double>> temp_features = mapper.map_all(temp_input, data.input_types, data.input_active_flags);

                // initialize features to size fitting the number of computed features
                if (data.features.Count == 0)
                {
                    for (int j = 0; j < temp_features.Count; j++)
                    {
                        data.features.Add(new List<double>());
                    }
                }

                // add computed features to features variable
                for (int j = 0; j < temp_features.Count; j++)
                {
                    data.features[j].AddRange(temp_features[j]);
                }

                // add outputs corresponding to features
                data.feature_outputs.AddRange(Enumerable.Repeat(output_value, temp_features[0].Count));
                start_ind = end_ind;
            }
        }

        /// <summary>
        /// Splits data into training and testing set, trains model, and calculates accuracy
        /// </summary>
        public void train_model()
        {
            // map inputs to features and shuffle
            map_trainingdata_to_features();
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

        /// <summary>
        /// predicts scores for a single input
        /// </summary>
        /// <param name="inputs_"></param>
        /// <returns></returns>
        public double[] get_scores(List<List<double>> input)
        {
            List<List<double>> features_ = mapper.map_all(input, data.input_types, data.input_active_flags);
            double[] features_flattenned = Data.transpose_list_list(features_).SelectMany(i => i).ToArray();
            double[] scores = model.predict(features_flattenned);
            double[] post_processed_scores = postprocessor.process(scores);
            return post_processed_scores;
        }
    }

    public class PostProcessor
    {
        int num_outputs;

        Queue<int> majorityVotingQueue;
        int majorityVotingLength;
        public bool majorityVoteFlag = false;

        List<double> velocityRampScore;
        double velocityRampIncrement;
        public bool velocityRampFlag = false;

        public PostProcessor()
        {
            return;
        }

        public void init_majorityVoting(int majorityVotingLength_)
        {
            majorityVotingLength = majorityVotingLength_;

            majorityVotingQueue = new Queue<int>();
            for (int i = 0; i < majorityVotingLength; i++)
            {
                majorityVotingQueue.Enqueue(0);
            }

            majorityVoteFlag = true;
        }

        public void init_velocityRamping(int num_outputs_, double velocityRampIncrement_)
        {
            num_outputs = num_outputs_;
            velocityRampIncrement = velocityRampIncrement_;

            velocityRampScore = new List<double>();
            for (int i = 0; i < num_outputs; i++)
            {
                velocityRampScore.Add(0);
            }

            velocityRampFlag = true;
        }

        public double[] majorityVoting(double[] scores)
        {
            int current_output = scores.IndexOf(scores.Max());
            
            majorityVotingQueue.Dequeue();
            majorityVotingQueue.Enqueue(current_output);
            int mode = getMode(majorityVotingQueue);

            // one-hot encode scores
            for (int i = 0; i < scores.Length; i++)
            {
                if (i == mode)
                {
                    scores[i] = 1;
                }
                else
                {
                    scores[i] = 0;
                }
            }

            return scores;
        }

        private T getMode<T>(IEnumerable<T> list)
        {
            T mode = list.GroupBy(i => i).OrderByDescending(grp => grp.Count()).Select(grp => grp.Key).First();
            return mode;
        }

        public double[] velocityRamp(double[] scores)
        {
            int current_output = scores.IndexOf(scores.Max());

            for (int i=0; i< num_outputs; i++)
            {
                if (i == current_output)
                {
                    velocityRampScore[i] = Math.Min(velocityRampScore[i] + velocityRampIncrement, 1);
                }
                else
                {
                    velocityRampScore[i] = Math.Max(velocityRampScore[i] - velocityRampIncrement, 0);
                }
            }

            return velocityRampScore.ToArray();
        }

        public double[] process(double[] scores)
        {

            if (majorityVoteFlag)
            {
                scores = majorityVoting(scores);

            }
            if (velocityRampFlag)
            {
                scores = velocityRamp(scores);
            }

            return scores;
        }
    }

}

