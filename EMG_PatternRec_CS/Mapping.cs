using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NWaves.Filters;


namespace EMG_PatternRec_CS.Mapping
{

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
        static public List<double> dummy_feature_example(
            List<double> raw_values,
            int window_size_n,
            int window_overlap_n,
            double offset)
        {
            List<double> feature_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                feature_values.Add(raw_values[i] + offset);
            }

            return feature_values;
        }

        /// <summary>
        /// Returns the windowed raw value (downsampled to match other windowed features)
        /// </summary>
        /// <param name="raw_values"></param>
        /// <param name="window_size_n"></param>
        /// <param name="window_overlap_n"></param>
        /// <returns></returns>
        static public List<double> RAW(
            List<double> raw_values,
            int window_size_n,
            int window_overlap_n)
        {
            List<double> feature_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                feature_values.Add(raw_values[i]);
            }

            return feature_values;
        }

        /// <summary>
        /// Returns the windowed mean value
        /// </summary>
        /// <param name="raw_values">List of unwindowed values</param>
        /// <param name="window_size_n">Number of data points in each window</param>
        /// <param name="window_overlap_n">Number of data points to overlap in each subsequent window</param>
        /// <returns></returns>
        static public List<double> MV(
            List<double> raw_values,
            int window_size_n,
            int window_overlap_n)
        {
            List<double> feature_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);
                feature_values.Add(sub_window.Sum() / window_size_n);
            }

            return feature_values;
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
            List<double> feature_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double sum = 0;
                for (int j = 0; j < window_size_n; j++)
                {
                    sum += Math.Abs(sub_window[j]);
                }
                feature_values.Add(sum);
            }

            return feature_values;
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
            List<double> feature_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double sum = 0;
                for (int j = 0; j < sub_window.Count; j++)
                {
                    sum += Math.Abs(sub_window[j]);
                }
                feature_values.Add(sum / window_size_n);
            }

            return feature_values;
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
            List<double> feature_values = new List<double>();

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
                feature_values.Add(sum / window_size_n);
            }

            return feature_values;
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
            List<double> feature_values = new List<double>();

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
                feature_values.Add(sum / window_size_n);
            }

            return feature_values;
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
            List<double> feature_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double sum = 0;
                for (int j = 0; j < window_size_n; j++)
                {
                    sum += Math.Pow(sub_window[j], 2);
                }
                feature_values.Add(sum);
            }

            return feature_values;
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
            List<double> feature_values = new List<double>();
            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);

                double variance = 0;
                for (int j = 0; j < window_size_n; j++)
                {
                    variance += Math.Pow(sub_window[j], 2); ;
                }
                feature_values.Add(variance / (window_size_n - 1));
            }

            return feature_values;
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
            List<double> feature_values = new List<double>();

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
                feature_values.Add(crossing_count);
            }

            return feature_values;
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
            List<double> feature_values = ZC(difference_values, window_size_n, window_overlap_n);
            return feature_values;
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
            List<double> feature_values = new List<double>();
            for (int i = 0; i <= (raw_values.Count - window_size_n); i += (window_size_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_size_n);
                double length = 0;
                for (int j = 1; j < sub_window.Count; j++)
                {
                    length += Math.Abs(sub_window[j] - sub_window[j - 1]);
                }
                feature_values.Add(length);
            }
            return feature_values;
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
            List<double> feature_values = new List<double>();
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
                feature_values.Add(wilson_amplitude);
            }

            return feature_values;
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
            // safety check for dividing by zero
            if (max_values[channel_num] - min_values[channel_num] == 0)
            {
                return raw_values;
            }

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
            // safety check for dividing by zero
            if (max_values[channel_num] - min_values[channel_num] == 0)
            {
                return raw_values;
            }

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

        // centers all inputs to have a mean value of 0 and a standard deviation of 1
        static public List<double> Standardize(List<double> raw_values, List<double> mean_values, List<double> stdev_values, int channel_num)
        {
            // safety check for dividing by zero
            if (stdev_values[channel_num] == 0)
            {
                return raw_values;
            }

            List<double> filtered_values = new List<double>();

            for (int i = 0; i < raw_values.Count; i++)
            {
                double new_value = (raw_values[i] - mean_values[channel_num]) / stdev_values[channel_num];
                filtered_values.Add(new_value);
            }

            return filtered_values;
        }
    }

    /// <summary>
    /// static class of various filter techniques using the NWaves library for preprocessing signals.
    /// </summary>
    /// <para>
    /// note that signals cannot share a single filter, as digital filters depend on previous inputs and outputs on a per-signal basis
    /// </para>
    public static class Filters
    {
        /// <summary>
        /// create a notch filter
        /// </summary>
        /// <param name="f_notch"> notch frequency </param>
        /// <param name="fs"> sampel frequency </param>
        /// <returns></returns>
        public static NWaves.Filters.BiQuad.NotchFilter create_notch_filter(double f_notch, double fs)
        {
            var notchFilter = new NWaves.Filters.BiQuad.NotchFilter(f_notch / fs);
            return notchFilter;
        }

        /// <summary>
        /// create a low pass butterworth filter
        /// </summary>
        /// <param name="fc"> cutoff frequency </param>
        /// <param name="fs"> sample frequency </param>
        /// <param name="order"> filter order </param>
        /// <returns></returns>
        public static NWaves.Filters.Butterworth.LowPassFilter create_lowpass_butterworth_filter(double fc, double fs, int order)
        {
            var butterLPfilter = new NWaves.Filters.Butterworth.LowPassFilter(fc / fs, order);
            return butterLPfilter;
        }

        /// <summary>
        /// create a a high pass butterworth filter
        /// </summary>
        /// <param name="fc"> cutoff frequency </param>
        /// <param name="fs"> sample frequency </param>
        /// <param name="order"> filter order</param>
        /// <returns></returns>
        public static NWaves.Filters.Butterworth.HighPassFilter create_highpass_butterworth_filter(double fc, double fs, int order)
        {
            var butterHPfilter = new NWaves.Filters.Butterworth.HighPassFilter(fc / fs, order);
            return butterHPfilter;
        }

        /// <summary>
        /// create a moving average filter
        /// </summary>
        /// <param name="order">filter order number</param>
        /// <returns></returns>
        public static NWaves.Filters.MovingAverageFilter create_movingaverage_filter(int order)
        {
            var maFilter = new MovingAverageFilter(order);
            return maFilter;
        }

        /// <summary>
        /// applies a filter to all samples in the signal
        /// </summary>
        /// <param name="filter"></param>
        /// <param name="raw_values"></param>
        /// <returns></returns>
        public static List<double> apply_filter(NWaves.Filters.Base.IOnlineFilter filter, List<double> raw_values)
        {
            List<double> filtered_values = new List<double>();
            for (int i = 0; i < raw_values.Count; i++)
            {
                filtered_values.Add(filter.Process((float)raw_values[i]));
            }

            return filtered_values;
        }

        /// <summary>
        /// selects the filter from a list of filters and applies it to all samples in the signal
        /// </summary>
        /// <para>
        /// This overloaded function allows a single delegate function to apply a filter type to all signals in the Mapping class.
        /// The delegate created will have the list of filters specified, and which filter is used is specified by the channel_num parameter.
        /// </para>
        /// <param name="filters"></param>
        /// <param name="raw_values"></param>
        /// <param name="channel_num"></param>
        /// <returns></returns>
        public static List<double> apply_filter(List<NWaves.Filters.Base.IOnlineFilter> filters, List<double> raw_values, int channel_num)
        {
            return apply_filter(filters[channel_num], raw_values);
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

        // variables required for Scalers
        public List<double> max_values;
        public List<double> min_values;
        public List<double> mean_values;
        public List<double> stdev_values;

        /// <summary>
        /// delegate for Filter functions
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="filter"></param>
        /// <param name="raw_values"></param>
        /// <returns></returns>
        public delegate List<double> filter_pipeline_func(List<double> raw_values, int channel_num_);

        /// <summary>list of filter functions pipeline will iterate through for generic signals</summary>
        public List<filter_pipeline_func> generic_filter_pipeline = new List<filter_pipeline_func>();
        /// <summary>list of filter functions pipeline will iterate through for emg signals</summary>
        public List<filter_pipeline_func> emg_filter_pipeline = new List<filter_pipeline_func>();
        /// <summary>list of filter function keywords for generic signals</summary>
        public List<string> generic_filter_pipeline_titles = new List<string>();
        /// <summary>list of filter function keywords for emg signals</summary>
        public List<string> emg_filter_pipeline_titles = new List<string>();

        // variables required for saving filters
        public Dictionary<string, double> emg_filter_params = new Dictionary<string, double>();
        public Dictionary<string, double> generic_filter_params = new Dictionary<string, double>();

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
        /// Apply all Filter functions in filter pipelines to both generic and emg signals
        /// </summary>
        /// <param name="raw_inputs">raw signals of size timestepnum X inputnum</param>
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
                        filtered_values[i] = (f(filtered_values[i], i));
                    }
                }
                else if (input_types[i] == 1 && input_active_flags[i] == true)
                {
                    foreach (filter_pipeline_func f in emg_filter_pipeline)
                    {
                        filtered_values[i] = (f(filtered_values[i], i));
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
    }

    /// <summary>
    /// class to hold all post processing techniques
    /// </summary>
    public class PostProcessor
    {
        int num_outputs;

        Queue<int> majorityVotingQueue;
        public int majorityVotingBufLength;
        public bool majorityVoteFlag = false;

        List<double> velocityRampScore;
        public double velocityRampIncrement;
        public bool velocityRampFlag = false;

        public PostProcessor()
        {
            return;
        }

        /// <summary>
        /// enables majority voting technique
        /// </summary>
        /// <param name="majorityVotingBufLength_">number of previous outputs keep in bufferk</param>
        public void init_majorityVoting(int majorityVotingBufLength_)
        {
            majorityVotingBufLength = majorityVotingBufLength_;

            majorityVotingQueue = new Queue<int>();
            for (int i = 0; i < majorityVotingBufLength; i++)
            {
                majorityVotingQueue.Enqueue(0);
            }

            majorityVoteFlag = true;
        }

        /// <summary>
        /// enables velocity ramping technique
        /// </summary>
        /// <param name="num_outputs_">number of available outputs</param>
        /// <param name="velocityRampIncrement_">percentage to increment/decrement each signal with each new output</param>
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

        /// <summary>
        /// applies majority voting technique and returns udpated scores
        /// </summary>
        /// <param name="scores">probability predictions for each output</param>
        /// <returns></returns>
        public double[] majorityVoting(double[] scores)
        {
            int current_output = Array.IndexOf(scores, scores.Max());

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

        /// <summary>
        /// helper function to get the mode from a list of values
        /// </summary>
        /// <typeparam name="T">primitive type</typeparam>
        /// <param name="list">values</param>
        /// <returns></returns>
        private T getMode<T>(IEnumerable<T> list)
        {
            T mode = list.GroupBy(i => i).OrderByDescending(grp => grp.Count()).Select(grp => grp.Key).First();
            return mode;
        }

        /// <summary>
        /// applies velocity ramping technique and returns updated scores
        /// </summary>
        /// <param name="scores">probability predictions for each output</param>
        /// <returns></returns>
        public double[] velocityRamp(double[] scores)
        {
            int current_output = Array.IndexOf(scores, scores.Max());

            for (int i = 0; i < num_outputs; i++)
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

        /// <summary>
        /// applies all enabled post-processing techniques to scores
        /// </summary>
        /// <param name="scores">probability predictions for each output</param>
        /// <returns></returns>
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
