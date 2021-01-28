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


// useful: https://dotnetfiddle.net/
// potential library: http://accord-framework.net/

namespace RealTimePatternRec.PatternRec
{

    public class Data
    {
        // Class to hold all information related to the data being used by the pattern rec class
        public List<double> timestamps = new List<double>();
        public List<List<double>> inputs = new List<List<double>>();    // raw input values
        public List<int> outputs = new List<int>(); // raw output values
        public List<int> input_types = new List<int>();     // types of input signals (0 = generic, 1 = emg)
        public List<bool> input_active_flags = new List<bool>();    //  indicating which inputs are active in model
        public List<string> output_labels = new List<string>(); // list of output label names
        public List<string> input_labels = new List<string>();  // list of input label names
        public int input_num;   // number of inputs in data
        public int output_num;  // number of outputs in data
        public int freq;    // frequency data was recorded at

        public void Clear()
        {
            timestamps.Clear();
            inputs.Clear();
            outputs.Clear();
            input_types.Clear();
            input_active_flags.Clear();
            output_labels.Clear();
            input_labels.Clear();
            input_num = 0;
            output_num = 0;
            freq = 0;
        }

        public bool LoadFileToListCols(string filepath)
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
            freq = (int)Math.Round(timestamps.Count * 1000 / (timestamps.Last() - timestamps.First()));

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
    /// 
    /// </summary>
    public class Model
    {
        public Data data;
        public dataLogger logger;    // is this a good idea???c 

        public int freq;    // frequency of data that model was trained on
        public int window_time;
        public int window_overlap; 
        public int window_n;
        public int window_overlap_n;
        public double train_test_split;
        public double accuracy;

        public bool modelFlag = false;
        public string model_type;
        public Dictionary<string, string> model_params;

        public List<List<double>> features = new List<List<double>>();  // inputs after mapping to features
        public List<int> feature_outputs = new List<int>(); // outputs trimmed to account for windowing

        public delegate List<double> pipeline_func(List<double> data);
        public List<pipeline_func> emg_pipeline = new List<pipeline_func>();
        public List<pipeline_func> generic_pipeline = new List<pipeline_func>();
        public dynamic accord_model = new System.Dynamic.ExpandoObject();

        //Thread t;
        //Stopwatch sw;
        //public int prevtime;
        //public int curtime;

        public Model()
        {
            data = new Data();
            accord_model.learner = null;
        }

        //public void start_realtime()
        //{
        //    sw = new Stopwatch();
        //    sw.Start();
        //    prevtime = sw.ElapsedMilliseconds;

        //}

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
            features.Clear();
            feature_outputs.Clear();
            freq = data.freq;

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

                if (features.Count == 0)
                {
                    for (int j = 0; j < temp_features.Count; j++)
                    {
                        features.Add(new List<double>());
                    }
                }
                for (int j = 0; j < temp_features.Count; j++)
                {
                    features[j].AddRange(temp_features[j]);
                }

                feature_outputs.AddRange(Enumerable.Repeat(output_value, temp_features[0].Count));
                start_ind = end_ind;
            }
        }

        public void train_model_Accord_list()
        {

            map_features_training();
            Data.shuffle_training_data(features, feature_outputs);

            // split to test/train set
            List<List<double>> features_rows = Data.transpose_list_list(features);

            int N_train = (int)(features_rows.Count * (1 - train_test_split));

            List<List<double>> training_features = features_rows.GetRange(0, N_train);
            List<int> training_outputs = feature_outputs.GetRange(0, N_train);

            List<List<double>> testing_features = features_rows.GetRange(N_train, features_rows.Count - N_train);
            List<int> testing_outputs = feature_outputs.GetRange(N_train, features_rows.Count - N_train);

            // train model
            accord_model.learner = accord_model.teacher.Learn(training_features.Select(a => a.ToArray()).ToArray(), training_outputs.ToArray());

            // Compute the machine answers for the data.inputs
            int[] answers = accord_model.learner.Decide(testing_features.Select(a => a.ToArray()).ToArray());
            bool[] correct = answers.Zip(testing_outputs.ToArray(), (x, y) => x == y).ToArray<bool>();

            accuracy = (double)correct.Sum() / correct.Length;  
        }

        public double predict(List<List<double>> rawdata)
        {
            double result = 0;
            List<List<double>> features_temp = map_features(rawdata, data.input_types, data.input_active_flags);
            double[] temp = Data.transpose_list_list(features_temp)[0].ToArray();
            result = (double)accord_model.learner.Decide(temp);

            return result;
        }
    }

    /// <summary>
    /// static class of EMG and other feature mapping methods for time series data structured as Lists
    /// </summary>
    public static class Features
    {

        static public List<double> RAW(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i+=(window_n - window_overlap_n))
            {
                
                filtered_values.Add(raw_values[i]);
            }

            return filtered_values;
        }

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
        /// Returns the windowed mean absolute value of a signal
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

        static public List<double> SSC(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> difference_values = raw_values.Zip(raw_values.Skip(1), (x, y) => y - x).ToList();
            difference_values.Add(difference_values.Last());    // append final value since one is lost during difference calculation, keeps feature sizes consistent
            List<double> filtered_values = ZC(difference_values, window_n, window_overlap_n);
            return filtered_values;
        }
    }

    public class PR_Logger:dataLogger
    {
        //public dataLogger logger;
        public int current_output;

        public int contraction_time;
        public int relax_time;
        public long start_time;
        public int collection_cycles;
        public int current_cycle;
        public int train_output_num;
        public bool trainFlag = false;  // flag to indicate training has begun
        public bool contractFlag = false;
        public List<string> output_labels;

        public PR_Logger()
        {
            //logger = new dataLogger();
            return;
        }

        public override void write_header(List<string> data)
        {
            data.Insert(0, "h");
            data.Add("output");
            write_csv(data);
        }

        public override void write_data_with_timestamp(List<string> data)
        {
            data.Insert(0, "d");
            data.Insert(1, curtime.ToString("F3"));
            data.Add(current_output.ToString());
            write_csv(data);
        }

        public void set_outputs(List<string> outputs_)
        {
            output_labels = outputs_;
            train_output_num = output_labels.Count;
        }

        public void PR_tick()
        {
            // updates the flags corresponding to current action being performed during training

            if (trainFlag)
            {
                long elapsed_time = (long)(curtime - start_time);
                long segment_time = relax_time + contraction_time;

                int segment_number = (int)Math.Floor((decimal)elapsed_time / segment_time);

                current_output = segment_number%train_output_num;

                long local_time = elapsed_time - segment_time * segment_number;
                contractFlag = local_time >= relax_time;

                if (contractFlag)
                {
                    recordflag = true;
                }
                else
                {
                    recordflag = false;
                }

                current_cycle = (int)Math.Floor((decimal)elapsed_time / (segment_time * train_output_num));

                if (current_cycle >= collection_cycles)
                {
                    end_data_collection();
                }
            }
        }

        public void start_data_collection()
        {
            current_output = 0;
            current_cycle = 0;
            trainFlag = true;
            tick();
            start_time = (long)curtime;
        }

        public void end_data_collection()
        {
            close_file();
            recordflag = false;
            trainFlag = false;
        }
    }

    public static class Tests
    {

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

