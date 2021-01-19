using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using RealTimePatternRec.DataLogging;


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
        public List<List<double>> inputs = new List<List<double>>();
        public List<List<double>> features = new List<List<double>>();
        public List<int> outputs = new List<int>();
        public List<int> feature_outputs = new List<int>();
        public List<int> input_types = new List<int>();
        public List<bool> input_active_flags = new List<bool>();
        public int input_num;
        public int output_num;

        public void Clear()
        {
            timestamps.Clear();
            inputs.Clear();
            features.Clear();
            outputs.Clear();
            feature_outputs.Clear();
            input_active_flags.Clear();
            input_types.Clear();
        }
    }

    public class PR_Logger
    {
        int windowlength = 100;
        public dataLogger logger = new dataLogger();
        public int current_output = 0;

        public int contraction_time = 1000;
        public int relax_time = 3000;
        public long start_time = 0;
        public int collection_cycles = 1;
        public int current_cycle = 0;

        public bool trainFlag = false;  // flag to indicate training has begun
        public bool contractFlag = false;

        public List<string> output_labels = new List<string>();
        public List<string> input_labels = new List<string>();

        public int model_update_freq;
        public int model_window_time;
        public int model_window_overlap;
        public int model_window_n;
        public int model_window_overlap_n;
        public double model_train_test_split = 0.9;

        public bool modelFlag = false;  // keeps track of whether a model is loaded

        public int train_output_num;
        

        public Data data;
        //public List<double> data.timestamps = new List<double>();
        //public List<List<double>> features = new List<List<double>>();

        //public List<int> data.outputs = new List<int>();
        //public List<int> feature_data.outputs = new List<int>();
        //public List<int> data.input_types = new List<int>();     // keep track of feature type, 0 = generic, 1 = EMG
        //public List<bool> data.input_active_flags = new List<bool>();

        public delegate List<double> pipeline_func(List<double> data);

        public List<pipeline_func> emg_pipeline = new List<pipeline_func>();
        public List<pipeline_func> generic_pipeline = new List<pipeline_func>();

        public List<Func<Vector<double>, Vector<double>>> emg_pipeline_matrix = new List<Func<Vector<double>, Vector<double>>>();
        public List<Func<Vector<double>, Vector<double>>> generic_pipeline_matrix = new List<Func<Vector<double>, Vector<double>>>();


        public dynamic model = new System.Dynamic.ExpandoObject();


        public PR_Logger()
        {
            data = new Data();
            return;
        }

        static public List<double> MAV(List<double> raw_values, int window_n, int window_overlap_n)
        {
            List<double> filtered_values = new List<double>();

            for (int i = 0; i <= (raw_values.Count - window_n); i += (window_n - window_overlap_n))
            {
                List<double> sub_window = raw_values.GetRange(i, window_n);
                filtered_values.Add(sub_window.Sum() / window_n);
            }

            return filtered_values;
        }

        public List<double> MAV(List<double> raw_values)
        {
            // overloaded method to eliminate window time and requency input parameters
            return MAV(raw_values, model_window_n, model_window_overlap);
        }

        List<float> ZeroCrossings(List<float> raw_values, float window_time, float freq)
        {
            // returns list of Zero Crossings features
            List<float> filtered_values = new List<float>();
            int window_n = (int)Math.Ceiling(window_time * freq / 1000);

            for (int i = 0; i < (raw_values.Count - window_n); i++)
            {
                List<float> window = raw_values.GetRange(i, window_n);

                int crossing_count = 0;
                bool positive_flag = (window[0] >= 0);    // true for 

                foreach (float value in window)
                {
                    if ((value >= 0) != positive_flag)
                    {
                        crossing_count++;
                        positive_flag = !positive_flag;
                    }
                }
                filtered_values.Add(crossing_count);
            }

            return filtered_values;
        }

        List<float> SSC(List<float> raw_values, float window_time, float freq)
        {
            // returns list of Slope Sign Change features
            int window_n = (int)Math.Ceiling(window_time * freq / 1000);

            List<float> difference_values = raw_values.Zip(raw_values.Skip(1), (x, y) => y - x).ToList();
            List<float> filtered_values = ZeroCrossings(difference_values, windowlength, freq);
            return filtered_values;
        }

        public bool LoadFileToListRows(string filepath)
        {
            // read data file into lists data.inputs and data.outputs, return false if unsuccessful

            string[] lines_arr = System.IO.File.ReadAllLines(filepath);
            List<string> lines = lines_arr.ToList();

            data.inputs = new List<List<double>>();
            data.outputs = new List<int>();
            data.timestamps = new List<double>();

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
                            if (input_labels[i].ToLower().Contains("ch"))
                            {
                                data.input_types.Add(1);
                            }
                            else
                            {
                                data.input_types.Add(0);
                            }
                        }
                    }
                }
                // parse data lines
                else if (line[0] == 'd')
                {
                    char[] chars_to_trim = { 'd', ',' };
                    List<double> vals = Array.ConvertAll(line.TrimStart(chars_to_trim).Split(','), Double.Parse).ToList();
                    data.outputs.Add((int)vals.Last());
                    vals.RemoveAt(vals.Count - 1);

                    data.timestamps.Add(vals.First());
                    vals.RemoveAt(0);

                    data.inputs.Add(vals);
                }
                else
                {
                    return false; // return if there is no leading character (i.e. not a proper data file)
                }
            }

            // find frequency used for logging
            model_update_freq = (int)Math.Round(data.timestamps.Count * 1000 / (data.timestamps.Last() - data.timestamps.First()));
            model_window_n = (int)Math.Ceiling((double)model_window_time * model_update_freq / 1000);
            return true;
        }

        public bool LoadFileToListCols(string filepath)
        {
            // read data file into lists data.inputs and data.outputs, return false if unsuccessful

            string[] lines_arr = System.IO.File.ReadAllLines(filepath);
            List<string> lines = lines_arr.ToList();

            int input_num = lines[0].Split(',').Length - 3;

            data.Clear();
            //public List<List<double>> inputs;
            //public List<List<double>> features;
            //public List<int> data.outputs;
            //public List<int> feature_data.outputs;
            //public List<int> data.input_types;
            //public List<bool> data.input_active_flags;

            //data.inputs.Clear();
            //data.outputs.Clear();
            //data.timestamps.Clear();
            //data.input_types.Clear();
            //data.input_active_flags.Clear();

            // preallocate data.inputs
            for (int i = 0; i < input_num; i++)
            {
                data.inputs.Add(new List<double>());
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
                            data.input_active_flags.Add(false);

                            if (input_labels[i].ToLower().Contains("ch"))
                            {
                                data.input_types.Add(1);
                            }
                            else
                            {
                                data.input_types.Add(0);
                            }
                        }
                    }
                }
                // parse data lines
                else if (line[0] == 'd')
                {
                    char[] chars_to_trim = { 'd', ',' };
                    List<double> vals = Array.ConvertAll(line.TrimStart(chars_to_trim).Split(','), Double.Parse).ToList();
                    data.outputs.Add((int)vals.Last());
                    vals.RemoveAt(vals.Count - 1);

                    data.timestamps.Add(vals.First());
                    vals.RemoveAt(0);

                    for (int i = 0; i < vals.Count; i++)
                    {
                        data.inputs[i].Add(vals[i]);
                    }
                }
                else
                {
                    return false; // return if there is no leading character (i.e. not a proper data file)
                }
            }

            // find frequency used for logging
            model_update_freq = (int)Math.Round(data.timestamps.Count * 1000 / (data.timestamps.Last() - data.timestamps.First()));

            data.input_num = input_num;
            data.output_num = data.outputs.Max() + 1;

            return true;
        }

        public static void shuffle_training_data<T>(List<List<T>> temp_inputs, List<int> temp_outputs)
        {
            Random rng = new Random();
            int n = temp_outputs.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);

                // swap all data.inputs
                for (int i = 0; i < temp_inputs.Count; i++)
                {
                    T value_T = temp_inputs[i][k];
                    temp_inputs[i][k] = temp_inputs[i][n];
                    temp_inputs[i][n] = value_T;
                }

                int value = temp_outputs[k];
                temp_outputs[k] = temp_outputs[n];
                temp_outputs[n] = value;
            }

            return;
        }

        public List<List<double>> map_features(List<List<double>> temp_inputs)
        {
            // map inputs "temp_inputs" to features using the set "emg_pipeline" and "generic_pipeline"
            // function lists.  Also requires that "data.input_types" are set and "data.input_active_flags" are set

            List<List<double>> temp_features = new List<List<double>>();

            for (int i = 0; i < data.input_types.Count; i++)
            {
                if (data.input_types[i] == 0 && data.input_active_flags[i] == true)
                {
                    foreach (pipeline_func f in generic_pipeline)
                    {
                        temp_features.Add(f(temp_inputs[i]));
                    }
                }
                else if (data.input_types[i] == 1 && data.input_active_flags[i] == true)
                {
                    foreach (pipeline_func f in emg_pipeline)
                    {
                        temp_features.Add(f(temp_inputs[i]));
                    }
                }
            }

            return temp_features;
        }

        public void map_features_training_2()
        {
            data.features.Clear();
            data.feature_outputs.Clear();

            // get indices of output changes
            List<int> output_change_indices = new List<int>();
            output_change_indices.Add(0);
            int cur_output = data.outputs[0];
            for (int i=1; i<data.outputs.Count; i++)
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
                int output_value = data.outputs[output_change_indices[i-1]];
                int end_ind = output_change_indices[i];

                List<List<double>> temp_input = invert_list_list(invert_list_list(data.inputs).GetRange(start_ind, end_ind - start_ind));
                List<List<double>> temp_features = map_features(temp_input);
                
                if (data.features.Count == 0)
                {
                    for (int j = 0; j < temp_features.Count; j++)
                    {
                        data.features.Add(new List<double>());
                    }
                }
                for (int j=0; j<temp_features.Count; j++)
                {
                    data.features[j].AddRange(temp_features[j]);
                }

                data.feature_outputs.AddRange(Enumerable.Repeat(output_value, temp_features[0].Count));

                start_ind = end_ind;
            }

        }

        public void map_features_training()
        {
            data.features.Clear();
            data.feature_outputs.Clear();

            for (int i = 0; i < data.input_types.Count; i++)
            {
                if (data.input_types[i] == 0 && data.input_active_flags[i] == true)
                {
                    foreach (pipeline_func f in generic_pipeline)
                    {
                        List<double> temp_input = new List<double>();
                        List<int> temp_output = new List<int>();
                        int ind = 0;
                        for (int j=0; j<data.output_num; j++)
                        {
                            int ind2 = data.outputs.FindLastIndex(x => x == j);
                            List<double> wtfkk = data.inputs[i].GetRange(ind, ind2-ind+1);
                            temp_input.AddRange(f(wtfkk));
                            temp_output.AddRange(Enumerable.Repeat(j, wtfkk.Count));
                            ind = ind2;
                        }
                        data.features.Add(temp_input);
                        data.feature_outputs = temp_output;
                    }
                }
                else if (data.input_types[i] == 1 && data.input_active_flags[i] == true)
                {
                    foreach (pipeline_func f in emg_pipeline)
                    {
                        List<double> temp_input = new List<double>();
                        List<int> temp_output = new List<int>();
                        int ind = 0;
                        for (int j = 0; j < data.output_num; j++)
                        {
                            int ind2 = data.outputs.FindLastIndex(x => x == j);
                            List<double> wtfkk = data.inputs[i].GetRange(ind, ind2 - ind + 1);
                            temp_input.AddRange(f(wtfkk));
                            temp_output.AddRange(Enumerable.Repeat(j, wtfkk.Count));
                            ind = ind2;
                        }
                        data.features.Add(temp_input);
                        data.feature_outputs = temp_output;
                    }
                }
            }
        }

        public void train_model_Accord_list()
        {

            //data.features = map_features(data.inputs);
            map_features_training_2();

            shuffle_training_data(data.features, data.feature_outputs);

            //// resize outputs to match size of features (i.e. some lines are lost from windowing)
            //data.feature_outputs.Clear();
            //for (int i = 0; i <= (data.outputs.Count - model_window_n); i += (model_window_n - model_window_overlap_n))
            //{
            //    data.feature_outputs.Add(data.outputs[i]);
            //}

            // split to test/train set
            List<List<double>> hmm = invert_list_list(data.features);

            int N_train = (int)(hmm.Count * model_train_test_split);

            List<List<double>> training_features = hmm.GetRange(0, N_train);
            List<int> training_outputs = data.feature_outputs.GetRange(0, N_train);

            List<List<double>> testing_features = hmm.GetRange(N_train, hmm.Count - N_train);
            List<int> testing_outputs = data.feature_outputs.GetRange(N_train, hmm.Count - N_train);

            // Train model
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            var svm = teacher.Learn(training_features.Select(a => a.ToArray()).ToArray(), training_outputs.ToArray());

            model.learner = svm;

            // Compute the machine answers for the data.inputs
            int[] answers = model.learner.Decide(training_features.Select(a => a.ToArray()).ToArray());
            bool[] correct = answers.Zip(training_outputs.ToArray(), (x, y) => x == y).ToArray<bool>();

            double accuracy = (double)correct.Sum() / correct.Length;
            model.accuracy = accuracy;
        }

        public static List<List<T>> invert_list_list<T>(List<List<T>> list)
        {
            // https://stackoverflow.com/questions/39484996/rotate-transposing-a-listliststring-using-linq-c-sharp
            List<List<T>> inverted_list = list
                        .SelectMany(inner => inner.Select((item, index) => new { item, index }))
                        .GroupBy(i => i.index, i => i.item)
                        .Select(g => g.ToList())
                        .ToList();

            return inverted_list;
        }

        public void update_data_with_output(string[] data)
        {
            /*
            appends the current output label to the data and sends data to data logger
            */
            string[] temp = new string[data.Length + 1];
            temp[data.Length] = current_output.ToString();

            for (int i = 0; i < data.Length; i++)
            {
                temp[i] = data[i];
            }
            logger.data_to_write = temp;
        }

        public void write_header_with_output(string[] data)
        {
            /*
            appends the current output label to the header and sends data to data logger
            */
            string[] temp = new string[data.Length + 1];
            temp[data.Length] = "output";

            for (int i = 0; i < data.Length; i++)
            {
                temp[i] = data[i];
            }
            logger.write_header(temp);
        }

        public void set_outputs(List<string> outputs_)
        {
            output_labels = outputs_;
            train_output_num = output_labels.Count;
        }

        public void tick()
        {
            // updates the flags corresponding to current action being performed during training

            if (trainFlag)
            {
                long elapsed_time = (long)(logger.curtime - start_time);
                long segment_time = relax_time + contraction_time;

                current_output = (int)Math.Floor((decimal)elapsed_time / segment_time);

                long local_time = elapsed_time - segment_time * current_output;
                contractFlag = local_time >= relax_time;

                if (contractFlag)
                {
                    logger.recordflag = true;
                }
                else
                {
                    logger.recordflag = false;
                }

                current_cycle = (int)Math.Floor((decimal)elapsed_time / (segment_time * train_output_num));

                if (current_cycle >= collection_cycles)
                {
                    end_data_collection();
                }
            }

            if (modelFlag)
            {

            }
        }

        public void start_data_collection()
        {
            trainFlag = true;
            logger.start();
            logger.tick();
            start_time = (long)logger.curtime;
        }

        public void end_data_collection()
        {
            logger.recordflag = false;
            trainFlag = false;
            logger.file.Flush();
            logger.close();
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

