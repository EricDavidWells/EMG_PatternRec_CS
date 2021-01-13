using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using MathNet.Numerics.LinearAlgebra;
using Microsoft.ML;
using Microsoft.ML.Data;

using Accord.Controls;
using Accord.IO;
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;


// useful: https://dotnetfiddle.net/
// potential library: http://accord-framework.net/

namespace brachIOplexus
{
    class PatternRec
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

        public List<string> output_labels;
        public List<string> input_labels;
        public int output_num;
        public int input_num;

        public int model_update_freq = 50;
        public int model_window_time = 100;
        public int model_window_overlap;

        public bool modelFlag = false;  // keeps track of whether a model is loaded

        public List<float> timestamps = new List<float>();
        public List<List<float>> inputs = new List<List<float>>();
        public List<List<float>> features = new List<List<float>>();
        public List<int> outputs = new List<int>();
        public List<int> feature_outputs = new List<int>();
        public List<int> input_types = new List<int>();     // keep track of feature type, 0 = generic, 1 = EMG

        public List<Func<List<float>, List<float>>> emg_pipeline = new List<Func<List<float>, List<float>>>();
        public List<Func<List<float>, List<float>>> generic_pipeline = new List<Func<List<float>, List<float>>>();

        public PatternRec()
        {
            //Matrix<float> A = Matrix<float>.Build.Random(3, 4);
            //A.SubMatrix
            //Accord.Math.Matrix.Get()
        }

        public List<float> MAV(List<float> raw_values, float window_time, float freq)
        {
            // returns list of Mean Absolute Value features

            List<float> filtered_values = new List<float>();
            int window_n = (int)Math.Ceiling(window_time * freq / 1000);

            for (int i=0; i<(raw_values.Count-window_n); i++)
            {
                List<float> window = raw_values.GetRange(i, window_n);

                int sum = 0;
                foreach(float value in window)
                {
                    sum += (int)Math.Abs(value);
                }
                filtered_values.Add(sum / window_n);
            }
            return filtered_values;
        }

        public List<float> MAV(List<float> raw_values)
            // overloaded method to eliminate window time and requency input parameters
        {
            return MAV(raw_values, model_window_time, model_update_freq);
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

            List<float> difference_values = raw_values.Zip(raw_values.Skip(1), (x, y) => y-x).ToList();
            List<float> filtered_values = ZeroCrossings(difference_values, windowlength, freq);
            return filtered_values;
        }

        public bool LoadFileToListRows(string filepath)
        {
            // read data file into lists inputs and outputs, return false if unsuccessful

            string[] lines_arr = System.IO.File.ReadAllLines(filepath);
            List<string> lines = lines_arr.ToList();

            inputs = new List<List<float>>();
            outputs = new List<int>();
            timestamps = new List<float>();

            foreach (string line in lines)
            {
                // parse header line
                if (line[0] == 'h')
                {
                    char[] chars_to_trim = {'h', ',' };
                    input_labels = line.TrimStart(chars_to_trim).Split(',').ToList<string>();

                    if (input_labels.Last() != "output")
                    {
                        return false;  // return if there is no output column
                    }
                    else
                    {
                        input_labels.RemoveAt(input_labels.Count - 1);
                        input_labels.RemoveAt(0);

                        for (int i=0; i<input_labels.Count; i++)
                        {
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
                    List<float> vals = line.TrimStart(chars_to_trim).Split(',').ToSingle().ToList<float>();
                    outputs.Add((int)vals.Last());
                    vals.RemoveAt(vals.Count-1);

                    timestamps.Add(vals.First());
                    vals.RemoveAt(0);

                    inputs.Add(vals);
                }
                else   
                {
                    return false; // return if there is no leading character (i.e. not a proper data file)
                }
            }

            // find frequency used for logging
            model_update_freq = (int)Math.Round(inputs.Count*1000 / (inputs.Last()[0] - inputs.First()[0]));
            return true;
        }

        public bool LoadFileToListCols(string filepath)
        {
            // read data file into lists inputs and outputs, return false if unsuccessful

            string[] lines_arr = System.IO.File.ReadAllLines(filepath);
            List<string> lines = lines_arr.ToList();

            int input_num = lines[0].Split(',').Length - 3;


            inputs.Clear();
            outputs.Clear();
            timestamps.Clear();
            input_types.Clear();

            // preallocate inputs
            for (int i = 0; i < input_num; i++)
            {
                inputs.Add(new List<float>());
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
                    List<float> vals = line.TrimStart(chars_to_trim).Split(',').ToSingle().ToList<float>();
                    outputs.Add((int)vals.Last());
                    vals.RemoveAt(vals.Count - 1);

                    timestamps.Add(vals.First());
                    vals.RemoveAt(0);

                    for (int i=0; i<vals.Count; i++)
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
            model_update_freq = (int)Math.Round(timestamps.Count * 1000 / (timestamps.Last() - timestamps.First()));
            return true;
        }

        public void map_features(List<List<float>> data)
        {

            features.Clear();
            feature_outputs.Clear();
            int new_feature_size = 0;

            foreach(Func<List<float>, List<float>> f in generic_pipeline)
            {
                foreach(List<float> input in data)
                {
                    features.Add(f(input));
                    new_feature_size = features[0].Count();
                }
            }

            // trim outputs due to windowing of features
            feature_outputs.AddRange(outputs.GetRange(0, new_feature_size));
        }

        public void train_model_Accord()
        {
            float[][] inputs = features.Select(a => a.ToArray()).ToArray();

            double[][] inputs_dub = Array.ConvertAll(inputs, x => Array.ConvertAll(x, y => (double)y));

            int n = inputs.Length;
            outputs.RemoveRange(n, outputs.Count - n);
            
            
            //outputs.RemoveRange(outputs.cou, 3);
            //{
            //    new double[] { 0 },
            //    new double[] { 3 },
            //    new double[] { 1 },
            //    new double[] { 2 },
            //};

            //// Outputs for each of the inputs
            //int[] outputs =
            //{
            //    0,
            //    3,
            //    1,
            //    2,
            //};


            // Create the Multi-label learning algorithm for the machine
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                Learner = (p) => new SequentialMinimalOptimization<Linear>()
                {
                    Complexity = 10000.0 // Create a hard SVM
                }
            };

            // Learn a multi-label SVM using the teacher
            var svm = teacher.Learn(inputs_dub, outputs.ToArray());

            // Compute the machine answers for the inputs
            int[] answers = svm.Decide(inputs_dub);

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
            output_num = output_labels.Count;
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

                current_cycle = (int)Math.Floor((decimal)elapsed_time / (segment_time*output_num));

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
            logger.recordflag = true;
        }

        public void end_data_collection()
        {
            logger.recordflag = false;
            trainFlag = false;
            logger.file.Flush();
            logger.close();
        }
    }
}

