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

        public int model_update_freq;
        public bool modelFlag = false;  // keeps track of whether a model is loaded

        public List<float> timestamps;
        public List<List<float>> inputs;
        public List<int> outputs;
        public List<int> feature_types;     // keep track of feature type, 0 = generic, 1 = EMG

        public PatternRec()
        {
            //Matrix<float> A = Matrix<float>.Build.Random(3, 4);
            //A.SubMatrix
        }

        void MAV(List<float> raw_values, float window_time, float freq)
        {
            // should I use timestamps or have consistent window length??????????? 

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
        }

        public bool LoadFileToList(string filepath)
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

        //public bool LoadFile(string filepath)
        //{

        //    string[] lines = System.IO.File.ReadAllLines(filepath);
        //    num_cols = lines[0].Split(',').Length - 1; // number of data columns in the file (-1 for line label)
        //    num_lines = lines.Length - 1;    // number of lines in the data file (-1 for header)

        //    // initialize raw data jagged array
        //    raw_data = new float[num_cols][];
        //    for (int i = 0; i < num_cols; i++)
        //    {
        //        raw_data[i] = new float[num_lines];
        //    }

        //    // ensure header is first line in file
        //    string headerline = lines[0];
        //    if (headerline[0] != 'h')
        //    {
        //        return false;
        //    }

        //    // parse header line
        //    char[] chars_to_trim = { 'd', 'h', ',' };
        //    headerline = headerline.TrimStart(chars_to_trim);
        //    raw_data_labels = headerline.Split(',');

        //    // parse data lines
        //    for (int i=0; i<num_lines-1; i++)
        //    {
        //        string line = lines[i+1];   // skip first line
                
        //        // ensure all lines have data stamp as first index
        //        if (line[0] != 'd')
        //        {
        //            return false;
        //        }

        //        line = line.TrimStart(chars_to_trim);
        //        string[] vals = line.Split(',');
                
        //        for (int j=0; j<num_cols; j++)
        //        {
        //            raw_data[j][i] = float.Parse(vals[j]);
        //        }
                
        //    }

        //    // split up into input and output data arrays
        //    input_data = new float[num_cols-2][];
        //    for (int i=0; i < num_cols - 2; i++)
        //    {
        //        input_data[i] = new float[num_lines];
        //        input_data[i] = raw_data[i + 1];
        //    }

        //    output_data = new int[num_lines];
        //    for (int i=0; i < num_lines; i++)
        //    {
        //        output_data[i] = (int)raw_data[num_cols-1][i];
        //    }

        //    return true;
        //}

        public void train_model_Accord()
        {
            double[][] inputs =
            {
                new double[] { 0 },
                new double[] { 3 },
                new double[] { 1 },
                new double[] { 2 },
            };

            // Outputs for each of the inputs
            int[] outputs =
            {
                0,
                3,
                1,
                2,
            };


            // Create the Multi-label learning algorithm for the machine
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                Learner = (p) => new SequentialMinimalOptimization<Linear>()
                {
                    Complexity = 10000.0 // Create a hard SVM
                }
            };

            // Learn a multi-label SVM using the teacher
            var svm = teacher.Learn(inputs, outputs);

            // Compute the machine answers for the inputs
            int[] answers = svm.Decide(inputs);
        }

        //public void train_model()
        //{
        //    // example from cookbook on github: https://github.com/dotnet/machinelearning/blob/master/docs/code/MlNetCookBook.md%23how-do-i-load-data-from-a-text-file
        //    var context = new MLContext();

        //    string trainDataPath = @"C:\Users\Rico5678\Desktop\garbage data\20210108_123341_keyboarddata.csv";


        //    //// Load the data into a data view, class based method
        //    //var data = context.Data.LoadFromTextFile<RegressionData>(trainDataPath,
        //    //                // Default separator is tab, but we need a semicolon.
        //    //                separatorChar: ',',
        //    //                // First line of the file is a header, not a data row.
        //    //                hasHeader: true
        //    //);

        //    var loader = context.Data.CreateTextLoader(new[] {
        //            // A boolean column depicting the 'label'.
        //            new TextLoader.Column("Features", DataKind.Single, 1, num_cols-1),
        //            // Three text columns.
        //            new TextLoader.Column("Label", DataKind.UInt16, num_cols),
        //        },
        //        // First line of the file is a header, not a data row.
        //        hasHeader: true, 
        //        separatorChar: ','
        //    );

        //    var data = loader.Load(trainDataPath);

        //    var pipeline = context.Transforms.Conversion.MapValueToKey("Label").Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy());
        //    //var temp = data.Preview();
        //    var split = context.Data.TrainTestSplit(data, testFraction: 0.1);
        //    var model = pipeline.Fit(split.TrainSet);
        //    var metrics = context.MulticlassClassification.Evaluate(model.Transform(split.TestSet));

        //    // make a prediction:

        //    var schemaDef = SchemaDefinition.Create(typeof(ClassificationVectorData));
        //    schemaDef["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, num_cols - 1);

        //    var temp__ = new ClassificationVectorData();
        //    temp__.Features = new float[] { 1, 0, 0, 500 };

        //    //var predictionFunc = context.Model.CreatePredictionEngine<ClassificationVectorData, ClassificationPrediction>(model, schemaDef);


        //    //var prediction = predictionFunc.Predict(temp__);

        //    //var dataViewBuilder = new DataViewSchema.Builder();
        //    //dataViewBuilder.AddColumn("Feature")


        //}


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

    class RegressionData
    {

        [LoadColumn(1), ColumnName("Features")]
        public float FeatureVector { get; set; }


        [LoadColumn(2), ColumnName("Label")]
        public uint Target { get; set; }
    }

    class ClassificationVectorData
    {
        [ColumnName("Features")]
        public float[] Features { get; set; }

        [ColumnName("Label")]
        public UInt16 Label { get; set;}

    }

    class ClassificationPrediction
    {
        [ColumnName("Label")]
        public UInt16 PredictedClass { get; set; }
    }
}

