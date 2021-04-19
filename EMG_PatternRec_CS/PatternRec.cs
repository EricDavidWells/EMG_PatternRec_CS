using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using System.Diagnostics;
using System.Threading;

using Accord.IO;
using Accord.Math;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Accord.Statistics.Analysis;
using Microsoft.ML;
using Microsoft.ML.Data;

using EMG_PatternRec_CS.DataLogging;
using EMG_PatternRec_CS.Mapping;

namespace EMG_PatternRec_CS.PatternRec
{
    /// <summary>
    /// class to hold all information from a data recording session
    /// </summary>
    public class DataManager
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
    /// Predictor interface to standardize implemented predictors
    /// </summary>
    public interface IPredictor
    {
        /// <summary>type of model</summary>
        string model_type { get; set; }
        /// <summary>indicates model has been trained</summary>
        bool is_trained { get; set; }

        /// <summary>
        /// predicts output
        /// </summary>
        /// <param name="input">input features</param>
        /// <returns></returns>
        double[] predict(double[] input);

        /// <summary>
        /// trains model
        /// </summary>
        /// <param name="inputs">input features</param>
        /// <param name="outputs">ground truth classes</param>
        void train(double[][] inputs, int[] outputs);

        /// <summary>
        /// serializes model to file and delete model parameter
        /// </summary>
        /// <param name="filepath"></param>
        void save(string filepath);

        /// <summary>
        /// deserializes model from file update model parameter 
        /// </summary>
        /// <param name="filepath"></param>
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
            double[] results = learner.Probabilities(input);
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
            double[] results = learner.Probabilities(input);
            
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
            double min_value = results.Min();
            double max_value = results.Max();
            for (int i = 0; i < results.Length; i++)
            {
                results[i] = (results[i] - min_value) / (max_value -min_value);
            }
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
            var pipeline = mlContext.Transforms.ApplyOnnxModel(filepath, fallbackToCpu: true);
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
        public DataManager data;
        public Mapper mapper;
        public IPredictor model;
        public PostProcessor postprocessor;

        /// <summary>percentage of total data to use for testing data</summary>
        public double train_test_split;
        /// <summary>accuracy of trained model</summary>
        public double accuracy;
        /// <summary>flag to indicate whether real-time is enabled</summary>

        public Model()
        {
            data = new DataManager();
            mapper = new Mapper();
            //logger = new DataLogger();
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

                List<List<double>> temp_input = DataManager.transpose_list_list(DataManager.transpose_list_list(data.inputs).GetRange(start_ind, end_ind - start_ind));
                List<List<double>> temp_input_filtered = mapper.filter_signals(temp_input, data.input_types, data.input_active_flags);
                List<List<double>> temp_input_scaled = mapper.scale_signals(temp_input_filtered, data.input_types, data.input_active_flags);
                List<List<double>> temp_features = mapper.map_features(temp_input_scaled, data.input_types, data.input_active_flags);
                //List<List<double>> temp_features = mapper.map_all(temp_input, data.input_types, data.input_active_flags);

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
            DataManager.shuffle_training_data(data.features, data.feature_outputs);

            // split to test/train set
            List<List<double>> features_rows = DataManager.transpose_list_list(data.features);
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
        /// predicts scores for a single input of filtered data
        /// </summary>
        /// <param name="inputs_">already filtered data</param>
        /// <returns></returns>
        public double[] get_scores(List<List<double>> input)
        {
            List<List<double>> scaled_signals = mapper.scale_signals(input, data.input_types, data.input_active_flags);
            List<List<double>> features_ = mapper.map_features(scaled_signals, data.input_types, data.input_active_flags);

            //double[] features_flattenned = DataManager.transpose_list_list(features_).SelectMany(i => i).ToArray();
            double[] features_flattenned = features_.SelectMany(i => i).ToArray();
            double[] scores = model.predict(features_flattenned);
            double[] post_processed_scores = postprocessor.process(scores);
            return post_processed_scores;
        }

    }

    /// <summary>
    /// Enables consistent real-time prediction from a Model object at the specified frequency it was trained on.
    /// Inherits from DataLogger class
    /// </summary>
    /// <para>
    /// A seperate digital filter must be used for each signal due to digital filters reliance on previous inputs.
    /// These digital filters rely on signals at the specified sampling frequency, and must be updated consistently, not just each time a prediction is made.
    /// Similarly, the post-processing algorithms are dependent on sampling rate, and must be done consistently.
    /// </para>
    public class RealTimeModel : DataLogger
    {
        public Model model;

        /// <summary> if true, data will be filtered using model.mapper each time data is collected</summary>
        public bool realtimeFilterFlag;
        /// <summary> if true, prediction will be updated each time data is collected</summary>
        public bool realtimePredictFlag;
        /// <summary> gets current realtime prediction scores</summary>
        public double[] RealTimeScores
        {
            get
            {
                lock (lockObj)
                {
                    return _realtimeScores;
                }
            }
        }
        private double[] _realtimeScores;


        public RealTimeModel(Model model_)
        {
            model = model_;
            historyflag = true;
            history_num = model.mapper.window_size_n + (model.mapper.window_size_n - model.mapper.window_overlap_n) * (model.mapper.window_n - 1);
            freq = model.data.freq;
            signal_num = model.data.input_num;
            _realtimeScores = new double[model.data.output_num];
            return;
        }


        /// <summary>
        /// override DataLoggers thread to filter all inputs and make a prediction on each successive data grab
        /// </summary>
        public override void thread_loop()
        {
            while (true)
            {
                tick();
                lock (lockObj)  // lock safe data writing
                {
                    // get data and filter if desired
                    if (realtimeFilterFlag)
                    {
                        List<List<double>> raw_data = new List<List<double>>();
                        raw_data.Add(get_data_f()); // get current data into a list of lists that mapper.filter_signals is expecting
                        // filter raw data
                        List<List<double>> filtered_data = model.mapper.filter_signals(DataManager.transpose_list_list(raw_data), model.data.input_types, model.data.input_active_flags);
                        _data = filtered_data.SelectMany(i => i).ToList();  // flatten list of lists back to single list
                    }
                    else
                    {
                        _data = get_data_f();
                    }

                    if (historyflag)
                    {
                        for (int i = 0; i < signal_num; i++)
                        {
                            _data_history[i].Add(_data[i]);
                            _data_history[i].RemoveAt(0);
                        }
                    }

                    if (realtimePredictFlag)
                    {
                        // make prediction
                        _realtimeScores = model.get_scores(Data_history);
                    }
                }
            }
        }
    }

}
