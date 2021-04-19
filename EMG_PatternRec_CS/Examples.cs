using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using System.Dynamic;
using System.Diagnostics;
using System.Threading;

using EMG_PatternRec_CS.PatternRec;
using EMG_PatternRec_CS.DataLogging;
using EMG_PatternRec_CS.Mapping;

using Accord.IO;
using Accord.Math;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Accord.Statistics.Analysis;
using NWaves.Filters;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace EMG_PatternRec_CS.Examples
{
    /// <summary>
    /// Here are example code snippets for each main item in the EMG_PatternRec_CS library
    /// </summary>
    public static class Examples
    {

        public static DataLogger DataLoggerBasicUsage()
        {

            DataLogger logger = new DataLogger();   // create logger
            string solutionFilePath = Directory.GetParent(System.IO.Directory.GetCurrentDirectory()).Parent.Parent.FullName;
            string dataFilePath = Path.Combine(solutionFilePath, @"data\dataloggerdemo.csv");

            logger.init_file(dataFilePath); // create filewriter
            logger.freq = 100;   // set frequency to 100Hz
            logger.signal_num = 8;  // set number of signals to 8
            get_data_f_func<double> f = () => random_data_grabber(logger.signal_num);   // create data grabbing function
            logger.get_data_f = f;  // set data grabbing function

            logger.start(); // start thread
            logger.recordflag = true; // start recording
            Thread.Sleep(3000); // record for 3 seconds
            logger.recordflag = false;  // stop recording
            logger.stop();  // stop thread
            logger.close_file();    // close file

            return logger;
        }

        public static PR_Logger PR_LoggerBasicUsage()
        {

            PR_Logger PR_logger = new PR_Logger();
            string solutionFilePath = Directory.GetParent(System.IO.Directory.GetCurrentDirectory()).Parent.Parent.FullName;
            string dataFilePath = Path.Combine(solutionFilePath, @"data\PR_dataloggerdemo.csv");
            string jsonfilepath = Path.Combine(solutionFilePath, @"data\PR_dataloggerdemo.json");

            PR_logger.init_file(dataFilePath);  // create filewriter
            PR_logger.freq = 100;    // set frequency to 100Hz
            PR_logger.signal_num = 8;   // set number of signals to 8
            get_data_f_func<double> f = () => random_data_grabber(PR_logger.signal_num);   // create data grabbing function to grab 8 inputs
            PR_logger.get_data_f = f;  // set data grabbing function

            PR_logger.collection_cycles = 1;    // set number of colleciton cycles
            PR_logger.contraction_time = 1000;   // set contraction time per output class to 1000ms
            PR_logger.relax_time = 1000;     // set relaxation time between outputs to 100ms
            PR_logger.output_labels = new List<string> { "rest", "extension", "flexion" };  // set output labels
            PR_logger.train_output_num = PR_logger.output_labels.Count; // set number of outputs


            int current_class = -1;

            PR_logger.start_data_collection();  // start data collection
            while (PR_logger.trainFlag) { 

                PR_logger.PR_tick();

                if (PR_logger.current_output != current_class)
                {
                    current_class = PR_logger.current_output;
                    Console.WriteLine("Collecting data for class: " + current_class.ToString());
                }

                if (PR_logger.recordflag)
                {
                    Console.WriteLine("Recording Data");
                }
                else
                {
                    Console.WriteLine("Relax");
                }
                

                Thread.Sleep(250);
            };

            // put all relevant information to DataManager object and save as JSON for future pattern rec use
            DataManager data = new DataManager();
            data.freq = PR_logger.freq;
            data.input_labels = new List<string> { "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8" };
            data.input_num = data.input_labels.Count;
            data.output_labels = PR_logger.output_labels;
            data.output_num = PR_logger.output_labels.Count;
            data.collection_cycles = PR_logger.collection_cycles;
            data.relaxation_time = PR_logger.relax_time;
            data.contraction_time = PR_logger.contraction_time;
            data.SetAllInputActiveFlags(true);  // set all inputs to active
            data.input_types = new List<int> { 1, 1, 1, 1, 1, 1, 1, 1 };
            ObjLogger.saveObjJson(jsonfilepath, data);

            return PR_logger;
        }

        public static Mapper MapperBasicUsage()
        {
            string solutionFilePath = Directory.GetParent(System.IO.Directory.GetCurrentDirectory()).Parent.Parent.FullName;
            string dataSettingsFilePath = Path.Combine(solutionFilePath, @"data\PR_dataloggerdemo.json");
            string dataFilePath = Path.Combine(solutionFilePath, @"data\PR_dataloggerdemo.csv");

            // load data (this file has only emg inputs)
            DataManager data = ObjLogger.loadObjJson<DataManager>(dataSettingsFilePath);
            data.LoadFileToListCols(dataFilePath);

            // create mapping object
            Mapper mapper = new Mapper();
            mapper.window_size_n = 5;
            mapper.window_overlap_n = 2;
            mapper.window_n = 1;

            // add scalers to pipeline for emg inputs
            mapper.mean_values = new List<double>();
            mapper.max_values = new List<double>();
            mapper.min_values = new List<double>();

            foreach (List<double> channel in data.inputs)
            {
                mapper.mean_values.Add(channel.Average());
                mapper.max_values.Add(channel.Max());
                mapper.min_values.Add(channel.Min());
            }

            Mapper.scaler_pipeline_func fscaler = (x, i) => Scalers.MinMaxZeroCenter(x, mapper.min_values, mapper.max_values, mapper.mean_values, i);
            mapper.emg_scaler_pipeline.Add(fscaler);

            // add notch filter to pipeline for emg inputs
            double notch_fc = 60;
            double fs = data.freq;
            var NotchFilter = Filters.create_notch_filter(notch_fc, fs);

            // create notch filter for each input signal
            List<NWaves.Filters.Base.IOnlineFilter> notchfilters50Hz = new List<NWaves.Filters.Base.IOnlineFilter>();
            for (int i = 0; i < data.input_num; i++)
            {
                notchfilters50Hz.Add(NotchFilter);
            }
            // add function to filter pipeline that accepts raw data and channel number as inputs
            mapper.emg_filter_pipeline.Add((x, i) => Filters.apply_filter(notchfilters50Hz, x, i));

            Mapper.feature_pipeline_func fdummy = (x) => Features.dummy_feature_example(x, mapper.window_size_n, mapper.window_overlap_n, 3);

            // add raw value feature to feature pipeline that accepts raw data as input
            Mapper.feature_pipeline_func f = (x) => Features.RAW(x, mapper.window_size_n, mapper.window_overlap_n);
            mapper.emg_feature_pipeline.Add(f);
            // add mean absolute value feature to feature pipeline that accepts raw data as input
            mapper.emg_feature_pipeline.Add(x => Features.MAV(x, mapper.window_size_n, mapper.window_overlap_n));
            // add variance feature to feature pipeline that accepts raw data as input
            mapper.emg_feature_pipeline.Add(x => Features.VAR(x, mapper.window_size_n, mapper.window_overlap_n));

            // get raw values
            List<List<double>> raw_values = data.inputs;
            List<int> input_types = data.input_types;
            List<bool> input_active_flags = data.input_active_flags;

            // get filtered values
            List<List<double>> filtered_values = mapper.filter_signals(raw_values, input_types, input_active_flags);

            // get scaled values using filtered values
            List<List<double>> scaled_values = mapper.scale_signals(filtered_values, input_types, input_active_flags);

            // get features using scaled values
            List<List<double>> features = mapper.map_features(scaled_values, input_types, input_active_flags);

            return mapper;
        }

        public static PostProcessor PostProcessorBasicUsage()
        {
            PostProcessor postprocessor = new PostProcessor();
            return postprocessor;
        }

        public static Model ModelBasicUsage()
        {
            string solutionFilePath = Directory.GetParent(System.IO.Directory.GetCurrentDirectory()).Parent.Parent.FullName;
            string dataSettingsFilePath = Path.Combine(solutionFilePath, @"data\PR_dataloggerdemo.json");
            string dataFilePath = Path.Combine(solutionFilePath, @"data\PR_dataloggerdemo.csv");
            string mapperSettingsFilePath = Path.Combine(solutionFilePath, @"data\mappingdemo.json");

            Model model = new Model();  // create model
            model.data = ObjLogger.loadObjJson<DataManager>(dataSettingsFilePath); // load data settings from DataLogger file 
            model.data.LoadFileToListCols(dataFilePath);    // load data from corresponding data file
            model.mapper = MapperBasicUsage();  // set up mapping
            model.model = new AccordLDAModel(); // create predictor

            model.train_test_split = 0.9;   // set test/train split
            model.train_model();    // train model

            // create random garbage input
            List<List<double>> random_input = new List<List<double>>();
            for (int i = 0; i < model.data.input_num; i++)
            {
                random_input.Add(random_data_grabber(model.mapper.window_size_n));
            }

            double[] scores = model.get_scores(random_input);   // predict score

            return model;
        }

        public static RealTimeModel RealTimeBasicUsage()
        {
            Model model = ModelBasicUsage();
            RealTimeModel RTmodel = new RealTimeModel(model);   // create realtime model

            // specify parameters for inherited DataLogger class to grab data in realtime
            get_data_f_func<double> f = () => random_data_grabber(RTmodel.model.data.input_num);   // create data grabbing function
            RTmodel.get_data_f = f;
            RTmodel.start();
            RTmodel.realtimeFilterFlag = true;
            RTmodel.realtimePredictFlag = true;

            // RTmodel will now be updating it's prediction at the same frequency it was trained on

            Stopwatch sw = new Stopwatch();
            sw.Start();
            TimeSpan duration = new TimeSpan(0, 0, 2);
            while (sw.Elapsed < duration)
            {
                // get current score from RTmodel
                double[] scores = RTmodel.RealTimeScores;
                Thread.Sleep(50);   // do other stuff for approximately 50 ms
            }

            return RTmodel;
        }

        public static IPredictor ONNXBasicUsage()
        {

            string solutionFilePath = Directory.GetParent(System.IO.Directory.GetCurrentDirectory()).Parent.Parent.FullName;
            string onnxfilepath = Path.Combine(solutionFilePath, @"data\generalized_classifier_v3.onnx");
            //string onnxfilepath = Path.Combine(solutionFilePath, @"data\output.onnx");

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

            return onnxmodel;
        }

        public static void ONNXFromSettingsBasicUsage()
        {

            string solutionFilePath = Directory.GetParent(System.IO.Directory.GetCurrentDirectory()).Parent.Parent.FullName;
            string modelSettingsFilePath = Path.Combine(solutionFilePath, @"data\generalized_classifier_v3.json");
            string onnxfilepath = Path.Combine(solutionFilePath, @"data\generalized_classifier_v3.onnx");
            //string onnxfilepath = Path.Combine(solutionFilePath, @"data\output.onnx");

            Model model = ObjLogger.loadObjJson<Model>(modelSettingsFilePath);
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

        static Random rng = new Random(1);
        private static List<double> random_data_grabber(int datanum)
        {
            List<double> data = new List<double>();
            for (int i = 0; i < datanum; i++)
            {
                data.Add(rng.Next());
            }
            return data;
        }

        private static List<double> sinusoid_data_grabber(int datanum, double fs, double[] freqs)
        {
            List<double> data = new List<double>(datanum);
            for (int i = 0; i < datanum; i++)
            {
                data.Add(0);
                foreach(double freq in freqs)
                {
                    data[i] += (Math.Sin(2 * Math.PI * freq * (i / fs)));
                }
            }
            return data;
        }

        private static List<double> time_data_grabber(int datanum, double fs)
        {
            List<double> data = new List<double>();
            for (int i = 0; i < datanum; i++)
            {
                data.Add(i / fs);
            }
            return data;
        }

        public static void FilterTesting()
        {
            string solutionFilePath = Directory.GetParent(System.IO.Directory.GetCurrentDirectory()).Parent.Parent.FullName;
            string dataSaveFilePath = Path.Combine(solutionFilePath, @"data\filterdemo.csv");

            int n_points = 200; // number of data points
            double fs = 200;    // sample rate (Hz)

            // raw data
            List<double> data = sinusoid_data_grabber(n_points, fs, new double[] { 5, 10, 60 });
            List<double> time = time_data_grabber(n_points, fs);

            // notch filter
            double notch_fc = 60;   // cutoff frequency (Hz)
            var NotchFilter = Filters.create_notch_filter(notch_fc, fs);
            List<double> notch_filtered_data = Filters.apply_filter(NotchFilter, data);

            // low pass filter
            double LP_fc = 20;  // cutoff frequency (Hz)
            int LP_order = 5;   // filter order (number of terms)
            var LPfilter = Filters.create_lowpass_butterworth_filter(LP_fc, fs, LP_order);
            List<double> LP_filtered_data = Filters.apply_filter(LPfilter, data);

            // high pass filter
            double HP_fc = 50;
            int HP_order = 5;
            var HPfilter = Filters.create_highpass_butterworth_filter(HP_fc, fs, HP_order);
            List<double> HP_filtered_data = Filters.apply_filter(HPfilter, data);

            // moving average filter
            int MA_order = 10;
            var MAfilter = Filters.create_movingaverage_filter(MA_order);
            List<double> MA_filtered_data = Filters.apply_filter(MAfilter, data);

            StreamWriter file = new StreamWriter(dataSaveFilePath);
            for (int i = 0; i < data.Count; i++)
            {
                file.WriteLine(time[i].ToString("F3") + "," +
                                data[i].ToString("F3") + "," +
                                notch_filtered_data[i].ToString("F3") + "," +
                                LP_filtered_data[i].ToString("F3") + "," +
                                HP_filtered_data[i].ToString("F3") + "," +
                                MA_filtered_data[i].ToString("F3"));
            }  

            file.Flush();

        }

        public static void MultidimensionalFilterTest()
        {
            int fs = 2000;
            int HP_fc = 50;
            int HP_order = 5;
            int num_points = 1000;
            int num_inputs = 5;

            // create multi-dimensional data of size num_inputs x num_points (i.e. each sub-list is an input signal)
            List<List<double>> multi_dimensional_data = new List<List<double>>();
            for (int i = 0; i < num_inputs; i++)
            {
                multi_dimensional_data.Add(sinusoid_data_grabber(num_points, fs, new double[] { 5, 10, 60 }));
            }

            // create list of high pass filters, one for each signal
            List<NWaves.Filters.Base.IOnlineFilter> HP_filters = new List<NWaves.Filters.Base.IOnlineFilter>();
            for (int i = 0; i < num_inputs; i++)
            {
                HP_filters.Add(Filters.create_highpass_butterworth_filter(HP_fc, fs, HP_order));
            }

            // apply each filter to each input signal seperately
            List<List<double>> filtered_multi_dimensional_data = new List<List<double>>();
            for (int i = 0; i < num_inputs; i++)
            {
                filtered_multi_dimensional_data.Add(Filters.apply_filter(HP_filters, multi_dimensional_data[i], i));
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

        private static void generate_random_data(string filepath, int input_num, int num_lines)
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
