using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Threading;
using Newtonsoft.Json;

namespace RealTimePatternRec.DataLogging
{
    public delegate List<T> get_data_f_func<T>();

    /// <summary>
    /// Creates an object that runs on it's own thread to manage data collection at reliable time intervals
    /// <para> 
    /// Note that the thread spins when not logging to update its timer and therefore clogs cpu usage.
    /// This makes data sample rate as consistent as possible on a windows OS
    /// </para>
    /// </summary>
    /// 
    public class dataLogger
    {
        /// <summary>path to streamwriter file location</summary>
        public string filepath;
        /// <summary>streamwriter object for writing values to file </summary>
        public StreamWriter file;
        /// <summary>flag to trigger data recording</summary>
        public bool recordflag = false;
        /// <summary>flag to trigger</summary>
        public bool historyflag = false;
        /// <summary>stopwatch to keep track of logging frequency</summary>
        public Stopwatch sw = new Stopwatch();
        /// <summary>frequency to log data in Hz</summary>
        public int freq;
        /// <summary>number of signals being logged</summary>
        public int signal_num;
        /// <summary>number of data points to keep in history</summary>
        public int history_num;
        /// <summary>delegate function used to grab data</summary>
        public get_data_f_func<double> get_data_f;
        /// <summary>object to be used in lock statement while reading data from logger</summary>
        public object lockObj = new object();

        protected List<double> _data;
        /// <summary>current data</summary>
        public List<double> Data
        {
            get
            {
                lock (lockObj)
                {
                    return _data;
                }
            }
        }
        protected List<List<double>> _data_history = new List<List<double>>();
        /// <summary>data history property</summary>
        public List<List<double>> Data_history
        {
            get
            {
                lock (lockObj)
                {
                    return _data_history;
                }
            }
        }

        // thread and timing variables
        protected float prevtime;
        protected float curtime;
        protected Thread t;

        /// <summary>format specifier for converting data values from double to strings</summary>
        protected string formatspec = "F4";

        public dataLogger()
        {
            // creates dataLogger object
        }

        /// <summary>
        /// initiates file stream writer
        /// </summary>
        /// <param name="filepath_"></param>
        public void init_file(string filepath_)
        {
            filepath = filepath_;
            file = new StreamWriter(filepath);
        }

        /// <summary>
        /// closes file stream writer
        /// </summary>
        public void close_file()
        {
            if (file != null)
            {
                file.Flush();
                file.Dispose();
                file = null;
            }
        }

        /// <summary>
        /// starts data grabbing thread
        /// </summary>
        public void start()
        {
            sw.Start();
            prevtime = sw.ElapsedMilliseconds;
            curtime = prevtime;

            // initiate data history
            if (historyflag)
            {
                _data_history.Clear();
                for (int i=0; i<signal_num; i++)
                {
                    List<double> temp_list = new List<double>();
                    for (int j=0; j<history_num; j++)
                    {
                        temp_list.Add(0);
                    }
                    _data_history.Add(temp_list);
                }
            }

            // start thread
            if (t == null)
            {
                t = new Thread(thread_loop);
                t.Start();
                //t.Priority = ThreadPriority.BelowNormal;
            }
        }

        /// <summary>
        /// kills thread if thread is currently running
        /// </summary>
        public void stop()
        {
            if (t != null)
            {
                t.Abort();
                t = null;
            }
        }

        /// <summary>
        /// writes data to file as comma seperated values
        /// </summary>
        /// <param name="data">data to write as list of strings</param>
        public void write_csv(List<string> data)
        {
            string newLine = "";
            for (int i=0; i<data.Count; i++)
            {
                newLine += data[i] + ",";
            }
            file.WriteLine(newLine.TrimEnd(','));
        }

        /// <summary>
        /// writes timestamp before writing data as comma seperated value
        /// </summary>
        /// <param name="data"></param>
        public virtual void write_data_with_timestamp(List<string> data)
        {
            file.Write(curtime.ToString(formatspec) + ',');
            write_csv(data);
        }

        /// <summary>
        /// updates stopwatch and flips timeflag if enough time has passed to log another value
        /// </summary>
        public void tick()
        {            
            while (curtime - prevtime < 1000f / freq)
            {
                curtime = sw.Elapsed.Ticks * 1000f / Stopwatch.Frequency;
            }
            prevtime = curtime;
        }

        /// <summary>
        /// main loop for logging thread.  waits till sample frequency specified delay before grabbing data and writing to file
        /// </summary>
        public void thread_loop()
        {
            while (true)
            {
                tick();
                lock (lockObj)  // lock safe data writing
                {
                    _data = get_data_f();
                    if (recordflag)
                    {
                        List<string> str_data = _data.Select(x => x.ToString(formatspec)).ToList();
                        write_data_with_timestamp(str_data);
                    }

                    if (historyflag)
                    {
                        for (int i = 0; i < signal_num; i++)
                        {
                            _data_history[i].Add(_data[i]);
                            _data_history[i].RemoveAt(0);
                        }
                    }
                }           
            }
        }

        public void close()
        {
            // aborts thread and deletes filewriter
            stop();
            close_file();
        }
    }

    /// <summary>
    /// class derived from regular dataLogger class to facilitate data collection for pattern recognition with ground truth labels
    /// </summary>
    public class PR_Logger : dataLogger
    {
        /// <summary>current ground truth output</summary>
        public int current_output;
        /// <summary>amount of time for each contraction in ms</summary
        public int contraction_time;
        /// <summary>amount of time for rest between contractions in ms</summary>
        public int relax_time;
        /// <summary>number of times to cycle through all output classest</summary>
        public int collection_cycles;
        /// <summary>number of times to cycle through all output classest</summary
        public int current_cycle;
        /// <summary>number of outputs to train</summary>
        public int train_output_num;
        /// <summary>time in ms to next contraction</summary>
        public int timetonext;
        /// <summary>flag indicating training is happening</summary>
        public bool trainFlag = false;
        /// <summary>flag indicating that participant should be contracting</summary>
        public bool contractFlag = false;
        /// <summary>list of output labels</summary>
        public List<string> output_labels;

        private long start_time;
        private Stopwatch PR_sw = new Stopwatch();

        public PR_Logger()
        {
            return;
        }

        /// <summary>
        /// writes data with timestamp and appended ground truth output class
        /// </summary>
        /// <param name="data">data to write as list of strings</param>
        public override void write_data_with_timestamp(List<string> data)
        {
            file.Write(curtime.ToString(formatspec) + ',');
            data.Add(current_output.ToString());
            write_csv(data);
        }

        /// <summary>
        /// sets the class labels to train with
        /// </summary>
        /// <param name="outputs_">list of class labels</param>
        public void set_outputs(List<string> outputs_)
        {
            output_labels = outputs_;
            train_output_num = output_labels.Count;
        }

        /// <summary>
        /// updates data collection variables
        /// </summary>
        public void PR_tick()
        {
            if (trainFlag)
            {
                long elapsed_time = PR_sw.ElapsedMilliseconds - start_time;
                recordflag = elapsed_time >= relax_time;    // turn recording on

                timetonext = (int)(relax_time - elapsed_time);

                if (elapsed_time >= relax_time + contraction_time)  // if single output relax and contract cycle is complete
                {
                    if (current_output < train_output_num-1)    // if there are more classes to train
                    {
                        current_output += 1;
                    }
                    else
                    {
                        if (current_cycle < collection_cycles-1)  // if there are additional cycles to complete
                        {
                            current_cycle += 1;
                            current_output = 0;
                        }
                        else
                        {
                            // if all cycles are complete
                            end_data_collection();
                        }
                    }
                    PR_sw.Restart();
                }
            }
        }

        /// <summary>
        /// initiates data collection sequence
        /// </summary>
        public void start_data_collection()
        {
            current_output = 0;
            current_cycle = 0;
            trainFlag = true;
            PR_sw.Restart();
            start_time = PR_sw.ElapsedMilliseconds;
        }

        /// <summary>
        /// ends data collection sequence
        /// </summary>
        public void end_data_collection()
        {
            close_file();
            recordflag = false;
            trainFlag = false;
        }
    }

    /// <summary>
    /// uses Newtonsoft.json to serialize and deserialize objects
    /// </summary>
    public static class ObjLogger
    {

        /// <summary>
        /// saves object to json file
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="filepath"></param>
        /// <param name="obj"></param>
        /// <returns></returns>
        public static bool saveObjJson<T>(string filepath, T obj)
        {
            if (!filepath.EndsWith(".json"))
            {
                filepath += ".json";
            }
            StreamWriter jsonWriter = new StreamWriter(filepath);
            string jsonString = JsonConvert.SerializeObject(obj, Formatting.Indented, new JsonSerializerSettings
            {
                ReferenceLoopHandling = ReferenceLoopHandling.Ignore,
                TypeNameHandling = TypeNameHandling.All
            });
            jsonWriter.Write(jsonString);
            jsonWriter.Flush();
            jsonWriter.Close();
            return true;
        }

        /// <summary>
        /// loads json file to object
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="filepath"></param>
        /// <returns></returns>
        public static T loadObjJson<T>(string filepath)
        {
            StreamReader jsonReader = new StreamReader(filepath);
            string jsonString = jsonReader.ReadToEnd();
            T obj = JsonConvert.DeserializeObject<T>(jsonString, new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All
            });

            jsonReader.Close();
            return obj;
        }

        public static dynamic loadObjJsonToDynamic(string filepath)
        {
            StreamReader jsonReader = new StreamReader(filepath);
            string jsonString = jsonReader.ReadToEnd();
            dynamic generic_obj = Newtonsoft.Json.Linq.JObject.Parse(jsonString);

            return generic_obj;
        }
    }  
}
