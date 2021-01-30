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
    public delegate List<T> get_data_func<T>();

    public class dataLogger
    {
        public StreamWriter file;
        public bool recordflag = false;
        public bool historyflag = false;
        public Stopwatch sw = new Stopwatch();
        public int freq;
        public float prevtime;
        public float curtime;
        public string filepath;
        Thread t;

        public int signal_num;
        public int history_num;
        public List<double> data;
        public get_data_func<double> get_data;
        public List<List<double>> data_history = new List<List<double>>();

        public dataLogger()
        {
            // creates dataLogger object
        }

        public void init_file(string filepath_)
        {
            // initiates filewriter
            filepath = filepath_;
            file = new StreamWriter(filepath);
        }

        public void close_file()
        {
            if (file != null)
            {
                file.Flush();
                file.Dispose();
                file = null;
            }
        }

        public void start()
        {
            // starts thread and initiates timing variables
            sw.Start();
            prevtime = sw.ElapsedMilliseconds;
            curtime = prevtime;

            if (historyflag)
            {

                data_history.Clear();
                for (int i=0; i<signal_num; i++)
                {
                    List<double> temp_list = new List<double>();
                    for (int j=0; j<history_num; j++)
                    {
                        temp_list.Add(0);
                    }
                    data_history.Add(temp_list);
                }
            }

            if (t == null)
            {
                t = new Thread(thread_loop);
                t.Start();
                //t.Priority = ThreadPriority.BelowNormal;
            }
        }

        public void stop()
        {
            // stops thread and flushes file without deleting filewriter
            if (t != null)
            {
                sw.Stop();
                t.Abort();
                t = null;
            }
        }

        protected void write_csv(List<string> data)
        {
            // writes data to file in csv format
            string newLine = "";
            for (int i=0; i<data.Count; i++)
            {
                newLine += data[i] + ",";
            }
            file.WriteLine(newLine.TrimEnd(','));
        }

        public virtual void write_header(List<string> data)
        {
            // sends data to write_csv with a prepended "h" character to indicate header line
            data.Insert(0, "h");
            write_csv(data);
        }

        public virtual void write_data_with_timestamp(List<string> data)
        {
            //sends data to write_csv with a prepended "d" character to indicate data line

            data.Insert(0, "d");
            data.Insert(1, curtime.ToString("F3"));
            write_csv(data);
        }

        public void tick()
        {
            // updates stopwatch and flips timeflag if enough time has passed to log another value
            
            while (curtime - prevtime < 1000f / freq)
            {
                curtime = sw.Elapsed.Ticks * 1000f / Stopwatch.Frequency;
            }
            prevtime = curtime;
        }

        public void thread_loop()
        {
            while (true)
            {
                tick();
                data = get_data();
                if (recordflag)
                {
                    List<string> str_data = data.Select(x => x.ToString()).ToList();
                    write_data_with_timestamp(str_data);
                }

                if (historyflag)
                {
                    for (int i=0; i<signal_num; i++)
                    {
                        data_history[i].Add(data[i]);
                        data_history[i].RemoveAt(0);
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

    public class PR_Logger : dataLogger
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

        //public override void write_header(List<string> data)
        //{
        //    data.Insert(0, "h");
        //    data.Add("output");
        //    write_csv(data);
        //}

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

                current_output = segment_number % train_output_num;

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

    public static class ObjLogger
    {
        public static bool saveObjJson<T>(string filepath, T obj)
        {
            if (!filepath.EndsWith(".json"))
            {
                filepath += ".json";
            }
            StreamWriter jsonWriter = new StreamWriter(filepath);
            string jsonString = JsonConvert.SerializeObject(obj, Formatting.Indented, new JsonSerializerSettings
            {
                ReferenceLoopHandling = ReferenceLoopHandling.Ignore
            });
            jsonWriter.Write(jsonString);
            jsonWriter.Flush();
            jsonWriter.Close();
            return true;
        }

        public static T loadObjJson<T>(string filepath)
        {
            StreamReader jsonReader = new StreamReader(filepath);
            string jsonString = jsonReader.ReadToEnd();
            T obj = JsonConvert.DeserializeObject<T>(jsonString);
            return obj;
        }
    }  
}
