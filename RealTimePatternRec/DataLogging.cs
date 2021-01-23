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
        private bool timeflag = false;
        public Stopwatch sw = new Stopwatch();
        public int samplefreq;
        public float prevtime;
        public float curtime;
        public string filepath;
        Thread t;

        public List<double> data_to_write;
        public get_data_func<double> get_data;

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

            if (t == null)
            {
                t = new Thread(thread_loop);
                t.Start();
                t.Priority = ThreadPriority.BelowNormal;
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
            
            while (curtime - prevtime < 1000f / samplefreq)
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
                if (recordflag)
                {
                    data_to_write = get_data();
                    List<string> str_data = data_to_write.Select(x => x.ToString()).ToList();
                    write_data_with_timestamp(str_data);
                }
            }
        }

        public bool saveModelJson<T>(string filepath, T obj)
        {
            if (!filepath.EndsWith(".json"))
            {
                filepath += ".json";
            }
            StreamWriter jsonWriter = new StreamWriter(filepath);
            string jsonString = JsonConvert.SerializeObject(obj);
            jsonWriter.Write(jsonString);
            jsonWriter.Flush();
            jsonWriter.Close();

            return true;
        }

        public bool loadModelJson<T>(string filepath, ref T obj)
        {
            StreamReader jsonReader = new StreamReader(filepath);
            string jsonString = jsonReader.ReadToEnd();
            obj = JsonConvert.DeserializeObject<T>(jsonString);
            return true;
        }

        public void close()
        {
            // aborts thread and deletes filewriter
            stop();
            close_file();

        }
    }
}
