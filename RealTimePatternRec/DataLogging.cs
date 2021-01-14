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
    class dataLogger
    {
        public StreamWriter file;
        public bool recordflag = false;
        private bool timeflag = false;
        public Stopwatch sw = new Stopwatch();
        public int samplefreq = 50;
        public float prevtime;
        public float curtime;
        public string filepath;
        Thread t;
        public string[] data_to_write;

        public dataLogger()
        {
            // creates dataLogger object
        }

        public void init_logger(string filepath_)
        {
            // initiates filewriter
            filepath = filepath_;
            file = new StreamWriter(filepath);
        }

        public void start()
        {
            // starts thread and initiates timing variables
            sw.Start();
            prevtime = sw.ElapsedMilliseconds;
            curtime = prevtime;

            t = new Thread(thread_loop);
            t.Start();
            t.Priority = ThreadPriority.BelowNormal;
        }

        public void stop()
        {
            // stops thread and flushes file without deleting filewriter
            sw.Stop();
            t.Abort();
            file.Flush();
        }

        private void write_csv(string[] data)
        {
            // writes data to file in csv format
            string newLine = "";
            foreach (string s in data)
            {
                newLine += s + ",";
            }
            file.WriteLine(newLine.TrimEnd(','));
        }

        public void write_header(string[] data)
        {
            // sends data to write_csv with a prepended "h" character to indicate header line

            string[] temp = new string[data.Length + 1];
            temp[0] = "h";
            for (int i=0; i<data.Length; i++)
            {
                temp[i + 1] = data[i];
            }
            write_csv(temp);
        }

        public void write_data_with_timestamp(string[] data)
        {
            //sends data to write_csv with a prepended "d" character to indicate data line
           
            string[] temp = new string[data.Length + 2];
            temp[0] = "d";
            temp[1] = curtime.ToString("F3");
            for (int i = 0; i < data.Length; i++)
            {
                temp[i + 2] = data[i];
            }
            write_csv(temp);
        }

        public void tick()
        {
            // updates stopwatch and flips timeflag if enough time has passed to log another value
            
            curtime = sw.Elapsed.Ticks*1000f / Stopwatch.Frequency;
            if (curtime - prevtime > 1000f / samplefreq)
            {
                timeflag = true;
                prevtime = curtime;
            }
            else
            {
                timeflag = false;
            }
        }

        public void thread_loop()
        {
            // data recording loop that logger thread sits in until aborted
            // records data when recordflag == true and timeflag == true
            // if recordflag != true sleeps thread in 100ms increments to not waste resources
            while (true)
            {
                if (recordflag)
                {
                    tick();
                    if (timeflag)
                    {
                        write_data_with_timestamp(data_to_write);
                    }

                Thread.Sleep(0);
                }
                else
                {
                    Thread.Sleep(100);
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
            file.Close();
            file = null;
        }
    }
}
