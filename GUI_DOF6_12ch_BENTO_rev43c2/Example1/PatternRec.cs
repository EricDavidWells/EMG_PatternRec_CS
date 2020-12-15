using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using MathNet.Numerics.LinearAlgebra;


// useful: https://dotnetfiddle.net/

namespace brachIOplexus
{
    class PatternRec
    {
        int windowlength = 100;
        public dataLogger logger = new dataLogger();
        public int current_output = 0;
        public string[] output_labels;
        public int output_num;
        public int contraction_time = 1000;
        public int relax_time = 3000;
        public long start_time = 0;
        public int collection_cycles = 1;
        public int current_cycle = 0;

        public bool trainFlag = false;  // flag to indicate training has begun
        public bool contractFlag = false;

        public PatternRec()
        {
            //Matrix<float> A = Matrix<float>.Build.Random(3, 4);
            //A.SubMatrix
        }

        public float[] MAV(float[] data, float[] timestamps, int windowlength)
        {
            int datalen = data.Length;
            float[] MAVdata = data;

            return MAVdata;

        }

        public void LoadFile(string filepath)
        {

            string[] lines = System.IO.File.ReadAllLines(filepath);
            


            foreach (string line in lines)
            {

            }
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

        public void set_outputs(string[] outputs_)
        {
            output_labels = outputs_;
            output_num = outputs_.Length;
        }

        public void tick()
        {
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
                    logger.recordflag = false;
                    trainFlag = false;
                    logger.file.Flush();
                }
            }
        }

        public void start_train()
        {
            trainFlag = true;
            logger.tick();
            start_time = (long)logger.curtime;
        }

        public void end_train()
        {
            trainFlag = false;
        }
    }
}
