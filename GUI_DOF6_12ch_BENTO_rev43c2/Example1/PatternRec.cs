using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using MathNet.Numerics.LinearAlgebra;


// dependencies: Accord, Accord.MachineLearning

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


        public float[][] raw_data;
        public float[][] input_data;
        public int[] output_data;
        public int num_cols;
        public string[] output_labels;
        public string[] input_labels;
        public string[] raw_data_labels;
        public int output_num;
        public int input_num;

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

        public bool LoadFile(string filepath)
        {

            string[] lines = System.IO.File.ReadAllLines(filepath);
            int num_cols = lines[0].Split(',').Length - 1; // number of data columns in the file (-1 for line label)
            int num_lines = lines.Length - 1;    // number of lines in the data file (-1 for header)

            // initialize raw data jagged array
            raw_data = new float[num_cols][];
            for (int i = 0; i < num_cols; i++)
            {
                raw_data[i] = new float[num_lines];
            }

            // ensure header is first line in file
            string headerline = lines[0];
            if (headerline[0] != 'h')
            {
                return false;
            }

            // parse header line
            char[] chars_to_trim = { 'd', 'h', ',' };
            headerline = headerline.TrimStart(chars_to_trim);
            raw_data_labels = headerline.Split(',');

            // parse data lines
            for (int i=0; i<num_lines-1; i++)
            {
                string line = lines[i+1];   // skip first line
                
                // ensure all lines have data stamp as first index
                if (line[0] != 'd')
                {
                    return false;
                }

                line = line.TrimStart(chars_to_trim);
                string[] vals = line.Split(',');
                
                for (int j=0; j<num_cols; j++)
                {
                    raw_data[j][i] = float.Parse(vals[j]);
                }
                
            }

            // split up into input and output data arrays
            input_data = new float[num_cols-2][];
            for (int i=0; i < num_cols - 2; i++)
            {
                input_data[i] = new float[num_lines];
                input_data[i] = raw_data[i + 1];
            }

            output_data = new int[num_lines];
            for (int i=0; i < num_lines; i++)
            {
                output_data[i] = (int)raw_data[num_cols][i];
            }

            return true;
        }

        public void train_model()
        {
            //var learner = new LinearDualCoordinateDescent();
            //var teacher = new MulticlassSupportVectorMachine<Linear>(3, );

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
                    logger.close();
                }
            }
        }

        public void start_train()
        {
            trainFlag = true;
            logger.start();
            logger.tick();
            start_time = (long)logger.curtime;
            logger.recordflag = true;
        }

        public void end_train()
        {
            logger.close();
            trainFlag = false;
        }
    }
}
