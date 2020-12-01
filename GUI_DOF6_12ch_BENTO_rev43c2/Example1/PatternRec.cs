using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;                // For save/open parameters functionality
using MathNet.Numerics.LinearAlgebra;

namespace brachIOplexus
{
    class PatternRec
    {
        int windowlength = 100;
        int current_output_label = 0;
        public dataLogger logger = new dataLogger();

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
    }
}
