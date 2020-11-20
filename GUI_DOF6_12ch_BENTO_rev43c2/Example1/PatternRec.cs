using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace brachIOplexus
{
    class PatternRec
    {
        int windowlength = 100;

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
    }
}
