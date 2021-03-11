using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using RealTimePatternRec;
using RealTimePatternRec.PatternRec;
using RealTimePatternRec.DataLogging;

namespace ExampleUse
{
    class Program
    {
        static void Main(string[] args)
        {
            //Console.WriteLine("Testing ONNX model");
            //Tests.ONNXTest();

            //Console.WriteLine("Testing Accord Model");
            //Tests.AccordSVMTest();

            //Console.WriteLine("Testing loading of ONNX model settings");
            //Tests.ONNXSettingsTest();

            //Console.WriteLine("Running Mapper Example");
            //Tests.MapperBasicUsage();

            //Console.WriteLine("Testing Data Logger Basic Usage");
            //Tests.DataLoggerBasicUsage();

            //Console.WriteLine("Running PR_Logger Example");
            //Tests.PR_LoggerBasicUsage();

            //Console.WriteLine("Running Model Example");
            //Tests.ModelBasicUsage();

            Console.WriteLine("Testing Filters");
            Tests.FilterTesting();

            Console.Read();
        }
    }
}
