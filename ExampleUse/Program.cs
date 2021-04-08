using System;

using RealTimePatternRec.Examples;


namespace ExampleUse
{
    class Program
    {
        static void Main(string[] args)
        {
            //Console.WriteLine("Running ONNX model Example");
            //Examples.ONNXBasicUsage();

            //Console.WriteLine("Running Accord Model Example");
            //Examples.AccordSVMTest();

            //Console.WriteLine("Running loading of ONNX model settings Example");
            //Examples.ONNXFromSettingsBasicUsage();

            //Console.WriteLine("Running Mapper Example");
            //Examples.MapperBasicUsage();

            //Console.WriteLine("Running Data Logger Basic Usage Example");
            //Examples.DataLoggerBasicUsage();

            //Console.WriteLine("Running PR_Logger Example");
            //Examples.PR_LoggerBasicUsage();

            //Console.WriteLine("Running Model Example");
            //Examples.ModelBasicUsage();

            //Console.WriteLine("Running Filters Example");
            //Examples.FilterTesting();

            Console.WriteLine("Running MultiDimensional Filter Example");
            Examples.MultidimensionalFilterTest();

            Console.WriteLine("Examples Complete");
            Console.Read();
        }
    }
}
