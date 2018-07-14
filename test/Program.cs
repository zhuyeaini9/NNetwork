using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace test
{
    class Program
    {
        static void Main(string[] args)
        {
            Matrix<double> m = Matrix<double>.Build.Random(3, 4);
            Console.WriteLine(m);

            Random r = new Random();
            Console.WriteLine(r.NextDouble());
            Console.WriteLine(r.NextDouble());
            Console.WriteLine(r.NextDouble());
            Console.WriteLine(r.NextDouble());
            Console.WriteLine(r.NextDouble());
            Console.WriteLine(r.NextDouble());


            Matrix<double> m2 = Matrix<double>.Build.Dense(3, 4, (i, j) => r.NextDouble());
            Console.WriteLine(m2);


        }
    }
}
