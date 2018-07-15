using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MyNetwork
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
    }

    public class CHelp
    {
        public static double getActiveVal(double z,ACTIVE_FUNCTION af)
        {
            double re = 0.1;
            try
            {
                if(af == ACTIVE_FUNCTION.RELU)
                {
                    /*
                    a = g(z) = max(0,z)
                    dz = 0/1
                    */
                    if (z > 0)
                        re = z;
                    else
                        re = 0;
                }
                if(af == ACTIVE_FUNCTION.LOGI)
                {
                    /*
                    a = g(z) = 1/(1+e(-z))
                    dz = a(1-a)
                    */
                    re = 1 / (1 + Math.Exp(-z));
                }
            }
            catch(Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }
            return re;
        }
        public static Matrix<double> getActiveValMatrix(Matrix<double> m,ACTIVE_FUNCTION af)
        {
            Matrix<double> re = Matrix<double>.Build.DenseIdentity(m.RowCount,m.ColumnCount);
            try
            {
                re = Matrix<double>.Build.Dense(m.RowCount, m.ColumnCount, (i, j) => getActiveVal(m.Row(i)[j], af));
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }
            return re;
        }
    }

    public enum ACTIVE_FUNCTION
    {
        LOGI,//0~1
        TAN,//-1~1
        RELU//max(0,z)
    }
    public class CNNLayer
    {
        public ACTIVE_FUNCTION mAFuction;
        public int mNodeNum;
        public int mInputNum;
        public Matrix<double> mA;
        public Matrix<double> mZ;
        public Matrix<double> mW;
        public Matrix<double> mB;
        public CNNLayer(int nodeNum,ACTIVE_FUNCTION af = ACTIVE_FUNCTION.RELU)
        {
            mAFuction = af;
            mNodeNum = nodeNum;
            Random r = new Random();
            mB = Matrix<double>.Build.Dense(nodeNum, 1, (i, j) => r.NextDouble());
            mZ = Matrix<double>.Build.Dense(nodeNum, 1, (i, j) => r.NextDouble());
            mA = Matrix<double>.Build.Dense(nodeNum, 1, (i, j) => r.NextDouble());
        }
        public void init(int inputNum)
        {
            mInputNum = inputNum;
            Random r = new Random();
            mW = Matrix<double>.Build.Dense(mNodeNum, inputNum, (i, j) => r.NextDouble());
        }
        public void cal(Matrix<double> input)
        {
            if (input == null || input.RowCount != mInputNum)
                throw new Exception("input count != layer input count");

            mZ = mW * input + mB;
            mA = CHelp.getActiveValMatrix(mZ, mAFuction);
        }
    }
    public class CInputOutputSample
    {
        public Matrix<double> mIn;
        public Matrix<double> mOut;
        CInputOutputSample(Matrix<double> i,Matrix<double> j)
        {
            if (i.ColumnCount != j.ColumnCount)
                throw new Exception("输入输出样本数量不匹配");
            mIn = i;
            mOut = j;
        }
        public int getInputNum()
        {
            return mIn.RowCount;
        }
        public int getOutputNum()
        {
            return mOut.RowCount;
        }
        public int getSampleCount()
        {
            return mIn.ColumnCount;
        }
        public Vector<double> getInSample(int index)
        {
            return mIn.Column(index);
        }
        public Vector<double> getOutSample(int index)
        {
            return mOut.Column(index);
        }
    }
    public class CNNetwork
    {
        /*
        L(a,y)=-(y*log(a)+(1-y)*log(1-a))
        da = -y/a+(1-y)/(1-a)
        */
        public double mCurDeviation = double.MaxValue;
        public CInputOutputSample mInputOutputSample;
        public List<CNNLayer> mNNlayers = new List<CNNLayer>();
        public CNNetwork(CInputOutputSample samples, params CNNLayer[] layer)
        {
            mInputOutputSample = samples;
            foreach (var v in layer)
            {
                mNNlayers.Add(v);
            }
            for(int i=0;i<mNNlayers.Count;i++)
            {
                if (i == 0)
                    mNNlayers[i].init(samples.getInputNum());
                mNNlayers[i].init(mNNlayers[i - 1].mNodeNum);
            }
        }
        void cal(Matrix<double> input)
        {
            if(mNNlayers.Count>0)
                _cal(input, 0);
        }
        void _cal(Matrix<double> i,int layIndex)
        {
            if (layIndex >= mNNlayers.Count)
                return;
            CNNLayer lay = mNNlayers[layIndex];
            lay.cal(i);
            layIndex++;
            _cal(lay.mA, layIndex);
        }
        bool checkDeviation()
        {
            bool re = true;
            mNNlayers[mNNlayers.Count-1].mA
            return re;
        }
        void update()
        {
            for(int i=0;i<mInputOutputSample.getSampleCount();i++)
            {
                Vector<double> inSample = mInputOutputSample.getInSample(i);
                Vector<double> outSample = mInputOutputSample.getOutSample(i);
                Matrix<double> inMatrix = Matrix<double>.Build.DenseOfColumnVectors(inSample);
                Matrix<double> outMatrix = Matrix<double>.Build.DenseOfColumnVectors(outSample);
                //计算正向误差
                cal(inMatrix);

                //计算反向更新参数

                //更新误差是否更小
            }           
        }
    }
}
