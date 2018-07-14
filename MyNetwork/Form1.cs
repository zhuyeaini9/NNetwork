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

    public class CNode
    {
        public double mWX;
        public double mWXActived;
        public List<double> mWParam = new List<double>();
        public double mBParam;
        public CNode(int inputNum)
        {
            Random r = new Random();
            for (int i = 0; i < inputNum; i++)
            {
                mWParam.Add(r.NextDouble());
            }
            mBParam = r.NextDouble();
        }
    }
    public class CNNetwork
    {
        public Dictionary<int, List<CNode>> mLayerNodes = new Dictionary<int, List<CNode>>();
        public CNNetwork(int inputNum,int layerNum,params int[] layerNodeNum)
        {
            if (layerNum < 2)
                throw new Exception("layerNum must >=2");
            if (layerNum != layerNodeNum.Length)
                throw new Exception("layerNum must == layerNodeNum.Length");

            for(int i=0;i<layerNum;i++)
            {
                List<CNode> ns = new List<CNode>();
                mLayerNodes[i] = ns;
                int iNum = 0;
                if (i == 0)
                    iNum = inputNum;
                else
                    iNum = layerNodeNum[i-1];
                for (int j = 0; j < layerNodeNum[i]; j++)
                {
                    ns.Add(new CNode(iNum));
                }
            }
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
                    if (z > 0)
                        re = z;
                    else
                        re = 0;
                }
                if(af == ACTIVE_FUNCTION.LOGI)
                {
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
        }
    }
    public class CNNetwork2
    {
        public CNNetwork2(int inputNum, int layerNum, params int[] layerNodeNum)
        {
            if (inputNum < 1)
                throw new Exception("inputNum must >=2");
            if (layerNum < 2)
                throw new Exception("layerNum must >=2");
            if (layerNum != layerNodeNum.Length)
                throw new Exception("layerNum must == layerNodeNum.Length");

            
        }
    }
}
