#ifndef NEURALNET_H
#define NEURALNET_H
#define ACTIVATION_RESPONSE 1.0
#include<string>
#include<vector>
#include<math.h>
#include<stdlib.h>
#include<iostream>
using namespace std;
class Data{
private:
    vector<vector<float> > SetIn;//记录训练用的数据
    vector<vector<float> > SetOut;//记录训练用数据的目标输出
    int NumPatterns;//数据量
    int InVectorSize;//数据大小
    int OutVectorSize;
public:
    Data(int invectorSize,int outvectorsize):InVectorSize(invectorSize),OutVectorSize(outvectorsize){

    }
    void AddData(vector<float> indata,vector<float> outdata){
        SetIn.push_back(indata);
        ++NumPatterns;
        SetOut.push_back(outdata);
    }
    vector<vector<float> > GetInputSet() {return SetIn;}
    vector<vector<float> > GetOutputSet(){return SetOut;}
};

struct Neuron{
    int NumInputs;//神经元的输入量
    vector<float> vecWeight;//权重
    float Activation;//这里是根据输入，通过某个线性函数确定，相当于神经元的输出
    float Error;//误差值
    float RandomClamped(){
        return -1+2*(rand()/((float)RAND_MAX+1));
    }
    Neuron(int inputs){
        NumInputs=inputs+1;
        Activation=0;
        Error=0;
        for(int i=0;i<NumInputs+1;i++)
            vecWeight.push_back(RandomClamped());//初始化权重
    }
};

struct NeuronLayer{
    int	NumNeurons;//每层拥有的神经元数
    vector<Neuron>	vecNeurons;
    NeuronLayer(int neurons, int perNeuron):NumNeurons(neurons){
        for(int i=0;i<NumNeurons;i++)
            vecNeurons.push_back(Neuron(perNeuron));
    }
};

typedef vector<float> iovector;
class NeuralNet{
private:
    int NumInputs;//输入量
    int NumOutputs;//输出量
    int NumHiddenLayers;//隐藏层数
    int NeuronsPerHiddenLayer;//隐藏层拥有的神经元
    float LearningRate;//学习率
    float ErrorSum;//误差总值
    bool Trained;//是否已经训练过
    int NumEpochs;//代数
    float ERROR_THRESHOLD;     //误差阈值（什么时候停止训练）
    long int Count;     //训练次数（什么时候停止训练）
    vector<NeuronLayer> vecLayers;//层数
    bool NetworkTrainingEpoch(vector<iovector > &SetIn,vector<iovector > &SetOut);//训练神经网络
    void CreateNet();//生成网络
    void InitializeNetwork();//初始化
    inline float Sigmoid(float activation, float response);
    float RandomClamped(){
        return -1+2*(rand()/((float)RAND_MAX+1));
    }
    bool Debug;//是否输出误差值
public:
    bool Train(Data* data);//开始训练
    enum STOPTYPE{COUNT,ERRORSUM}StopType;
    NeuralNet(int inputs,int outputs,int hiddenneurons,float learningRate,STOPTYPE type,bool debug=0);//初始化网络
    void SetErrorThrehold(float num){ERROR_THRESHOLD=num;}//设置误差
    void SetCount(long int num){Count=num;}//设置训练次数
    vector<float> Update(vector<float> inputs);//得到输出
    NeuralNet(string filename);//通过文件地址打开一个已经训练好的网络
    void saveNet(string filename);//保存已经训练的网络
};



#endif // NEURALNET_H
