#include"NeuralNet.h"
int main(){

    //训练数据
    vector<float> input1;
    input1.push_back(1.0);
    input1.push_back(1.0);
    vector<float> output1;
    output1.push_back(0.0);

    vector<float> input2;
    input2.push_back(1.0);
    input2.push_back(-1.0);
    vector<float> output2;
    output2.push_back(1.0);

    vector<float> input3;
    input3.push_back(-1.0);
    input3.push_back(-1.0);
    vector<float> output3;
    output3.push_back(0.0);

    vector<float> input4;
    input4.push_back(-1.0);
    input4.push_back(1.0);
    vector<float> output4;
    output4.push_back(1.0);

    //建立一个数据类
    Data* MyData=new Data(2,1);//2个输入，一个输出
    MyData->AddData(input1,output1);//添加数据
    MyData->AddData(input2,output2);
    MyData->AddData(input3,output3);
    MyData->AddData(input4,output4);

    NeuralNet* Brain=new NeuralNet(2,1,3,0.1,NeuralNet::ERRORSUM,true);//新建一个神经网络，输入神经元个数，输出神经元个数，隐藏层神经元个数，学习率，停止训练方法（次数或误差最小），是否输出误差值（用于观察是否收敛）
    Brain->SetErrorThrehold(0.01);//设置误差，默认0.01
    //Brain->SetCount(10000);设置次数，默认10000
    Brain->Train(MyData);//通过数据，开始训练
    cout<<Brain->Update(input1)[0]<<endl;//通过输入得到输出
    cout<<Brain->Update(input2)[0]<<endl;
    cout<<Brain->Update(input3)[0]<<endl;
    cout<<Brain->Update(input4)[0]<<endl;
    
    Brain->saveNet("D:\\1.txt");//保存网络
    
    NeuralNet* Brain2=new NeuralNet("D:\\1.txt");//通过文件读取网络
    cout<<Brain2->Update(input1)[0]<<endl;
    cout<<Brain2->Update(input2)[0]<<endl;
    cout<<Brain2->Update(input3)[0]<<endl;
    cout<<Brain2->Update(input4)[0]<<endl;

    return 0;
}
