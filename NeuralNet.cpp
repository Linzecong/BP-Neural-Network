#include "NeuralNet.h"
bool NeuralNet::NetworkTrainingEpoch(vector<iovector> &SetIn, vector<iovector> &SetOut){
    vector<float>::iterator curWeight;//指向某个权重
    vector<Neuron>::iterator curNrnOut,curNrnHid;//指向输出神经元和某个隐藏神经元
    ErrorSum=0;//置零
    //对每一个输入集合调整权值
    for(unsigned int vec=0;vec<SetIn.size();vec++){
        vector<float> outputs=Update(SetIn[vec]);//通过神经网络获得输出
        //根据每一个输出神经元的输出调整权值
        for(int op=0;op<NumOutputs;op++){
            float err=(SetOut[vec][op]-outputs[op])*outputs[op]*(1-outputs[op]);//误差的平方
            ErrorSum+=(SetOut[vec][op]-outputs[op])*(SetOut[vec][op]-outputs[op]);//计算误差总和，用于暂停训练
            vecLayers[1].vecNeurons[op].Error=err;//更新误差（输出层）
            curWeight=vecLayers[1].vecNeurons[op].vecWeight.begin();//标记第一个权重
            curNrnHid=vecLayers[0].vecNeurons.begin();//标记隐藏层第一个神经元
            //对该神经元的每一个权重进行调整
            while(curWeight!=vecLayers[1].vecNeurons[op].vecWeight.end()-1){
                *curWeight+=err*LearningRate*curNrnHid->Activation;//根据误差和学习率和阈值调整权重
                curWeight++;//指向下一个权重
                curNrnHid++;//指向下一个隐藏层神经元
            }
            *curWeight+=err*LearningRate*(-1);//偏移值
        }
        curNrnHid=vecLayers[0].vecNeurons.begin();//重新指向隐藏层第一个神经元
        int n=0;
        //对每一个隐藏层神经元
        while(curNrnHid!=vecLayers[0].vecNeurons.end()-1){
            float err=0;
            curNrnOut=vecLayers[1].vecNeurons.begin();//指向第一个输出神经元
            //对每一个输出神经元的第一个权重
            while(curNrnOut!=vecLayers[1].vecNeurons.end()){
                err+=curNrnOut->Error*curNrnOut->vecWeight[n];//某种计算误差的方法(BP)
                curNrnOut++;
            }
            err*=curNrnHid->Activation*(1-curNrnHid->Activation);//某种计算误差的方法(BP)
            for(int w=0;w<NumInputs;w++)
                curNrnHid->vecWeight[w]+=err*LearningRate*SetIn[vec][w];//根据误差更新隐藏层的权重
            curNrnHid->vecWeight[NumInputs]+=err*LearningRate*(-1);//偏移值
            curNrnHid++;//下一个隐藏层神经元
            n++;//下一个权重
        }
    }
    return 1;
}

void NeuralNet::CreateNet(){
    if(NumHiddenLayers>0){
        vecLayers.push_back(NeuronLayer(NeuronsPerHiddenLayer,NumInputs));
        for(int i=0;i<NumHiddenLayers-1;++i)
            vecLayers.push_back(NeuronLayer(NeuronsPerHiddenLayer,NeuronsPerHiddenLayer));
        vecLayers.push_back(NeuronLayer(NumOutputs,NeuronsPerHiddenLayer));
    }
    else{
        vecLayers.push_back(NeuronLayer(NumOutputs,NumInputs));
    }
}

void NeuralNet::InitializeNetwork(){
    for(int i=0;i<NumHiddenLayers+1;++i)
        for(int n=0;n<vecLayers[i].NumNeurons;++n)
            for(int k=0;k<vecLayers[i].vecNeurons[n].NumInputs;++k)
                vecLayers[i].vecNeurons[n].vecWeight[k]=RandomClamped();//随机生成权重
    ErrorSum=9999;
    NumEpochs=0;
}

float NeuralNet::Sigmoid(float activation, float response){
    return ( 1 / ( 1 + exp(-activation / response)));
}

NeuralNet::NeuralNet(int inputs, int outputs, int hiddenneurons, float learningRate, STOPTYPE type, bool debug):
    NumInputs(inputs),
    NumOutputs(outputs),
    NumHiddenLayers(1),
    NeuronsPerHiddenLayer(hiddenneurons),
    LearningRate(learningRate),
    ERROR_THRESHOLD(0.01),
    Count(10000),
    StopType(type),
    Debug(debug),
    ErrorSum(9999),
    Trained(false),
    NumEpochs(0){
    CreateNet();
}

vector<float> NeuralNet::Update(vector<float> inputs){
    vector<float> outputs;
    int cWeight = 0;
    if (inputs.size()!=NumInputs)
        return outputs;
    for(int i=0;i<NumHiddenLayers+1;++i){
        if(i>0)
            inputs=outputs;
        outputs.clear();
        cWeight = 0;
        for(int n=0;n<vecLayers[i].NumNeurons;++n){
            float netinput=0;
            int	numInputs=vecLayers[i].vecNeurons[n].NumInputs;
            for (int k=0;k<numInputs-1;++k)
                netinput+=vecLayers[i].vecNeurons[n].vecWeight[k]*inputs[cWeight++];
            netinput+=vecLayers[i].vecNeurons[n].vecWeight[numInputs-1]*(-1);
            vecLayers[i].vecNeurons[n].Activation=Sigmoid(netinput,ACTIVATION_RESPONSE);
            outputs.push_back(vecLayers[i].vecNeurons[n].Activation);//即输出
            cWeight = 0;
        }
    }
    return outputs;
}

NeuralNet::NeuralNet(string filename){
    FILE *sourcefile=fopen(filename.c_str(),"rb");
    fread(&this->NumInputs,sizeof(int),1,sourcefile);
    fread(&this->NumOutputs,sizeof(int),1,sourcefile);
    fread(&this->NumHiddenLayers,sizeof(int),1,sourcefile);
    fread(&this->NeuronsPerHiddenLayer,sizeof(int),1,sourcefile);
    fread(&this->LearningRate,sizeof(float),1,sourcefile);
    fread(&this->ErrorSum,sizeof(float),1,sourcefile);
    fread(&this->Trained,sizeof(bool),1,sourcefile);
    fread(&this->NumEpochs,sizeof(int),1,sourcefile);
    fread(&this->ERROR_THRESHOLD,sizeof(float),1,sourcefile);
    fread(&this->Count,sizeof(long int),1,sourcefile);
    this->CreateNet();
    for(int i=0;i<this->vecLayers.size();i++){
        fread(&this->vecLayers[i].NumNeurons,sizeof(int),1,sourcefile);
        for(int j=0;j<this->vecLayers[i].vecNeurons.size();j++){
            fread(&this->vecLayers[i].vecNeurons[j].NumInputs,sizeof(int),1,sourcefile);
            fread(&this->vecLayers[i].vecNeurons[j].Activation,sizeof(float),1,sourcefile);
            fread(&this->vecLayers[i].vecNeurons[j].Error,sizeof(float),1,sourcefile);
            for(int k=0;k<this->vecLayers[i].vecNeurons[j].vecWeight.size();k++)
                fread(&this->vecLayers[i].vecNeurons[j].vecWeight[k],sizeof(float),1,sourcefile);
        }
    }
    fclose(sourcefile);
}

void NeuralNet::saveNet(string filename){
    FILE *sourcefile=fopen(filename.c_str(),"wb");
    fwrite(&this->NumInputs,sizeof(int),1,sourcefile);
    fwrite(&this->NumOutputs,sizeof(int),1,sourcefile);
    fwrite(&this->NumHiddenLayers,sizeof(int),1,sourcefile);
    fwrite(&this->NeuronsPerHiddenLayer,sizeof(int),1,sourcefile);
    fwrite(&this->LearningRate,sizeof(float),1,sourcefile);
    fwrite(&this->ErrorSum,sizeof(float),1,sourcefile);
    fwrite(&this->Trained,sizeof(bool),1,sourcefile);
    fwrite(&this->NumEpochs,sizeof(int),1,sourcefile);
    fwrite(&this->ERROR_THRESHOLD,sizeof(float),1,sourcefile);
    fwrite(&this->Count,sizeof(long int),1,sourcefile);
    for(int i=0;i<this->vecLayers.size();i++){
        fwrite(&this->vecLayers[i].NumNeurons,sizeof(int),1,sourcefile);
        for(int j=0;j<this->vecLayers[i].vecNeurons.size();j++){
            fwrite(&this->vecLayers[i].vecNeurons[j].NumInputs,sizeof(int),1,sourcefile);
            fwrite(&this->vecLayers[i].vecNeurons[j].Activation,sizeof(float),1,sourcefile);
            fwrite(&this->vecLayers[i].vecNeurons[j].Error,sizeof(float),1,sourcefile);
            for(int k=0;k<this->vecLayers[i].vecNeurons[j].vecWeight.size();k++)
                fwrite(&this->vecLayers[i].vecNeurons[j].vecWeight[k],sizeof(float),1,sourcefile);
        }
    }
    fclose(sourcefile);
}

bool NeuralNet::Train(Data *data){
    cout<<"Training..."<<endl;
    if(Trained==1)
        return false;
    vector<vector<float> > SetIn=data->GetInputSet();
    vector<vector<float> > SetOut=data->GetOutputSet();
    InitializeNetwork();
    if(StopType==COUNT){
        long int i=Count;
        while(i--){
            if(Debug)
                cout<<"ErrorSum:"<<ErrorSum<<endl;
            NetworkTrainingEpoch(SetIn,SetOut);
        }
    }
    else{
        while(ErrorSum>ERROR_THRESHOLD){
            if(Debug)
                cout<<"ErrorSum:"<<ErrorSum<<endl;
            NetworkTrainingEpoch(SetIn,SetOut);
        }
    }
    Trained=true;
    cout<<"Done!!!"<<endl;
    return true;
}
