#include <iostream>
#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "net.h"

#define THRESHOLD 70.36f
using namespace std;
#include "time.h"


double duration=.0;

//这个函数是官方提供的用于打印输出的tensor
void pretty_print(const ncnn::Mat& m)
{
    for (int q = 0; q < m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

void dot(float* a,float* b,float& result,int len){
    result=.0f;
    for(int i=0;i<len;i++){
        result+=a[i] * b[i] ;
    }
}
void clip(float &a){
    if(a<-1.0f){
        a=-1.0f;
    }else if(a>1.0f){
        a=1.0f;
    }
}
void get_output(cv::Mat &img,ncnn::Net &net,float* a,bool flip){    
    cv::Mat img2= img;
    if (flip)
    {
        cv::flip(img, img2, 1);
    }
    // 把opencv的mat转换成ncnn的mat
    ncnn::Mat input = ncnn::Mat::from_pixels(img2.data, ncnn::Mat::PIXEL_BGR, img2.cols, img2.rows);
    
    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
  
    input.substract_mean_normalize(mean_vals, norm_vals);  
    // pretty_print(input);

    // clock_t start,finish;
    // start=clock();
    auto t1 = std::chrono::high_resolution_clock::now();            
    // ncnn前向计算
    ncnn::Extractor extractor = net.create_extractor();
    extractor.input("input", input);
    ncnn::Mat output;
    extractor.extract("output", output);

    std::chrono::duration<double, std::milli> fp_ms;
    auto t2 = std::chrono::high_resolution_clock::now();
    fp_ms = t2 - t1;
    // auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(fp_ms);    
    duration+=fp_ms.count();

    // finish=clock();
    // duration+=(double)(finish-start);
    float value=.0f;
    const float* ptr = output.channel(0);
    // float a[128];
    for (int y = 0; y < output.h; y++)
    {
        for (int x = 0; x < output.w; x++)
        {
            a[x]=ptr[x];
        }
        ptr += output.w;
    }       
}

void get_feature(string img_path,ncnn::Net &net,float* a){
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat img2;
    int input_width = 112;//转onnx时指定的输入大小
    int input_height = 112;
    // resize
    cv::resize(img, img2, cv::Size(input_width, input_height));
    
    float x1[128],x2[128];
    get_output(img2,net,x1,false);
    get_output(img2,net,x2,true);

    float value=.0f;
    for (int x = 0; x < 128; x++)
    {
        a[x]=x1[x]+x2[x];
        value+=a[x]*a[x];
    } 
    value = sqrt(value);
    for (int x = 0; x < 128; x++) {
        a[x]/=value;
    }
}
bool compare(string &path1,string &path2,ncnn::Net &net,int &answer){
    float img1[128];
    float img2[128];
    get_feature(path1,net,img1);
    get_feature(path2,net,img2);
    float result;
    dot(img1,img2,result,128);
    clip(result);    
    result = acos(result) * 180.0f / 3.1415926f;
    if(answer==1){
        if(result>THRESHOLD)
            return false;
    }else{
        if(result<=THRESHOLD)
            return false;        
    }
    return true;
}

//main函数模板
int main() {
    // 加载转换并且量化后的alexnet网络
    ncnn::Net net;
    net.opt.num_threads=4;
    net.load_param("my_mobileface-sim-opt.param");
    net.load_model("my_mobileface-sim-opt.bin");
    // cout<<CLOCKS_PER_SEC<<endl;

    string input_file="lfw_test_pair_mini.txt";
    string query;
    ifstream in(input_file);
    string info[3];
    int i=0;
    int sum=0;
    int wrong=0;
    while(in>>query){
        info[i++]=query;
        if(i==3){
            i=0;
            sum++;
            int answer;
            bool result;
            sscanf(info[2].c_str(), "%d", &answer);
            info[0]="./aligned_imgs/"+info[0];
            info[1]="./aligned_imgs/"+info[1];
            result = compare(info[0],info[1],net,answer);
            if(!result){
                wrong++;
            }
        }
    }   
    in.close();

    cout<<"sum "<<sum<<endl;
    cout<<"wrong "<<wrong<<endl; 
    cout<<"duration "<<duration<<endl;        
    return 0;
}
