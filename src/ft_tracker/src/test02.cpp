#include<iostream>
#include"ros/ros.h"
#include"std_msgs/String.h"

#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
#include <image_transport/image_transport.h>

#include"sensor_msgs/CameraInfo.h"
#include <sensor_msgs/Image.h>

#include"cv_bridge/cv_bridge.h"

#include <mutex>//明天加锁头

#include <vector>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include "ft_detector/Detection.h"


class Tracker{
    private:
    
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    int state_size = 6;//[x,y,z,vx,vy,vz]
    int measurement_size = 3;
    int control_size = 0;

    cv::Mat A;
    cv::Mat H;
    double Q;
    double R;
    double P;

    cv::Mat K;
    cv::Mat D;
    cv::Mat img;

    bool camerainfo = false;

    ros::Subscriber info_sub;
    std::string info_topic;

    ros::Publisher ft_pub;
    std::string ft_P_topic;

    image_transport::Subscriber ft_sub;
    std::string ft_S_topic;

    //锁头
    std::mutex info_mutex_;

    int queue_size = 1;

    sensor_msgs::CameraInfo::ConstPtr latest_info_;
    ros::Time last_stamp;
    double last_time_,now_time;
    std::string frame_id //用yaml定住一下
    cv::Point3f pos;

    cv::KalmanFilter kf;

    double T;//飞行时间
    double dt = 0.03;
    std::vector<geometry_msgs::Point> measurements;
    std::vector<geometry_msgs::Point> predictions;
    bool first = true;
    bool tracking;

    void CameraInfoCallback(const sensor_msgs::CameraInfo& msg);
    void ImageCallback(const sensor_msgs::ImageConstPtr& msg);
    cv::Point3f Tracker::KalmanFilter(cv::KalmanFilter& kf,cv::Point3f& pos,double dt);



    public:
    Tracker(ros::NodeHandle& nh): nh_("~"),it_(nh_)
    {
    nh_.param<std::string>("info_topic",info_topic,"/hk_camera/camera_info");
    nh_.param<std::string>("ft_pub",ft_P_topic,"/ft/track");
    nh_.param<std::string>("ft_sub",ft_S_topic,"/ft/detection");

    info_sub = nh_.subscribe(info_topic,queue_size,&Tracker::CameraInfoCallback,this);
    ft_sub = it_.subscribe(ft_P_topic,queue_size,&Tracker::ImageCallback,this);

    ft_pub = nh_.advertise<std_msgs::String>(ft_P_topic,queue_size);

    //卡尔曼
    kf.init(state_size,measurement_size,control_size);
    cv::namedWindow("tracking",cv::WINDOW_AUTOSIZE);

    //状态转移矩阵
    A = cv::Mat::eye(state_size,state_size,CV_32F);
    A.at<float>(0,3) = 1;//x = x+dx;
    A.at<float>(1,4) = 1;//y = y+dy;
    A.at<float>(2,5) = 1;//z = z+dz
    kf.transitionMatrix = A;
    
    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov,cv::Scalar::all(Q));//测量
    cv::setIdentity(kf.measurementNoiseCov,cv::Scalar::all(R));//传感器
    cv::setIdentity(kf.errorCovPost,cv::Scalar::all(P));//后验误差（rows，cols）
    /**
     * 轨迹滞后‌ ：(太信模型，不信观测)
     * 轨迹抖动‌ ：(太信观测，不信模型)
     * 发散/飞掉‌：P 初始值不当 或 Q/R 极端不匹配
     * **/
}
};


//拿
void Tracker::msgcallback(const ft_detector::Detection::ConstPtr& msg){
    pos.x = msg -> x;
    pos.y = msg -> y;
    pos.z = msg -> z;

    last_stamp = msg ->header.stamp.toSec;
    frame_id = msg -> header.frame_id;
    measurements.push_back(pos);

    visualization_msgs::Marker line_measurements_Marker;

    line_measurements_Marker.header.stamp = msg->header.stamp;
    line_measurements_Marker.header.frame_id = msg->header.frame_id;
    line_measurements_Marker.ns = "measurements";
    line_measurements_Marker.id = 2;
    line_measurements_Marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_measurements_Marker.action = visualization_msgs::Marker::ADD;

    line_measurements_Marker.scale.x = 0.02 //线粗细，yaml
    line_measurements_Marker.color.r = 0.0;
    line_measurements_Marker.color.g = 1.0;
    line_measurements_Marker.color.b = 0.0;
    line_measurements_Marker.color.a = 1.0;//不透明度

    line_measurements_Marker.points = measurements;
    ft_pub.publish(line_measurements_Marker);
}



//相机
void Tracker::CameraInfoCallback(const sensor_msgs::CameraInfo& msg){
    K = cv::Mat(3,3,CV_64F,(void*)msg.K.data()).clone();
    D = cv::Mat(msg.D.size(),1,CV_64F,(void*)msg.D.data()).clone();

    camerainfo = true;
}


//死机
void stop(double last_time_,bool first){
    last_time_ = 0.0;
    first = true;
    measurements.clear();
    predictions.clear();
}


//画面
void Tracker::ImageCallback(const sensor_msgs::ImageConstPtr& msg){
 
    double height_image = msg -> height;
    double width_image = msg -> width;
    now_time = last_time_;

    now_time = msg ->header.seq;

    
    cv_bridge::CvImageConstPtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvShare(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception:%s",e.what());
        return;
    }

    bool test = camerainfo;

    if(test){
        cv::undistort(cv_ptr -> image,img,K,D);
    }
    else{
        img = cv_ptr -> image.clone();
    }
    

    //传来的信息等于center

    for(size_t i =0;i< center.size();i++){
        dt = now_time - last_time_;
        if(dt >0.5){
            //调用函数
            stop(last_time_,first);
        }
    tracking = true;

    
    //调用卡尔曼后:
    cv::Point3f position_predict = Tracker::KalmanFilter(kf,center[i],dt);
    predictions.push_back(position_predict);


    visualization_msgs::Marker line_prediction_Marker;

    line_prediction_Marker.header.stamp = msg->header.stamp;
    line_prediction_Marker.header.frame_id = msg->header.frame_id;
    line_prediction_Marker.ns = "measurements";
    line_prediction_Marker.id = 4;
    line_prediction_Marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_prediction_Marker.action = visualization_msgs::Marker::ADD;

    line_prediction_Marker.scale.x = 0.02 //线粗细，yaml
    line_prediction_Marker.color.r = 1.0;//红色预测
    line_prediction_Marker.color.g = 0.0;
    line_prediction_Marker.color.b = 0.0;
    line_prediction_Marker.color.a = 1.0;//不透明度

    line_prediction_Marker.points = measurements;
    ft_pub.publish(line_prediction_Marker);

    last_time_ = now_time;

    }
}




//卡尔曼(由返回T后值，无返回预测值)
cv::Point3f Tracker::KalmanFilter(cv::KalmanFilter& kf,cv::Point3f& pos,double dt){
    cv::Point3f output_pred;

    if(first){
        kf.statePost.at<float>(0) = pos.x;
        kf.statePost.at<float>(1) = pos.y;
        kf.statePost.at<float>(2) = pos.z;
        kf.statePost.at<float>(3) = 0.0;
        kf.statePost.at<float>(4) = 0.0;
        kf.statePost.at<float>(5) = 0.0;

        first = false;
        output_pred = pos;
        return output_pred;
    }

    A.at<float>(0,3) = dt;//x = x+dx;
    A.at<float>(1,4) = dt;//y = y+dy;
    A.at<float>(2,5) = dt;//z = z+dz;

        //预测
        cv::Mat prediction = kf.predict();
        cv::Point3f x_pred;
        x_pred.x = prediction.at<float>(0);
        x_pred.y = prediction.at<float>(1);
        x_pred.z = prediction.at<float>(2);

        try{
            cv::Mat meas = cv::Mat_<float>(3,1);
            meas.at<float>(0) = pos.x;
            meas.at<float>(1) = pos.y;
            meas.at<float>(2) = pos.z;

            //矫正
            cv::Mat correction = kf.correct(meas);

            //当前最优位置
            float x = kf.statePost.at<float>(0);
            float y = kf.statePost.at<float>(1);
            float z = kf.statePost.at<float>(2);
            float vx = kf.statePost.at<float>(3);
            float vy = kf.statePost.at<float>(4);
            float vz = kf.statePost.at<float>(5);

            //提前T时候
            output_pred.x = x + vx*T;
            output_pred.y = y + vy*T;
            output_pred.z = z + vz*T;
            return output_pred; 
        }
        catch(const cv::Exception& e){
            output_pred.x = kf.statePre.at<float>(0);
            output_pred.y = kf.statePre.at<float>(1);
            output_pred.z = kf.statePre.at<float>(2);

            tracking = false;
            return output_pred;
        }
    
}




int main(int argc,char **argv){
    setlocale(LC_CTYPE,"zh_CN.utf8");
    ros::NodeHandle nh;
    ros::init(argc,argv,"ft_tracker");

    Tracker ic(nh);
    ros::spin();
    //程序退出前销毁窗口
    cv::destroyAllWindows;
    return 0;
}