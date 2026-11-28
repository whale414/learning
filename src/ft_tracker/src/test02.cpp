#include<iostream>
#include"ros/ros.h"
#include"std_msgs/String.h"

#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
//#include <image_transport/image_transport.h>

#include"sensor_msgs/CameraInfo.h"
#include <sensor_msgs/Image.h>

//#include"cv_bridge/cv_bridge.h"

#include <mutex>//明天加锁头

#include <vector>
#include <cmath>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include "ft_detector/all.h"
#include "ft_detector/armor.h"
#include "ft_tracker/track.h"

#include<yaml-cpp/yaml.h>
#include <ros/package.h> 

class Tracker{
    private:
    
    ros::NodeHandle nh_;
    int state_size = 6;//[x,y,z,vx,vy,vz]
    int measurement_size = 3;
    int control_size = 0;

    cv::Mat A;
    cv::Mat H;
    double Q = 0.01;
    double R = 0.1;
    double P =1.0;

    cv::Mat K;
    cv::Mat D;
    cv::Mat img;

    bool camerainfo = false;

    //ros::Subscriber info_sub;
    //std::string info_topic;

    ros::Publisher ft_pub;
    std::string ft_P_topic;

    ros::Publisher track_pub;
    std::string ft_T_topic;

    ros::Subscriber ft_sub;
    std::string ft_S_topic;

    //锁头
    std::mutex info_mutex_;

    int queue_size = 1;

    sensor_msgs::CameraInfo::ConstPtr latest_info_;

    double last_obs_time_ = 0.0;   //最近一次有观测的时间（超时用）
    double last_time_ = 0.0;       //上一帧图像时间（dt用）
    double now_time = 0.0;

    std::string frame_id; //用yaml定住一下

    cv::KalmanFilter kf;

    double T = 0.1;        //飞行时间yaml
    double gate = 0.5;     //关联门限yaml
    double lost_timeout = 0.5; //连续丢失删除轨迹yaml

    std::vector<geometry_msgs::Point> measurements;
    std::vector<geometry_msgs::Point> predictions;

    bool first = true;
    bool tracking = false;   //本帧是否有观测命中

    bool has_target_ = false; //是否已有轨迹
    int track_id_ = 0;        //轨迹ID

    size_t MAX_TRAIL; //轨迹点上限

    //void CameraInfoCallback(const sensor_msgs::CameraInfo& msg);
    //void ImageCallback(const sensor_msgs::ImageConstPtr& msg);
    cv::Point3f KalmanFilter(cv::KalmanFilter& kf,cv::Point3f& pos,double dt,bool has_measurement);
    void msgcallback(const ft_detector::all::ConstPtr& msg);
    void stop();

    static double dist3(const cv::Point3f& a,const cv::Point3f& b)//两点距离
    {
        double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

//ROS 订阅是异步的，回调可能在任何时候触发，建议private初始化一次
    void configureKalman()
    {
        kf.init(state_size,measurement_size,control_size);

        A = cv::Mat::eye(state_size,state_size,CV_32F);
        A.at<float>(0,3) = 1;//x = x+dx;
        A.at<float>(1,4) = 1;//y = y+dy;
        A.at<float>(2,5) = 1;//z = z+dz
        kf.transitionMatrix = A;

        cv::setIdentity(kf.measurementMatrix);
        cv::setIdentity(kf.processNoiseCov,cv::Scalar::all(Q));//测量
        cv::setIdentity(kf.measurementNoiseCov,cv::Scalar::all(R));//传感器
        cv::setIdentity(kf.errorCovPost,cv::Scalar::all(P));//后验误差
    }

    //用观测初始化状态
    void initKalman(const cv::Point3f& p0)
    {
        configureKalman();
        kf.statePost.at<float>(0) = p0.x;//xyz
        kf.statePost.at<float>(1) = p0.y;
        kf.statePost.at<float>(2) = p0.z;
        kf.statePost.at<float>(3) = 0.0;
        kf.statePost.at<float>(4) = 0.0;
        kf.statePost.at<float>(5) = 0.0;
        kf.statePre = kf.statePost.clone();

        first = false;
    }









//构造函数
public:
    Tracker(ros::NodeHandle& nh): nh_("~")
    {

    //读
        std::string file_path = ros::package::getPath("ft_detector") + "/config/yaml.yaml";
    
        YAML::Node config = YAML::LoadFile(file_path);

    //nh_.param<std::string>("info_topic",info_topic,"/hk_camera/camera_info");
    nh_.param<std::string>("ft_pub",ft_P_topic,"/ft/debug/marker");
    nh_.param<std::string>("ft_sub",ft_S_topic,"/ft/detection");
    nh_.param<std::string>("track_pub",ft_T_topic,"/ft/track");

    /*
  Q: 0.01

  R: 0.1

  P: 1.0

  queue_size: 1

  T: 0.1

  gate: 0.5

  lost_timeout: 0.5
    */


        //读
    YAML::Node det = config["ft_tracker_node"];   // ← 先取子节点

    Q = det["Q"].as<double>();
    R = det["R"].as<double>();
    P = det["P"].as<double>();
    queue_size = det["queue_size"].as<int>();
    T = det["T"].as<double>();
    gate = det["gate"].as<double>();
    lost_timeout = det["lost_timeout"].as<double>();
    MAX_TRAIL = det["MAX_TRAIL"].as<size_t>();
    
    //info_sub = nh_.subscribe(info_topic,queue_size,&Tracker::CameraInfoCallback,this);
    ft_sub = nh_.subscribe(ft_S_topic,queue_size,&Tracker::msgcallback,this);

    //注意消息类型
    ft_pub = nh_.advertise<visualization_msgs::Marker>(ft_P_topic,queue_size);
    track_pub = nh_.advertise<ft_tracker::track>(ft_T_topic,queue_size);

    //卡尔曼
    configureKalman();
}
};



//拿
void Tracker::msgcallback(const ft_detector::all::ConstPtr& msg){

    now_time = msg->header.stamp.toSec();
    frame_id = msg->header.frame_id;

    //dt用两帧图像时间戳相减，首帧dt=0
    double frame_dt = 0.0;
    if(last_time_ > 0.0){
        frame_dt = now_time - last_time_;
        if(frame_dt < 0.0) frame_dt = 0.0;
    }
    last_time_ = now_time;

    //有装甲板
    bool has_measurement = !msg->armors.empty();
    cv::Point3f pos_msg(0.f,0.f,0.f);

    if(has_measurement){
        pos_msg.x = msg->armors[0].Pose.position.x;
        pos_msg.y = msg->armors[0].Pose.position.y;
        pos_msg.z = msg->armors[0].Pose.position.z;
    }
    else{
        ROS_WARN_THROTTLE(1.0,"漏检，纯预测");
    }

    //有轨迹但超时
    if(has_target_ && (now_time - last_obs_time_ > lost_timeout)){
        ROS_WARN("连续丢失 > %.2f 秒，轨迹重置！",lost_timeout);
        stop();

        //没有装甲板 没有对象无法命中
        if(!has_measurement){
            tracking = false;
            return;
        }
    }

    //无轨迹无对象
    if(!has_target_){
        if(!has_measurement){
            tracking = false;
            return;
        }

        //初始化
        initKalman(pos_msg);
        has_target_ = true;

        //门限重置和超时重新捕捉的时候重新找id
        track_id_++;
        tracking = true;
        last_obs_time_ = now_time;

        //观测
        geometry_msgs::Point p;
        p.x = pos_msg.x; 
        p.y = pos_msg.y; 
        p.z = pos_msg.z;
        measurements.push_back(p);
        
        //超过范围得删掉，不然滞留
        if(measurements.size() > MAX_TRAIL) measurements.erase(measurements.begin());

        cv::Point3f vel(0.f,0.f,0.f);
        cv::Point3f pos_predict(pos_msg.x + vel.x*T, pos_msg.y + vel.y*T, pos_msg.z + vel.z*T);

        ft_tracker::track tmsg;
        tmsg.header = msg->header;
        tmsg.id = track_id_;
        tmsg.tracking = true;
        tmsg.position.x = pos_msg.x; tmsg.position.y = pos_msg.y; tmsg.position.z = pos_msg.z;
        tmsg.velocity.x = 0.f; tmsg.velocity.y = 0.f; tmsg.velocity.z = 0.f;
        track_pub.publish(tmsg);


        //建议：
        //画观测点初始位置
        visualization_msgs::Marker line_measurements_Marker;
        line_measurements_Marker.header.stamp = ros::Time::now();
        line_measurements_Marker.header.frame_id = msg->header.frame_id;
        line_measurements_Marker.ns = "measurements";
        line_measurements_Marker.id = 1;
        line_measurements_Marker.type = visualization_msgs::Marker::LINE_STRIP;
        line_measurements_Marker.action = visualization_msgs::Marker::ADD;
        line_measurements_Marker.scale.x = 0.02;

        line_measurements_Marker.color.r = 0.0;
        line_measurements_Marker.color.g = 1.0;//绿色
        line_measurements_Marker.color.b = 0.0;

        line_measurements_Marker.color.a = 0.2;
        line_measurements_Marker.points = measurements;

        ft_pub.publish(line_measurements_Marker);

        //预测
        geometry_msgs::Point b;
        b.x = pos_predict.x; 
        b.y = pos_predict.y; 
        b.z = pos_predict.z;
        predictions.push_back(b);

        if(predictions.size() > MAX_TRAIL) predictions.erase(predictions.begin());

        //画预测点初始位置
        visualization_msgs::Marker line_prediction_Marker;
        line_prediction_Marker.header.stamp = ros::Time::now();
        line_prediction_Marker.header.frame_id = msg->header.frame_id;
        line_prediction_Marker.ns = "predictions";
        line_prediction_Marker.id = 4;
        line_prediction_Marker.type = visualization_msgs::Marker::LINE_STRIP;
        line_prediction_Marker.action = visualization_msgs::Marker::ADD;
        line_prediction_Marker.scale.x = 0.02;

        line_prediction_Marker.color.r = 1.0;//红色
        line_prediction_Marker.color.g = 0.0;
        line_prediction_Marker.color.b = 0.0;

        line_prediction_Marker.color.a = 1.0;
        line_prediction_Marker.points = predictions;

        ft_pub.publish(line_prediction_Marker);

        ROS_INFO("捕获目标 id=%d",track_id_);
        return;
    }


    //关联门限：先预测一步再比较观测
    if(frame_dt > 0.0){//排异常
        A.at<float>(0,3) = frame_dt;
        A.at<float>(1,4) = frame_dt;
        A.at<float>(2,5) = frame_dt;
        kf.transitionMatrix = A;
    }
    
    //预测
    cv::Mat prediction = kf.predict();
    cv::Point3f pred_now(prediction.at<float>(0),prediction.at<float>(1),prediction.at<float>(2));

    if(has_measurement){
        double d = dist3(pred_now,pos_msg);
        if(d > gate){
            ROS_WARN("观测偏离预测 %.3f > %.3f,重新捕获",d,gate);
            stop();


            initKalman(pos_msg);
            has_target_ = true;
            track_id_++;
            tracking = true;
            last_obs_time_ = now_time;

            geometry_msgs::Point p;
            p.x = pos_msg.x; 
            p.y = pos_msg.y; 
            p.z = pos_msg.z;
            measurements.push_back(p);

            if(measurements.size() > MAX_TRAIL) measurements.erase(measurements.begin());

            cv::Point3f vel(0.f,0.f,0.f);
            cv::Point3f pos_predict(pos_msg.x + vel.x*T, pos_msg.y + vel.y*T, pos_msg.z + vel.z*T);

            ft_tracker::track tmsg;
            tmsg.header = msg->header;
            tmsg.id = track_id_;
            tmsg.tracking = true;
            tmsg.position.x = pos_msg.x; tmsg.position.y = pos_msg.y; tmsg.position.z = pos_msg.z;
            tmsg.velocity.x = 0.f; tmsg.velocity.y = 0.f; tmsg.velocity.z = 0.f;
            track_pub.publish(tmsg);

            geometry_msgs::Point b;
            b.x = pos_predict.x; 
            b.y = pos_predict.y; 
            b.z = pos_predict.z;
            predictions.push_back(b);

            if(predictions.size() > MAX_TRAIL) predictions.erase(predictions.begin());
            return;
        }
    }

    
    //有观测才校正
    if(has_measurement){
        cv::Mat meas = (cv::Mat_<float>(3,1) << pos_msg.x,pos_msg.y,pos_msg.z);
        kf.correct(meas);
        //ROS_INFO("obs=(%.2f,%.2f,%.2f) filt=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f,%.2f) pred=(%.2f,%.2f,%.2f) d=%.2f trk=%d id=%d");

        tracking = true;
        last_obs_time_ = now_time;

        geometry_msgs::Point p;
        p.x = pos_msg.x; 
        p.y = pos_msg.y; 
        p.z = pos_msg.z;
        measurements.push_back(p);

        if(measurements.size() > MAX_TRAIL) measurements.erase(measurements.begin());
    }
    else{
        tracking = false;
    }


    //输出滤波后状态 + 提前量T
    cv::Point3f pos_filtered(
        kf.statePost.at<float>(0),
        kf.statePost.at<float>(1),
        kf.statePost.at<float>(2));

    cv::Point3f vel_filtered(
        kf.statePost.at<float>(3),
        kf.statePost.at<float>(4),
        kf.statePost.at<float>(5));

    cv::Point3f pos_predict(
        pos_filtered.x + vel_filtered.x*(float)T,
        pos_filtered.y + vel_filtered.y*(float)T,
        pos_filtered.z + vel_filtered.z*(float)T);

    //发布/ft/track
    ft_tracker::track tmsg;
    tmsg.header = msg->header;
    tmsg.id = track_id_;
    tmsg.tracking = tracking;
    tmsg.position.x = pos_filtered.x;
    tmsg.position.y = pos_filtered.y;
    tmsg.position.z = pos_filtered.z;
    tmsg.velocity.x = vel_filtered.x;
    tmsg.velocity.y = vel_filtered.y;
    tmsg.velocity.z = vel_filtered.z;
    track_pub.publish(tmsg);


    //Marker观测线
    visualization_msgs::Marker line_measurements_Marker;
    line_measurements_Marker.header.stamp = ros::Time::now();
    line_measurements_Marker.header.frame_id = msg->header.frame_id;
    line_measurements_Marker.ns = "measurements";
    line_measurements_Marker.id = 2;
    line_measurements_Marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_measurements_Marker.action = visualization_msgs::Marker::ADD;
    line_measurements_Marker.scale.x = 0.02;

    line_measurements_Marker.color.r = 0.0;
    line_measurements_Marker.color.g = 1.0;//绿色
    line_measurements_Marker.color.b = 0.0;

    line_measurements_Marker.color.a = 0.2;
    line_measurements_Marker.points = measurements;
    ft_pub.publish(line_measurements_Marker);


    geometry_msgs::Point b;
    b.x = pos_predict.x; b.y = pos_predict.y; b.z = pos_predict.z;
    predictions.push_back(b);
    if(predictions.size() > MAX_TRAIL) predictions.erase(predictions.begin());


    //预测线条
    visualization_msgs::Marker line_prediction_Marker;
    line_prediction_Marker.header.stamp = ros::Time::now();
    line_prediction_Marker.header.frame_id = msg->header.frame_id;
    line_prediction_Marker.ns = "predictions";
    line_prediction_Marker.id = 4;
    line_prediction_Marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_prediction_Marker.action = visualization_msgs::Marker::ADD;
    line_prediction_Marker.scale.x = 0.02;
    if(tracking){//命中就绿色
        line_prediction_Marker.color.r = 0.0;
        line_prediction_Marker.color.g = 1.0;
        line_prediction_Marker.color.b = 0.0;
    }
    else{//不命中就黄色
        line_prediction_Marker.color.r = 1.0;
        line_prediction_Marker.color.g = 1.0;
        line_prediction_Marker.color.b = 0.0;
    }
    line_prediction_Marker.color.a = 1.0;
    line_prediction_Marker.points = predictions;

    ft_pub.publish(line_prediction_Marker);


    //球
    visualization_msgs::Marker point_prediction_Marker;
    point_prediction_Marker.header.stamp = ros::Time::now();
    point_prediction_Marker.header.frame_id = msg->header.frame_id;
    point_prediction_Marker.ns = "predict_point";
    point_prediction_Marker.id = 3;
    point_prediction_Marker.type = visualization_msgs::Marker::SPHERE;
    point_prediction_Marker.action = visualization_msgs::Marker::ADD;
    point_prediction_Marker.pose.position.x = pos_predict.x;
    point_prediction_Marker.pose.position.y = pos_predict.y;
    point_prediction_Marker.pose.position.z = pos_predict.z;
    point_prediction_Marker.pose.orientation.w = 1.0;
    point_prediction_Marker.scale.x = 0.05;
    point_prediction_Marker.scale.y = 0.05;
    point_prediction_Marker.scale.z = 0.05;

    point_prediction_Marker.color.r = 1.0;
    point_prediction_Marker.color.g = 0.0;
    point_prediction_Marker.color.b = 0.0;
    
    point_prediction_Marker.color.a = 1.0;
    ft_pub.publish(point_prediction_Marker);
}


//死机
void Tracker::stop(){
    first = true;
    has_target_ = false;
    tracking = false;
    last_obs_time_ = 0.0;

    //last_time_不清零，避免dt过大
    measurements.clear();
    predictions.clear();
    configureKalman();
}


//卡尔曼
cv::Point3f Tracker::KalmanFilter(cv::KalmanFilter& kf,cv::Point3f& pos_msg,double dt,bool has_measurement){
    cv::Point3f output_pred;

    if(first){
        if(!has_measurement)
        return pos_msg;//不初始化

        initKalman(pos_msg);
        return pos_msg;
    }

    if(dt > 0.0){
        A.at<float>(0,3) = dt;
        A.at<float>(1,4) = dt;
        A.at<float>(2,5) = dt;
        kf.transitionMatrix = A;
    }

    //预测
    cv::Mat prediction = kf.predict();

    //有观测才矫正
    if(has_measurement){
        cv::Mat meas = (cv::Mat_<float>(3,1) << pos_msg.x,pos_msg.y,pos_msg.z);
        kf.correct(meas);
        tracking = true;
    }
    else{
        tracking = false;
    }

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




int main(int argc,char **argv){
    setlocale(LC_CTYPE,"zh_CN.utf8");
    ros::init(argc,argv,"ft_tracker_node");
    ros::NodeHandle nh;

    Tracker ic(nh);
    ros::spin();

    return 0;
}