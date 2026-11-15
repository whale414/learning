#include"ros/ros.h"
#include"image_transport/image_transport.h"
#include"cv_bridge/cv_bridge.h"
#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"sstream"
#include"sensor_msgs/CameraInfo.h"
#include"iostream"
#include"vector"

#include"mutex"//用于线程安全


#include<opencv2/imgproc.hpp>



class ImageConverter{
    private:
    ros::NodeHandle nh_;

    image_transport::ImageTransport it_;
    
    image_transport::Subscriber image_sub;
    ros::Subscriber info_sub;
    image_transport::Publisher image_pub;

    double fx ,fy , cx , cy ,img_height ,img_width;
    cv::Mat K;
    cv::Mat D;
    std::string image_topic,info_topic,pub_topic;

    bool camera_info = false;

    //缓存最新的camera_info
    sensor_msgs::CameraInfo::ConstPtr latest_info_;
    std::mutex info_mutex_;

    void ImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void CameraCallback(const sensor_msgs::CameraInfo& msg);

    public:
    ImageConverter():it_(nh_){
        //读取参数
        nh_.param<std::string>("image_topic",image_topic,"/hk_camera/image_raw");
        nh_.param<std::string>("info_topic",info_topic,"/hk_camera/camera_info");
        nh_.param<std::string>("pub_topic",pub_topic,"/ft/debug/image");

        image_sub = it_.subscribe(image_topic,1,&ImageConverter::ImageCallback,this);
        info_sub = nh_.subscribe(info_topic,1,&ImageConverter::CameraCallback,this);

        image_pub = it_.advertise(pub_topic,1);
    }

    
};

    void ImageConverter::ImageCallback(const sensor_msgs::ImageConstPtr& msg){
        
        ros::Time stamp = msg->header.stamp;
        std::string frame_id = msg->header.frame_id;
        
        cv_bridge::CvImageConstPtr cv_ptr;

        try{
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
         }
    
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception:%s",e.what());
            return;
        }

        cv::Mat img;
        cv::Mat img01;
        {
            std::lock_guard<std::mutex>lock(info_mutex_);
            if (camera_info)
            {
                cv::undistort(cv_ptr->image,img,K,D);
            }
            else{
                return;
            }
            //需要将 cv::Mat 转回 ROS 消息
        cv_bridge::CvImage out_cv_img;
        out_cv_img.header = msg->header; // 保留原始 header 的时间戳和 frame_id
        out_cv_img.encoding = sensor_msgs::image_encodings::BGR8;
        out_cv_img.image = img;

        image_pub.publish(out_cv_img.toImageMsg());
        //

        cv::Mat frame =cv_ptr->image;
        cv::Mat ProcessImage(const cv::Mat&frame);
        cv::VideoCapture cap("image_topic");
        cv::Mat hsv,mask,res;
        cv::cvtColor(frame,hsv,cv::COLOR_BGR2HSV);

        cv::Scalar lower_blue(110,50,50);
        cv::Scalar upper_blue(130,255,255);

        cv::inRange(hsv,lower_blue,upper_blue,mask);
        cv::bitwise_and(frame,frame,res,mask);

        //把Mat信息转化为ros
        cv_bridge::CvImage out_color_img;
        out_color_img.header = msg->header;
        out_color_img.encoding ="bgr8";
        out_color_img.image = res;
        image_pub.publish(out_color_img.toImageMsg());

        }
        
    }
    void ImageConverter::CameraCallback(const sensor_msgs::CameraInfo& msg)
    {
        std::lock_guard<std::mutex>lock(info_mutex_);

        K = cv::Mat(3,3,CV_64F,(void*)msg.K.data()).clone();

        D = cv::Mat(msg.D,true);

        img_width = msg.width;
        img_height = msg.height;

        camera_info = true;

    }


    int main(int argc,char** argv){
        setlocale(LC_CTYPE,"zh_CN.utf8");

        ros::init(argc,argv,"image");
        ImageConverter ic;
        ImageConverter converter;

        ros::spin();
        return 0;
    }