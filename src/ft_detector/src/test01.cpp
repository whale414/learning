#include"ros/ros.h"
#include"image_transport/image_transport.h"
#include"cv_bridge/cv_bridge.h"
#include"opencv2/highgui/highgui.hpp"
#include"sstream"
#include"sensor_msgs/Image.h"
#include"iostream"

class ImageConverter{
    private:
    ros::NodeHandle nh_;

    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub;
    ros::Subscriber info_sub;

    void ImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void CameraCallback(const sensor_msgs::CameraInfo& msg);

    public:
    ImageConverter():it_(nh_){
        image_sub = it_.subscribe("/hk_camera/image_raw/compressed",1,&ImageConverter::ImageCallback,this);
        info_sub = nh_.subscribe("/hk_camera/camera_info",1,CameraCallback,this);
    }
};


    void ImageConverter::ImageCallback(const sensor_msgs::ImageConstPtr& msg){
        cv_bridge::CvImagePtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        }
    
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception:%s",e.what());
            return;
        }

    void CameraCallback(const sensor_msgs::CameraInfo& msg)
    {
        //提取内参矩阵
        double fx = msg ->K[0];
        double fy = msg ->K[4];
        double cx = msg ->K[2];
        double cy = msg ->K[5];

        //提取畸变系数D
        std::vector<double> D = msg ->D;
        std::string model = msg ->distortion_model;
    }


    int main(int argc,char** argv){
        ros::init(argc,argv,"image");
        ImageConverter ic;
        ros::spin();
        return 0;
    }
