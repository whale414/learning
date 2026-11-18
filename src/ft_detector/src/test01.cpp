#include"ros/ros.h"
#include"image_transport/image_transport.h"
#include"cv_bridge/cv_bridge.h"
#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"sstream"
#include"sensor_msgs/CameraInfo.h"
#include"iostream"
#include"vector"

#include<std_msgs/Header.h>

#include"mutex"//用于线程安全


#include<opencv2/imgproc.hpp>



class ImageConverter{
    private:
    ros::NodeHandle nh_;

    image_transport::ImageTransport it_;
    
    image_transport::Subscriber image_sub;
    ros::Subscriber info_sub;
    image_transport::Publisher image_pub;
    image_transport::Publisher mark_pub;
    image_transport::Publisher contours_pub;



    double fx ,fy , cx , cy ,img_height ,img_width;
    cv::Mat K;
    cv::Mat D;
    std::string image_topic,info_topic,pub_topic,mark_topic,contours_topic;

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
        nh_.param<std::string>("mark_topic",mark_topic,"/ft/debug/mask_image");
        nh_.param<std::string>("contours_topic",contours_topic,"/ft/debug/contours_image");

        image_sub = it_.subscribe(image_topic,1,&ImageConverter::ImageCallback,this);
        info_sub = nh_.subscribe(info_topic,1,&ImageConverter::CameraCallback,this);


        image_pub = it_.advertise(pub_topic,1);
        mark_pub = it_.advertise(mark_topic,1);
        contours_pub = it_.advertise(contours_topic,1);
    }

};   


    void ImageConverter::ImageCallback(const sensor_msgs::ImageConstPtr& msg){
        //检验
        ROS_INFO("Received image,encoding:%s",msg->encoding.c_str());

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
    
        bool test = false;

        {
        //把锁缩小到仅读取标定参数的瞬间
        std::lock_guard<std::mutex>lock(info_mutex_);
        test = camera_info;
        }

            if (test)
            {
                cv::undistort(cv_ptr->image,img,K,D);
            }
            else{
                img = cv_ptr->image.clone();
            }

        //需要将 cv::Mat 转回 ROS 消息
        cv_bridge::CvImage out_cv_img;
        out_cv_img.header = msg->header; // 保留原始 header 的时间戳和 frame_id
        out_cv_img.encoding = sensor_msgs::image_encodings::BGR8;
        out_cv_img.image = img;
        image_pub.publish(out_cv_img.toImageMsg());

        cv::Mat frame = img;
        cv::Mat hsv,mask,res;
        cv::Mat mask1,mask2,red_mask,res1;
        cv::Mat mask_final;
        cv::Mat mask_orange;
        cv::Mat mask_0;

        cv::cvtColor(frame,hsv,cv::COLOR_BGR2HSV);

        cv::Scalar lower_red1(0,200,100);
        cv::Scalar upper_red1(10,255,255);

        cv::Scalar lower_red2(160,200,100);
        cv::Scalar upper_red2(180,255,255);

        cv::Scalar lower_orange1(10,100,50);
        cv::Scalar upper_orange1(30,255,255);


        cv::Scalar lower_blue(100,80,0);
        cv::Scalar upper_blue(125,255,255);

        cv::inRange(hsv,lower_blue,upper_blue,mask);
        cv::inRange(hsv,lower_red1,upper_red1,mask1);
        cv::inRange(hsv,lower_red2,upper_red2,mask2);

        cv::inRange(hsv,lower_orange1,upper_orange1,mask_orange);

        cv::bitwise_or(mask1,mask2,red_mask);
        cv::bitwise_or(red_mask,mask,mask_0);
        cv::bitwise_or(mask_0,mask_orange,mask_final);
        cv::bitwise_and(frame,frame,res,mask_final);

        cv_bridge::CvImage out_mark_img;
        out_mark_img.header = msg->header;
        out_mark_img.encoding =sensor_msgs::image_encodings::BGR8;
        out_mark_img.image = res;
        mark_pub.publish(out_mark_img.toImageMsg());

        
        //转二值
        
        std::vector<std::vector<cv::Point>> contours;   //轮廓
        std::vector<cv::Vec4i> hierarchy;   //层级关系
        cv::findContours(mask_final,contours,hierarchy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);


         cv::Mat result_img = mask_final.clone();//画布要放到for外面
        
        for (size_t i = 0;i < contours.size();i++){

            //过滤面积过于小的噪点
            if(cv::contourArea(contours[i])< 50)
                continue;
        
            cv::RotatedRect rect = cv::minAreaRect(contours[i]);
            
            float angle = rect.angle;
            float width = rect.size.width;
            float height = rect.size.height;

            float area = rect.size.width * rect.size.height;
            if(area < 1000.0f){
                continue;
            }

            if(width <0.1f &&height < 0.1f)
            continue;

            //ai建议：height不一定一直大于width
            if(width > height){
                std::swap(width,height);
            }

                double rate = height / width;

                if(rate<1.0 || rate>4.0)
                continue;
                
                if(std::abs(angle) >25.0f)
                continue;



                cv::Rect rect_box = rect.boundingRect();

                rect_box.x = std::max(0,rect_box.x);
                rect_box.y = std::max(0,rect_box.y);
                rect_box.width = std::min(img.cols - rect_box.x,rect_box.width);
                rect_box.height = std::min(img.rows - rect_box.y,rect_box.height);

                cv::Scalar draw_color = cv::Scalar(0,255,0);
                if(rect_box.width > 0 && rect_box.height > 0){
                    cv::Mat mask_roi = mask_final(rect_box);
                    int white_pixels = cv::countNonZero(mask_roi);

                }
                else{
                    draw_color = cv::Scalar(255,0,0);
                }

                //获取矩阵的四个顶点

                std::vector<std::vector<cv::Point>> boxContours(1);

                cv::Point2f vertices[4];
                rect.points(vertices);

                    for (int n = 0;n<4;n++){
                        boxContours[0].push_back(cv::Point(cvRound(vertices[n].x),cvRound(vertices[n].y)));
                    }
                    
                    cv::drawContours(mask_final,boxContours,-1,draw_color,2);
                    //绘制轮廓，线绿色，宽2

                }

                    cv_bridge::CvImage out_contours_img;
                    out_contours_img.header = msg->header;
                    out_contours_img.image = result_img;
                    out_contours_img.encoding = sensor_msgs::image_encodings::BGR8;
                    contours_pub.publish(out_contours_img.toImageMsg());


            
        
        }
    

    void ImageConverter::CameraCallback(const sensor_msgs::CameraInfo& msg)
    {
        std::lock_guard<std::mutex>lock(info_mutex_);

        K = cv::Mat(3,3,CV_64F,(void*)msg.K.data()).clone();

        D = cv::Mat(msg.D.size(),1,CV_64F , (void*)msg.D.data()).clone();


        camera_info = true;

    }



    int main(int argc,char** argv){
        setlocale(LC_CTYPE,"zh_CN.utf8");

        ros::init(argc,argv,"image");
        ImageConverter ic;

        ros::spin();
        return 0;
    }