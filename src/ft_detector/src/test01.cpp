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

#include<cstring>

//tf发布
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

//四元数
#include<Eigen/Dense>         //先引用Eigen再用细致的
#include <opencv2/core/eigen.hpp>   //矩阵转四元数



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

    void poseCallback (const Eigen::Quaterniond &q , const cv::Mat& tvec ,ros::Time stamp,int index);

    tf2_ros::TransformBroadcaster tf_broadcaster_;

    struct ArmorResult {
        double x_center;
        Eigen::Quaterniond q;
        cv::Mat tvec;
    };
    std::vector<ArmorResult> results;
    
  



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

       Eigen::Quaterniond matToQuat(const cv::Mat& R){

        Eigen::Matrix3d m;
        cv::cv2eigen(R,m);
        //矩阵构造四元数；
        Eigen::Quaterniond q(m);   //必须单位化
        q.normalize();
        return q;
        
    }




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

        cv::Scalar lower_orange1(10,100,100);
        cv::Scalar upper_orange1(35,255,255);


        cv::Scalar lower_blue(90,120,80);
        cv::Scalar upper_blue(130,255,255);

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

        cv::Mat dst;

        ///// 1. 先用小核中值滤波消除掩码里的椒盐噪点
        // 注意：为了防止原地操作引起问题，这里用临时变量过渡
        cv::Mat temp_mask;
        if (mask_final.channels() == 3) {
                cv::cvtColor(mask_final, mask_final, cv::COLOR_BGR2GRAY); 
        // 强制转单通道，防患未然
        }
        cv::medianBlur(mask_final, temp_mask, 3);
        // 安全判断：如果掩膜全黑或为空，直接跳过后续处理
        if (temp_mask.empty() || cv::countNonZero(temp_mask) == 0) {
                ROS_WARN("Mask is empty, skip morphology.");
        }


        // 2. 开运算：去除孤立反光噪点
        cv::Mat cleaned_mask;
        cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(temp_mask, cleaned_mask, cv::MORPH_OPEN, kernel_open);
        // 3. 闭运算：连接断裂灯条
       
        cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)); 
        cv::morphologyEx(cleaned_mask, cleaned_mask, cv::MORPH_CLOSE, kernel_close);

   //修复 之前的dst完全是空的，这里给它赋值：把清理后的掩码膨胀连通
    cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::dilate(cleaned_mask, dst, kernel_dilate);

        // 4. 将清理后的掩膜赋值回 mask_final（如果你后续还要用 mask_final）
        mask_final = cleaned_mask.clone(); 
        // 5. 找轮廓
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask_final, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat result_img = img.clone();

        //一开始用split拆分，会发现很容易受环境的影响，后面改成提取mask但是单通道无法直接转换
cv::Mat result_img_1 = img.clone();
cv::Mat copy = mask_final.clone();
// 1. 确保 mask 是单通道
if (copy.channels() != 1) {
    cv::cvtColor(copy, copy, cv::COLOR_BGR2GRAY);
}
// 2. 使用 mask 提取原始图像中的感兴趣区域 (ROI) 像素
// 方法：计算原始图像在 mask 非零区域的均值
cv::Scalar mean_bgr = cv::mean(result_img_1,copy); 

double B = mean_bgr[0];
double G = mean_bgr[1];
double R = mean_bgr[2];

// 3. 增加鲁棒性判断：不仅比较 R 和 B，还要看饱和度或差值
// 红色特征：R 远大于 B 和 G
// 蓝色特征：B 远大于 R 和 G

double diff_rb = R - B;

if (diff_rb > 10) { 
    // 红色分量显著高于蓝色
    ROS_INFO("Light is RED\n");
} 
else if (diff_rb < -10) { 
    // 蓝色分量显著高于红色
    ROS_INFO("Light is BLUE\n");
} 
else {
    // 差值太小，可能是白色、灰色或噪声
    ROS_INFO("Color Unclear\n");
}



        std::vector<cv::RotatedRect> Light_bars;


        //由于卡顿：提取声明；
        cv::Mat mask_roi;

        for (size_t i = 0;i < contours.size();i++){

            double area = cv::contourArea(contours[i]);
            if (area < 20) 
            continue; 

            // ：凸度过滤
            std::vector<cv::Point> hull;
            cv::convexHull(contours[i], hull);
            double hull_area = cv::contourArea(hull);

            if (hull_area < 1e-6) 
            continue;
            double solidity = area / hull_area;

            if (solidity < 0.8) 
            continue; 

        
            cv::RotatedRect rect = cv::minAreaRect(contours[i]);

            cv::Rect rect_box = rect.boundingRect();

             // --- 新增的安全保护：把矩形限制在图像边界内 ---
             // 1. 防止 x 或 y 为负数
             rect_box.x = std::max(0, rect_box.x);
             rect_box.y = std::max(0, rect_box.y);
             // 2. 防止宽度或高度超出图像右下角边界
             rect_box.width = std::min(dst.cols - rect_box.x, rect_box.width);
             rect_box.height = std::min(dst.rows - rect_box.y, rect_box.height);
             // 3. 再次检查裁剪区域是否有效（防止宽高为0或负数）
             if (rect_box.width <= 0 || rect_box.height <= 0) {
                continue;
            }

    
            mask_roi = dst(rect_box);
            int white_pixels = cv::countNonZero(mask_roi);
        
            
            float angle = rect.angle;
            float width = rect.size.width;
            float height = rect.size.height;

            if(area < 100.0  || area > 5000){
                continue;
            }

            if(width <0.1f ||height < 0.1f)
            continue;

            //ai建议：height不一定一直大于width
            if(width > height){
                std::swap(width,height);
            }

            //填充率过滤
            
            float rect_area = width * height;
            if (rect_area < 1e-6) 
            continue;
            double fill_ratio = area / rect_area;

            if (fill_ratio < 0.4) 
            continue;


            double rate = height / width;

                if(rate<1 || rate>6)
                continue;
                
           

                // ROS_INFO("Contour: area=%.2f, w=%.2f, h=%.2f, rate=%.2f, angle=%.2f",area, width, height, height/width, angle);


        
            
            //绿色筛选框
                // cv::Scalar draw_color = cv::Scalar(0,255,0);

                // //获取矩阵的四个顶点

                // std::vector<std::vector<cv::Point>> boxContours(1);
                // cv::Point2f vertices[4];
                // rect.points(vertices);

                // //     for (int n = 0;n<4;n++){
                // //         boxContours[0].push_back(cv::Point(cvRound(vertices[n].x),cvRound(vertices[n].y)));
                // //     }
                // //     std::vector<std::vector<cv::Point>> boxContours1(1,boxContours[0]);
                    
                // //     cv::drawContours(result_img,boxContours,-1,draw_color,2);
                // //     //绘制轮廓，线绿色，宽2

                // // }
                  

                Light_bars.push_back(rect);

        }
                std::vector<cv::Point2f> centers_final_last;

                int armor_id = 0;
               

                for(size_t i = 0;i < Light_bars.size();i++){
                    for(size_t j = i+1;j < Light_bars.size();j++){
                        //；两个灯条中心
                        cv::Point2f center1 = Light_bars[i].center;
                        cv::Point2f center2 = Light_bars[j].center;
                        //两个等条的旋转矩形信息
                        cv::RotatedRect rect1 = Light_bars[i];
                        cv::RotatedRect rect2 = Light_bars[j];

                        //两个灯条筛选出长宽之后，找最短的长度
                        float height_light_1 = std::max(rect1.size.width,rect1.size.height);
                        float height_light_2 = std::max(rect2.size.width,rect2.size.height);
                        float min_height = std::min (height_light_1,height_light_2);

                        //两个灯条之间的距离
                        float distance = std::sqrt((center1.x-center2.x)*(center1.x-center2.x)+(center1.y-center2.y)*(center1.y-center2.y));

                        //装甲板的长和灯条宽度比值
                        double rate_light_1 = distance / min_height;

                        if(rate_light_1 < 1.5 || rate_light_1 >6)
                        continue;

                        //灯条角度
                        float angle_light_1 = rect1.angle;
                        float angle_light_2 = rect2.angle;

                        if(angle_light_1 < -45)
                        angle_light_1 += 90;
                        
                        if(angle_light_2 < -45)
                        angle_light_2 += 90;

                        float dis_angle = abs(angle_light_1 - angle_light_2);
                        
                        if(dis_angle > 3)
                        continue;
                        

                        //装甲板的水平距离和数值长度
                        float horizontal_distance = std::abs(center1.x - center2.x);
                        float vertical_distanch = std::abs(center1.y - center2.y);

                        //补充：查y轴偏离
                        if(vertical_distanch >20.0f)
                        continue;

                        if(horizontal_distance < vertical_distanch)
                        continue;

                        //计算装甲板中心
                        cv::Point2f center_final((center1.x+center2.x)/2,(center1.y+center2.y)/2);

                        cv::Point2f center_point = (center1 + center2)/2.0;

                        centers_final_last.push_back(center_final); //把后者信息赋给前者、


                        

                        double  center_height = (height_light_1 + height_light_2)/2;
                        double  light_width = (rect1.size.width + rect2.size.width)/2;
                        double  center_angle = (angle_light_1 + angle_light_2)/2;

                        double center_width = distance + light_width;
                        

                

                        // 构造装甲板的旋转矩形//2D


                        cv::RotatedRect ban_rect (center_point,cv::Size2f(center_width,center_height),center_angle);

                        cv::Point2f ban_vertices[4];
                        ban_rect.points(ban_vertices);


                       
                       
                        int TL=0,TR=0,BR=0,BL=0;
                        
                        double a=ban_vertices[0].x+ban_vertices[0].y;
                        double b=ban_vertices[0].x-ban_vertices[0].y;
                        double c=ban_vertices[0].x+ban_vertices[0].y;
                        double d=ban_vertices[0].x-ban_vertices[0].y;
                        

                        for(int i = 0; i < 4; i++){
                            double add = ban_vertices[i].x+ban_vertices[i].y;
                            double diff =ban_vertices[i].x-ban_vertices[i].y;
                        
                            //左上 -x -y；
                            if(add<a){
                                a =add;
                                TL = i;
                            }

                            //右上  +x，-y；
                            if(diff > b){
                                b = diff;
                                TR = i;
                            }

                            //BR +x +y
                            if(add >c){
                                c = add;
                                BR = i;
                            }

                            //-x,+y;
                            if(diff <d){
                                d = diff;
                                BL = i;
                            }

                        }


                std::vector<cv::Point2f> img_points{
                    ban_vertices[TL],
                    ban_vertices[TR],
                    ban_vertices[BR],
                    ban_vertices[BL]
                };

                
  
                        double ARMOR_WIDTH ;
                        double ARMOR_HEIGHT;
                    
                        if(rate_light_1 <= 3.5){


                            for(int i =0 ;i<4;i++){
                            cv::line(result_img,cv::Point2f(ban_vertices[i]),cv::Point2f(ban_vertices[(i+1)%4]),cv::Scalar(0,255,0),3);
                        }
                        ARMOR_WIDTH = 0.125;
                        ARMOR_HEIGHT = 0.120;
                        
                        // ROS_INFO("Small armor plate detected");
                    }
                        else {
                        for(int i =0 ;i<4;i++){
                
                            cv::line(result_img,cv::Point2f(ban_vertices[i]),cv::Point2f(ban_vertices[(i+1)%4]),cv::Scalar(0,0,255),3);
                        }
                        ARMOR_WIDTH = 0.235;
                        ARMOR_HEIGHT = 0.127;
                        // ROS_INFO("Large armor plate detected");
                    }
            

// 安全读取并校验相机内参
cv::Mat local_K, local_D;
bool local_camera_info = false;
{
    std::lock_guard<std::mutex> lock(info_mutex_);
    local_K = K.clone();
    local_D = D.clone();
    local_camera_info = camera_info;
}
             

                    //pnp 位姿解算


                    std::vector <cv::Point3f> object_points{//板子四个点
                        {-float(ARMOR_WIDTH/2),-float(ARMOR_HEIGHT/2),0},
                        {float(ARMOR_WIDTH/2),-float(ARMOR_HEIGHT/2),0},
                        {float(ARMOR_WIDTH/2),float(ARMOR_HEIGHT/2),0},
                        {-float(ARMOR_WIDTH/2),float(ARMOR_HEIGHT/2),0}
                    };
                    

                    
                    bool solvepnp =false;

                    if(camera_info  && img_points.size() == 4 && object_points.size() == 4){
                        std::vector<cv::Mat> rvecs,tvecs;
                       // solvepnp = cv::solvePnP(object_points,img_points,local_K,local_D,rvec,tvec,false,cv::SOLVEPNP_IPPE);
                    // std::vector<cv::Point2f> object_1_points;
                    // std::vector<cv::Point3f> test_point_test01{
                    //     {0.03,0,0},
                    //     {0,0.03,0},
                    //     {0,0,0.03},
                    // };

                    //if(solvepnp){
                    //cv::projectPoints(test_point_test01,rvec,tvec,local_K,local_D,object_1_points);
                    //}

                    // 从中心点画三轴，直观展示位姿
                //         if(object_1_points.size() == 3){
                //             cv::line(result_img, center_point, object_1_points[0], cv::Scalar(0,0,255), 2); 
                //             cv::line(result_img, center_point, object_1_points[1], cv::Scalar(0,255,0), 2); 
                //             cv::line(result_img, center_point, object_1_points[2], cv::Scalar(255,0,0), 2); 

                // }


                //重投影
                
                
             bool success = cv::solvePnPGeneric(object_points, img_points, local_K, local_D, rvecs, tvecs, cv::SOLVEPNP_IPPE);

         
    // 解算结果有效性校验
if (!success || rvecs.empty() || tvecs.empty() || rvecs.size() != tvecs.size()) {
    ROS_WARN("PnP solve failed or no valid solution");
    continue;
}

                cv::Mat best_rvec, best_tvec;
                double min_error = std::numeric_limits<double>::max();

                bool found_valid_solution = false;
                
                for (int i = 0; i < rvecs.size(); i++) {

                    cv::Mat current_rvec = rvecs[i];
                    cv::Mat current_tvec = tvecs[i];
            
                    if(current_tvec.at<double>(2) <= 0.01){
                        continue;}


                    if (!success||rvecs.empty() || tvecs.empty()){
                    continue;
                    }

                    std::vector<cv::Point2f> projectedPoints;
                    // 把3D点用这个解投影回2D图像
                       cv::projectPoints(object_points, current_rvec, current_tvec, local_K, local_D, projectedPoints);
                    // 计算重投影误差 (所有点的误差平方和)
                    double error = 0;
                    for (size_t j = 0; j < img_points.size(); j++) {
                        
                        error += cv::norm(cv::Point2f(img_points[j]) - projectedPoints[j]);
                    }
                    // 选出误差最小的那个解
                    if (error < min_error) {
                        min_error = error;
                        best_rvec = current_rvec;
                        best_tvec = current_tvec;
                        found_valid_solution = true;
                    }
                }
                    // 此时 best_rvec 和 best_tvec 就是最符合图像的解

                    //防止反转
                if(found_valid_solution && !best_tvec.empty()){
                cv::Mat R;
                cv::Rodrigues(best_rvec,R);

                Eigen::Quaterniond q = matToQuat(R);

                ImageConverter::ArmorResult res;
                res.x_center = center_point.x;

                res.q = q;
                res.tvec = best_tvec.clone();

                results.push_back(res);
                //ROS_INFO("Quat: x=%.3f, y=%.3f, z=%.3f, w=%.3f", q.x(), q.y(), q.z(), q.w());
                }
                
                
            }

        }

    

            
}
    std::set<int>published_ids;

    std::sort(results.begin(), results.end(), [](const ArmorResult& a, const ArmorResult& b) {
        return a.x_center < b.x_center;
    });

    for(size_t i = 0;i<results.size();i++){
        if(published_ids.count(i)){
            continue;
        }
        poseCallback(results[i].q,results[i].tvec, msg->header.stamp,i);
        
    }

        cv_bridge::CvImage out_contour_img;
        out_contour_img.header = msg -> header;
        out_contour_img.encoding = sensor_msgs::image_encodings::BGR8;
        out_contour_img.image = result_img;
        contours_pub.publish(out_contour_img.toImageMsg());
    }




    struct posedata{
        
        std::string frame_id;
    };

    void ImageConverter::poseCallback (const Eigen::Quaterniond& q , const cv::Mat& tvec ,ros::Time stamp,int index){

        static tf2_ros::TransformBroadcaster br;//激活
        geometry_msgs::TransformStamped transformstamped;//通信对象；

        transformstamped.header.stamp = stamp;
        transformstamped.header.frame_id = "camera_optical_frame";
        
        transformstamped.child_frame_id = "armor_" + std::to_string(index);

        //tvec
        transformstamped.transform.translation.x = tvec.at<double>(0);
        transformstamped.transform.translation.y = tvec.at<double>(1);
        transformstamped.transform.translation.z = tvec.at<double>(2);

        //四元数
        transformstamped.transform.rotation.w = q.w();
        transformstamped.transform.rotation.x = q.x();
        transformstamped.transform.rotation.y = q.y();
        transformstamped.transform.rotation.z = q.z();

        // ROS_INFO("tvec:x=%.3f,y=%.3f,z=%.3f",tvec.at<double>(0),tvec.at<double>(1),tvec.at<double>(2));
        
        br.sendTransform(transformstamped);

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