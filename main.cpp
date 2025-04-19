#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <opencv2/opencv.hpp>
#include "opencv2/core/version.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "yolo_v2_class.hpp"            // imported functions from .so
#include "RTSPcam.h"
#include "Regression.h"
#include "Tjson.h"
#include "MJPG_sender.h"

#include <queue>
#include <mutex>
#include <condition_variable>

std::queue<std::string> textQueue;
std::mutex queueMutex;
std::condition_variable queueCondVar;
std::set<std::string> processed_carplates;


using namespace std;

//----------------------------------------------------------------------------------------
// set the config.json with its settings global

Tjson Js;

//----------------------------------------------------------------------------------------
void draw_vehicle(cv::Mat& bgr, bbox_t& v)
{
    //Create the rectangle
    cv::Rect roi(v.x+Js.RoiCrop.x, v.y+Js.RoiCrop.y, v.w, v.h);
    if(v.obj_id == 0) cv::rectangle(bgr, roi, cv::Scalar(255, 255,   0),2); //cyan - car
    else              cv::rectangle(bgr, roi, cv::Scalar(255,   0, 255),2); //magenta - motorcycle
}
//----------------------------------------------------------------------------------------
void draw_plate(cv::Mat& bgr, bbox_t& v, bbox_t& p)
{
    //Create the rectangle
    cv::Rect roi(p.x+v.x+Js.RoiCrop.x, p.y+v.y+Js.RoiCrop.y, p.w, p.h);
    cv::rectangle(bgr, roi, cv::Scalar(0, 255, 0),2); //green - plate
}
//----------------------------------------------------------------------------------------
void draw_ocr(cv::Mat& bgr, bbox_t& v, bbox_t& p, vector<bbox_t> result_vec, vector<string> obj_names)
{
    char text[32];
    memset(text, 0, sizeof(text));
    size_t i;
    int baseLine = 0;

    size_t max_len = sizeof(text) - 2;

    if(result_vec.size()==0) return;


    for (i = 0; i < result_vec.size() && i < max_len; i++) {
            int obj_id = result_vec[i].obj_id;
            if (obj_id < obj_names.size()) {
            strncat(text, obj_names[obj_id].c_str(), sizeof(text) - strlen(text) - 1);
            } else {
            strncat(text, "?", sizeof(text) - strlen(text) - 1);
            }
        }
    text[i] = '\0'; //closing (0=endl);


    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = p.x+v.x+Js.RoiCrop.x;
    int y = p.y+v.y+Js.RoiCrop.y - label_size.height - baseLine;
    if (y < 0) y = 0;
    if (x + label_size.width > bgr.cols)  x = bgr.cols - label_size.width;

    cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

}


//----------------------------------------------------------------------------------------
void print_result(vector<bbox_t> const result_vec, vector<string> const obj_names)
{
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) cout << obj_names[i.obj_id] << " - ";
        cout << setprecision(3) << "prob = " << i.prob << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h << endl;
    }
    cout << " " << endl;
}
//----------------------------------------------------------------------------------------
vector<string> objects_names_from_file(string const filename)
{
    ifstream file(filename);
    vector<string> file_lines;

    if (!file.is_open()) return file_lines;

    for(string line; file >> line;) file_lines.push_back(line);

    cout << "object names loaded \n";

    return file_lines;
}
//----------------------------------------------------------------------------------------
//note, SortSingleLine can erase elements from cur_bbox_vec
//that way shorten the length of the vector. Hence size_t& bnd
void SortSingleLine(vector<bbox_t>& cur_bbox_vec, float ch_wd, float ch_ht, size_t StartPos, size_t& StopPos)
{
    size_t i, j;
    bbox_t tmp_box;
    int d, i1, i2;

    if((StopPos-StartPos)<=1) return;

    //sort by x position
    for(i=StartPos; i<StopPos; i++){
        for(j=i+1; j<StopPos; j++){
            if(cur_bbox_vec[j].x<cur_bbox_vec[i].x){
                //swap
                tmp_box=cur_bbox_vec[j];
                cur_bbox_vec[j]=cur_bbox_vec[i];
                cur_bbox_vec[i]=tmp_box;
            }
        }
    }

    //get the distance between each char, too close? select the highest prob.
    for(i=StartPos; i<StopPos-1; i++){
        i1=cur_bbox_vec[i].x; i2=cur_bbox_vec[i+1].x;
        d=(i2-i1)*2;            //d<0? two lines and jumping from the top to the bottom line.
        if(d>=0 && d<ch_wd){
            if(cur_bbox_vec[i+1].prob < cur_bbox_vec[i].prob) cur_bbox_vec.erase(cur_bbox_vec.begin()+i+1);
            else                                              cur_bbox_vec.erase(cur_bbox_vec.begin()+i);
            StopPos--;  i--;    //one element less in the array, due to the erase
        }
    }
}
// Convert cv::Mat to image_t
image_t mat_to_image(const cv::Mat& mat) {
    int h = mat.rows;
    int w = mat.cols;
    int c = mat.channels();
    image_t im;
    im.data = new float[h * w * c];
    im.h = h;
    im.w = w;
    im.c = c;

    for (int i = 0; i < h; ++i) {
        for (int k = 0; k < c; ++k) {
            for (int j = 0; j < w; ++j) {
                im.data[k * h * w + i * w + j] =
                    mat.ptr<uchar>(i)[j * c + k] / 255.0f;
            }
        }
    }
    return im;
}

//----------------------------------------------------------------------------------------
void SortPlate(vector<bbox_t>& cur_bbox_vec)
{
    size_t i, j, n, bnd;
    size_t len=cur_bbox_vec.size();
    bbox_t tmp_box;
    float prb,ch_wd, ch_ht;
    double A, B, R;
    TLinRegression LinReg;

    if(len < 2) return;         //no need to investigate 1 character

    //check nr of chars
    while(len > 10){
        //get the lowest probability
        for(prb=1000.0, i=0;i<len;i++){
            if(cur_bbox_vec[i].prob < prb){ prb=cur_bbox_vec[i].prob; n=i;}
        }
        //delete the lowest
        cur_bbox_vec.erase(cur_bbox_vec.begin()+n);
        len=cur_bbox_vec.size();
    }

    //get average width and height of the characters
    for(ch_ht=ch_wd=0.0, i=0; i<len; i++){
        ch_wd+=cur_bbox_vec[i].w;
        ch_ht+=cur_bbox_vec[i].h;
    }
    ch_wd/=len; ch_ht/=len;

    if(len > 4){
        //get linear regression through all (x,y)
        for(i=0; i<len; i++){
            LinReg.Add(cur_bbox_vec[i].x, cur_bbox_vec[i].y);
        }
        LinReg.Execute(A,B,R);
        //now you can do a warp perspective if the skew is too large.
        //in that case, you have to run the ocr detection again.
        //here we see how well a single line fits all the characters.
        //if the standard deviation is high, you have one line of text
        //if the R is low, you have a two-line number plate.

//        cout << "A = " << A << "  B = " << B << "  R = " << R << endl;
    }
    else{
        R=1.0;  // with 4 or fewer characters, assume we got always one line.
    }

    if(R<0.08 && A>-1.0 && A<1.0){
        //two lines -> sort on y first
        for(i=0; i<len; i++){
            for(j=i+1; j<len; j++){
                if(cur_bbox_vec[j].y<cur_bbox_vec[i].y){
                    //swap
                    tmp_box=cur_bbox_vec[j];
                    cur_bbox_vec[j]=cur_bbox_vec[i];
                    cur_bbox_vec[i]=tmp_box;
                }
            }
        }

        //get the boundary between the first and second line.
        for(n=0, i=0; i<len-1; i++){
            j=cur_bbox_vec[i+1].y-cur_bbox_vec[i].y;
            if(j>n){ n=j; bnd=i+1; }
        }
        //sort the first line 0 < bnd
        SortSingleLine(cur_bbox_vec, ch_wd, ch_ht, 0, bnd);

        len=cur_bbox_vec.size();        //SortSingleLine can shorten the length of the vector
        //sort the second line bnd < len
        SortSingleLine(cur_bbox_vec, ch_wd, ch_ht, bnd, len);
    }
    else{
        //one line -> sort by x position
        SortSingleLine(cur_bbox_vec, ch_wd, ch_ht, 0, len);
    }
}
//----------------------------------------------------------------------------------------
static std::mutex mtx_mjpeg;

void send_mjpeg(cv::Mat& mat, int port, int timeout, int quality)
{
    try {
        std::lock_guard<std::mutex> lock(mtx_mjpeg);
        static MJPG_sender wri(port, timeout, quality);

        wri.write(mat);
    }
    catch (...) {
        cerr << " Error in send_mjpeg() function \n";
    }
}
//----------------------------------------------------------------------------------------
bool send_json_http(vector<bbox_t> cur_bbox_vec, vector<string> obj_names, string frame_id,
                    string filename = string(), int timeout = 400000, int port = 8070){
    string send_str;

    char *tmp_buf = (char *)calloc(1024, sizeof(char));
    if (!filename.empty()) {
        sprintf(tmp_buf, "{\n \"frame_id\":%s, \n \"filename\":\"%s\", \n \"objects\": [ \n", frame_id.c_str(), filename.c_str());
    }
    else {
        sprintf(tmp_buf, "{\n \"frame_id\":%s, \n \"objects\": [ \n", frame_id.c_str());
    }
    send_str = tmp_buf;
    free(tmp_buf);

    for (auto & i : cur_bbox_vec) {
        char *buf = (char *)calloc(2048, sizeof(char));

        sprintf(buf, "  {\"class_id\":%d, \"name\":\"%s\", \"absolute_coordinates\":{\"center_x\":%d, \"center_y\":%d, \"width\":%d, \"height\":%d}, \"confidence\":%f",
            i.obj_id, obj_names[i.obj_id].c_str(), i.x, i.y, i.w, i.h, i.prob);

        send_str += buf;

        if (!isnan(i.z_3d)) {
            sprintf(buf, "\n    , \"coordinates_in_meters\":{\"x_3d\":%.2f, \"y_3d\":%.2f, \"z_3d\":%.2f}",
                i.x_3d, i.y_3d, i.z_3d);
            send_str += buf;
        }

        send_str += "}\n";

        free(buf);
    }

    send_str += "\n ] \n}";

    if(Js.Json_Folder!="none"){
        ofstream Jfile(Js.Json_Folder+"/"+frame_id);
        Jfile << send_str;
        Jfile.close();
    }

    send_json_custom(send_str.c_str(), port, timeout);
    return true;
}
//----------------------------------------------------------------------------------------
void CropMat(cv::Mat& In, cv::Mat& Out) //checks the RoI parameters on forehand
{
    cv::Rect R;

    if(Js.RoiCrop.width  <= In.cols) R.width  = Js.RoiCrop.width;
    else                             R.width  = In.cols;

    if(Js.RoiCrop.height <= In.rows) R.height = Js.RoiCrop.height;
    else                             R.height = In.rows;

    if(Js.RoiCrop.x < 0 ) R.x=0;
    else{
        if((Js.RoiCrop.x+R.width) <= In.cols) R.x=Js.RoiCrop.x;
        else                                  R.x=In.cols-R.width;
    }

    if(Js.RoiCrop.y < 0 ) R.y=0;
    else{
        if((Js.RoiCrop.y+R.height) <= In.rows) R.y=Js.RoiCrop.y;
        else                                   R.y=In.rows-R.height;
    }

    Out = In(R);
    //important update the Js.RoiCrop as it is used as offset in the remaining code.
    //in fact you may overrule the config.json here.
    Js.RoiCrop = R;
}

void makeCurlRequest(const std::string& vehicle) {
    std::string apiKey = "api_key";
    std::string type = "entry";
    std::string url = "url";
    std::string curlCommand = "curl -X GET \"" + url + "\"";

    FILE* pipe = popen(curlCommand.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error: popen failed for " << vehicle << "\n";
        return;
    }
    char buffer[128];
    std::string result;
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != nullptr)
            result += buffer;
    }
    int curlResult = pclose(pipe);
    if (curlResult != 0) {
        std::cerr << "Error: Curl request failed for vehicle " << vehicle << "\n";
    } else if (result.find("Success") != std::string::npos) {
        std::cerr << "Success: GET request using cURL for " << vehicle << "\n";
    } else {
        std::cerr << "Server error for vehicle " << vehicle << "\n";
    }
}


// int ProducerThread(){

//     // Read the image path from the config
//     string image_path = "./images/car.jpg" ;// Adjust if different accessor is needed

//     if (image_path.empty()) {
//         cerr << "Error: Image path not specified in config.json" << endl;
//         return -1;
//     }

//     // Load the image
//     cv::Mat frame_full = cv::imread(image_path);
//     if (frame_full.empty()) {
//         cerr << "Error: Unable to read the image from " << image_path << endl;
//         return -1;
//     }

//     cv::Mat frame;
//     cv::Mat frame_full_render;
//     vector<bbox_t> result_ocr;

//     cout << "ALPR Version : " << Js.Version << endl;

//     // Create output directories if needed
//     Js.MakeFolders();

//     Detector CarNet(Js.Cstr + ".cfg", Js.Cstr + ".weights");
//     auto CarNames = objects_names_from_file(Js.Cstr + ".names");

//     Detector PlateNet(Js.Lstr + ".cfg", Js.Lstr + ".weights");
//     auto PlateNames = objects_names_from_file(Js.Lstr + ".names");

//     Detector OcrNet(Js.Ostr + ".cfg", Js.Ostr + ".weights");
//     auto OcrNames = objects_names_from_file(Js.Ostr + ".names");

//     frame_full_render = frame_full.clone();
//     CropMat(frame_full, frame);


//     // Draw crop borders
//     cv::rectangle(frame_full_render, Js.RoiCrop, cv::Scalar(0, 128, 255), 2);

//     // Detect cars
//     image_t image_car = mat_to_image(frame);
//     vector<bbox_t> result_car = CarNet.detect(image_car, Js.ThresCar);
//     delete[] image_car.data; // Clean up memory

//     unsigned int Wd = frame.cols, Ht = frame.rows;

//     for (auto& car : result_car) {
//         if ((100 * car.w > 95 * Wd) || (100 * car.h > 95 * Ht)) continue;

//         if ((car.w > 40) && (car.h > 40) && ((car.x + car.w) < Wd) && ((car.y + car.h) < Ht)) {
//             cv::Rect roi(car.x, car.y, car.w, car.h);
//             cv::Mat frame_car = frame(roi);

//             draw_vehicle(frame_full_render, car);

//             if (Js.Car_Folder != "none") {
//                 cv::imwrite(Js.Car_Folder + "/car_utc.png", frame_car);
//             }

//             // Detect plates
//             image_t image_plate = mat_to_image(frame_car);
//             vector<bbox_t> result_plate = PlateNet.detect(image_plate, Js.ThresPlate);
//             delete[] image_plate.data; // Clean up memory

//             for (auto& plate : result_plate) {
//                 if ((plate.w > 20) && (plate.h > 10) &&
//                     ((plate.x + 2 + plate.w) < frame_car.cols) &&
//                     ((plate.y + 2 + plate.h) < frame_car.rows)) {

//                     cv::Rect roi(plate.x, plate.y, plate.w + 2, plate.h + 2);
//                     cv::Mat frame_plate = frame_car(roi);

//                     draw_plate(frame_full_render, car, plate);

//                     if (Js.Plate_Folder != "none") {
//                         cv::imwrite(Js.Plate_Folder + "/plate_utc.png", frame_plate);
//                     }

//                     // Detect OCR
//                         result_ocr.clear();
//                         image_t image_ocr = mat_to_image(frame_plate);
//                         result_ocr = OcrNet.detect(image_ocr, Js.ThresOCR);
//                         delete[] image_ocr.data; // Clean up memory
//                         std::sort(result_ocr.begin(), result_ocr.end(), [](const bbox_t& a, const bbox_t& b) {
//                             return a.x < b.x;  // Sort by x-coordinate
//                         });
//                         std::string plateText;
//                         for (size_t i = 0; i < result_ocr.size(); i++) {
//                                 int obj_id = result_ocr[i].obj_id;
//                                 if (obj_id < OcrNames.size()) {
//                                     plateText += OcrNames[obj_id][0];  // Append detected character
//                                 } else {
//                                     plateText += '?';  // Handle invalid index
//                                 }
//                             }
//                         if (!plateText.empty()) {
//                                 std::unique_lock<std::mutex> lock(queueMutex);
//                                 if (textQueue.empty() || textQueue.back() != plateText) {
//                                     textQueue.push(plateText);
//                                     std::cout << "Extracted Text Pushed: [" << plateText << "], Queue Size = " << textQueue.size() << std::endl;
//                                     queueCondVar.notify_one();
//                                 } else {
//                                     std::cout << "Skipping duplicate OCR text: [" << plateText << "]" << std::endl;
//                                 }
//                             }

//                     if (Js.HeuristicsOn) {
//                         SortPlate(result_ocr);
//                     }

//                     if (Js.PrintOnCli) {
//                         print_result(result_ocr, OcrNames);
//                     }

//                     draw_ocr(frame_full_render, car, plate, result_ocr, OcrNames);
//                 }
//             }
//         }

//     }

//     if (Js.Render_Folder != "none") {
//         cv::imwrite(Js.Render_Folder + "/render_utc.png", frame_full_render);
//     }

//     if (Js.PrintOnRender) {
//         cv::imshow("Image Output", frame_full_render);
//         cv::waitKey(0);
//     }

//     return 0;
// }



int ProducerThread() {
    // Read the video path from the config
    std::string video_path = "video_path";  // Change this to your video file

    // Open the video file
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the video file: " << video_path << std::endl;
        return -1;
    }

    std::cout << "ALPR Version: " << Js.Version << std::endl;

    // Create output directories if needed
    Js.MakeFolders();

    Detector CarNet(Js.Cstr + ".cfg", Js.Cstr + ".weights");
    auto CarNames = objects_names_from_file(Js.Cstr + ".names");

    Detector PlateNet(Js.Lstr + ".cfg", Js.Lstr + ".weights");
    auto PlateNames = objects_names_from_file(Js.Lstr + ".names");

    Detector OcrNet(Js.Ostr + ".cfg", Js.Ostr + ".weights");
    auto OcrNames = objects_names_from_file(Js.Ostr + ".names");

    cv::Mat frame_full, frame_full_render, frame;
    std::vector<bbox_t> result_ocr;

    if (Js.PrintOnRender) {
        cv::namedWindow("Video Output", cv::WINDOW_NORMAL);
        cv::resizeWindow("Video Output", 640, 480);
    }

    int frame_counter = 0;
    int frame_skip = 5;

    while (true) {
        // Read a new frame from the video
        cap >> frame_full;
        if (frame_full.empty()) {
            std::cout << "End of video stream." << std::endl;
            break;  // Stop processing when video ends
        }

        frame_counter++;
        if (frame_counter % frame_skip != 0) {
        continue;  // Skip this frame
        }

        frame_full_render = frame_full.clone();
        CropMat(frame_full, frame);  // Crop the region of interest

        // Draw crop borders
        cv::rectangle(frame_full_render, Js.RoiCrop, cv::Scalar(0, 128, 255), 2);

        // Detect cars
        image_t image_car = mat_to_image(frame);
        std::vector<bbox_t> result_car = CarNet.detect(image_car, Js.ThresCar);
        delete[] image_car.data;  // Clean up memory

        unsigned int Wd = frame.cols, Ht = frame.rows;

        for (auto& car : result_car) {
            if ((100 * car.w > 95 * Wd) || (100 * car.h > 95 * Ht)) continue;
            if ((car.w > 40) && (car.h > 40) && ((car.x + car.w) < Wd) && ((car.y + car.h) < Ht)) {

                cv::Rect roi(car.x, car.y, car.w, car.h);
                cv::Mat frame_car = frame(roi);
                draw_vehicle(frame_full_render, car);

                if (Js.Car_Folder != "none") {
                    cv::imwrite(Js.Car_Folder + "/car_frame_" + std::to_string(cap.get(cv::CAP_PROP_POS_FRAMES)) + ".png", frame_car);
                }

                // Detect plates
                image_t image_plate = mat_to_image(frame_car);
                std::vector<bbox_t> result_plate = PlateNet.detect(image_plate, Js.ThresPlate);
                delete[] image_plate.data;  // Clean up memory

                for (auto& plate : result_plate) {
                    if ((plate.w > 20) && (plate.h > 10) &&
                        ((plate.x + 2 + plate.w) < frame_car.cols) &&
                        ((plate.y + 2 + plate.h) < frame_car.rows)) {

                        cv::Rect roi(plate.x, plate.y, plate.w + 2, plate.h + 2);
                        cv::Mat frame_plate = frame_car(roi);
                        draw_plate(frame_full_render, car, plate);

                        if (Js.Plate_Folder != "none") {
                            cv::imwrite(Js.Plate_Folder + "/plate_frame_" + std::to_string(cap.get(cv::CAP_PROP_POS_FRAMES)) + ".png", frame_plate);
                        }

                        // Detect OCR
                        image_t image_ocr = mat_to_image(frame_plate);
                        result_ocr = OcrNet.detect(image_ocr, Js.ThresOCR);
                        delete[] image_ocr.data;  // Clean up memory
                        std::sort(result_ocr.begin(), result_ocr.end(), [](const bbox_t& a, const bbox_t& b) {
                            return a.x < b.x;  // Sort by x-coordinate
                        });
                        std::string plateText;
                        for (size_t i = 0; i < result_ocr.size(); i++) {
                                int obj_id = result_ocr[i].obj_id;
                                if (obj_id < OcrNames.size()) {
                                    plateText += OcrNames[obj_id][0];  // Append detected character
                                } else {
                                    plateText += '?';  // Handle invalid index
                                }
                            }
                        if (!plateText.empty()) {
                                std::unique_lock<std::mutex> lock(queueMutex);
                                if (textQueue.empty() || textQueue.back() != plateText) {
                                    textQueue.push(plateText);
                                    std::cout << "Extracted Text Pushed: [" << plateText << "], Queue Size = " << textQueue.size() << std::endl;
                                    queueCondVar.notify_one();
                                } else {
                                    std::cout << "Skipping duplicate OCR text: [" << plateText << "]" << std::endl;
                                }
                            }

                        if (Js.HeuristicsOn) {
                            SortPlate(result_ocr);
                        }

                        if (Js.PrintOnCli) {
                            print_result(result_ocr, OcrNames);
                        }

                        draw_ocr(frame_full_render, car, plate, result_ocr, OcrNames);
                    }
                }
            }
        }

        if (Js.Render_Folder != "none") {
            cv::imwrite(Js.Render_Folder + "/render_frame_" + std::to_string(cap.get(cv::CAP_PROP_POS_FRAMES)) + ".png", frame_full_render);
        }

        if (Js.PrintOnRender) {
            cv::imshow("Image Output", frame_full_render);
            if (cv::waitKey(1) == 27) {  // Press 'ESC' to exit
                std::cout << "User terminated the process." << std::endl;
                break;
            }
        }
    }

    cap.release();  // Release video capture
    cv::destroyAllWindows();  // Close OpenCV windows
    return 0;
}




void consumerThread() {
    while (true) {
        std::string extractedText;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCondVar.wait(lock, [] { return !textQueue.empty(); });
            extractedText = textQueue.front();
            textQueue.pop();
        }
        if (processed_carplates.find(extractedText) == processed_carplates.end()) {
            makeCurlRequest(extractedText);
            processed_carplates.insert(extractedText);
        }
    }
}


//----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    Js.LoadFromFile("./config.json");

    bool success = Js.GetSettings();
    if (!success) {
        cerr << "Error: Failed to load settings from config.json" << endl;
        return -1;
    }
    //std::thread consumer(consumerThread);
    std::thread producer(ProducerThread);
    producer.join();
    //consumer.detach();


    return 0;
}

//--------------------------------------
