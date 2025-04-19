// (everything above remains unchanged)...

using namespace std;
using namespace cv;
using namespace nlohmann;

atomic<bool> keep_running(true);
mutex queueMutex;
condition_variable queueCondVar;
queue<string> textQueue;
unordered_set<string> processed_carplates;

void ProducerThread() {
    // --- [Initialize networks & video] ---
    string car_cfg = Js.Car_cfg;
    string car_weights = Js.Car_weights;
    string car_names = Js.Car_names;
    string plate_cfg = Js.Plate_cfg;
    string plate_weights = Js.Plate_weights;
    string plate_names = Js.Plate_names;
    string ocr_cfg = Js.OCR_cfg;
    string ocr_weights = Js.OCR_weights;
    string ocr_names = Js.OCR_names;

    Detector CarNet(car_cfg, car_weights);
    Detector PlateNet(plate_cfg, plate_weights);
    Detector OcrNet(ocr_cfg, ocr_weights);
    vector<string> CarNames = objects_names_from_file(car_names);
    vector<string> PlateNames = objects_names_from_file(plate_names);
    vector<string> OcrNames = objects_names_from_file(ocr_names);

    VideoCapture cap(Js.VideoFile);
    if (!cap.isOpened()) {
        cerr << "Error opening video file: " << Js.VideoFile << endl;
        return;
    }

    Mat frame;
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);

    // --- [FPS calculation setup] ---
    auto last_time = chrono::steady_clock::now();
    float fps = 0.0;
    static int frame_count = 0;  // Frame counter for skipping frames

    while (keep_running) {
        cap >> frame;
        if (frame.empty()) break;

        // --- [Resize frame and skip frames] ---
        resize(frame, frame, Size(640, 480));  // Resize the frame
        if (frame_count % 5 != 0) {  // Skip every 5th frame
            frame_count++;
            continue;
        }
        frame_count++;

        Mat frame_full_render = frame.clone();
        vector<bbox_t> result_car = CarNet.detect(mat_to_image(frame), Js.ThresCar);

        for (auto& car : result_car) {
            if (CarNames[car.obj_id] == "car") {
                Rect roi_car(car.x, car.y, car.w, car.h);
                Mat frame_car = frame(roi_car);
                draw_car(frame_full_render, car);

                vector<bbox_t> result_plate = PlateNet.detect(mat_to_image(frame_car), Js.ThresPlate);
                for (auto& plate : result_plate) {
                    if (PlateNames[plate.obj_id] == "plate") {
                        Rect roi_plate(plate.x, plate.y, plate.w + 2, plate.h + 2);
                        Mat frame_plate = frame_car(roi_plate);
                        draw_plate(frame_full_render, car, plate);

                        if (Js.Plate_Folder != "none") {
                            imwrite(Js.Plate_Folder + "/plate_frame_" + to_string((int)cap.get(CAP_PROP_POS_FRAMES)) + ".png", frame_plate);
                        }

                        image_t image_ocr = mat_to_image(frame_plate);
                        vector<bbox_t> result_ocr = OcrNet.detect(image_ocr, Js.ThresOCR);
                        delete[] image_ocr.data;

                        if (result_ocr.size() >= 4) {
                            SortPlate(result_ocr);
                            draw_ocr(frame_full_render, car, plate, result_ocr, OcrNames);

                            string plate_text;
                            for (auto& c : result_ocr) {
                                if (c.obj_id < OcrNames.size()) {
                                    plate_text += OcrNames[c.obj_id];
                                }
                            }

                            if (!plate_text.empty()) {
                                unique_lock<mutex> lock(queueMutex);
                                if (processed_carplates.find(plate_text) == processed_carplates.end()) {
                                    processed_carplates.insert(plate_text);
                                    textQueue.push(plate_text);
                                    queueCondVar.notify_one();
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- [Calculate and draw FPS] ---
        auto now = chrono::steady_clock::now();
        float elapsed = chrono::duration<float>(now - last_time).count();
        last_time = now;
        fps = 0.9f * fps + 0.1f * (1.0f / elapsed); // smooth fps

        string fps_text = "FPS: " + to_string(int(fps));

        // --- [Draw FPS on the top-left corner of the frame] ---
        putText(frame_full_render, fps_text, Point(10, 70), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 255, 0), 3, LINE_AA); // Larger font size, thicker text

        // --- [Show output] ---
        imshow("Video Output", frame_full_render);  // Renamed window title to "Video Output"

        if (waitKey(1) == 27) {
            keep_running = false;
        }
    }

    cap.release();
    destroyAllWindows();
    keep_running = false;
    queueCondVar.notify_all();
}
