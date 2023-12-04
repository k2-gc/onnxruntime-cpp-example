#include <iostream>
#include <vector>
#include <optional>
#include <array>

#include <fstream>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


std::vector<std::string> split_string(std::string base_string, std::string delimiter){
    std::vector<std::string> splitted_strings;
    std::string word = "";
    for (int i = 0; i < base_string.length(); i++){
        const char tmp = base_string[i];
        std::string current_character = {tmp};
        if (current_character == delimiter){
            splitted_strings.push_back(word);
            word = "";
            continue;
        }
        word += current_character;
    }
    splitted_strings.push_back(word);
    return splitted_strings;
}

void show_info(Ort::Session &session){
    Ort::TypeInfo input_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();

    Ort::TypeInfo output_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();
    std::optional<Ort::AllocatedStringPtr> input_name;
    std::optional<Ort::AllocatedStringPtr> output_name;
    Ort::AllocatorWithDefaultOptions ort_alloc;
    input_name.emplace(session.GetInputNameAllocated(0, ort_alloc));
    output_name.emplace(session.GetOutputNameAllocated(0, ort_alloc));
    std::cout << "************ Model Info ************" << std::endl;
    std::cout << "Input Name: " << "'" << input_name->get() << "'" << std::endl;
    std::cout << "Input Shape: [";
    for (int i = 0; i < input_dims.size(); i++){
        std::cout << input_dims[i];
        if (i != input_dims.size() - 1){
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "Output Name: " << "'" << output_name->get() << "'" << std::endl;
    std::cout << "Output Shape: [";
    for (int i = 0; i < output_dims.size(); i++){
        std::cout << output_dims[i];
        if (i != output_dims.size() - 1){
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "************************************" << std::endl;
}


int main(void){
    std::string model_path = "./best.onnx";
    Ort::SessionOptions session_options;
    Ort::Env env;
    Ort::Session session = Ort::Session(env, model_path.c_str(), session_options);

    std::optional<Ort::AllocatedStringPtr> input_name;
    std::optional<Ort::AllocatedStringPtr> output_name;
    Ort::AllocatorWithDefaultOptions ort_alloc;
    input_name.emplace(session.GetInputNameAllocated(0, ort_alloc));
    output_name.emplace(session.GetOutputNameAllocated(0, ort_alloc));
    show_info(session);

    Ort::TypeInfo input_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
    Ort::TypeInfo output_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();

    const std::vector<int64_t> input_shape = { input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    const std::array<int64_t, 2> output_shape = { output_dims[0], output_dims[1]};

    int input_element_size = 1;
    int output_element_size = 1;
    for (int i = 0; i < input_dims.size(); i++){
        input_element_size *= input_dims[i];
    }
    for (int i = 0; i < output_dims.size(); i++){
        output_element_size *= output_dims[i];
    }

    std::vector<float> input_array(input_element_size);
    std::vector<float> output_array(output_element_size);
    const std::array<const char*, 1>input_name_array = {input_name->get()};
    const std::array<const char*, 1>output_name_array = {output_name->get()};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_array.data(), input_element_size, input_shape.data(), input_shape.size());
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_array.data(), output_element_size, output_shape.data(), output_shape.size());

    std::string image_list_txt_path = "../image_list.txt";
    std::ifstream f(image_list_txt_path);
    std::string line;
    while (std::getline(f, line))
    {
        const auto splitted_strings_vector = split_string(line, " ");
        const auto image_path = splitted_strings_vector[0];
        const auto ground_truth = splitted_strings_vector[1];
        cv::Mat input_image = cv::imread(image_path, cv::IMREAD_COLOR);

        cv::Mat float_img;
        input_image.convertTo(float_img, CV_32F);
        cv::resize(float_img, float_img, cv::Size(input_dims[2], input_dims[3]));
        
        // transposeND()
        std::vector<cv::Mat> each_channel;
        cv::split(float_img, each_channel);
        for (int i = 0; i < 3; i++){
            std::copy(each_channel[i].begin<float>(), each_channel[i].end<float>(), input_array.begin() + input_dims[2] * input_dims[3] * i);
        }

        // Inference
        session.Run(Ort::RunOptions{nullptr}, input_name_array.data(), &input_tensor, 1, output_name_array.data(), &output_tensor, 1);

        // Result
        const auto prediction_result = std::distance(output_array.begin(),std::max_element(output_array.begin(), output_array.end()));
        std::cout << "Ground Truth: " << ground_truth << ", Prediction: " << prediction_result << ", Score: " << output_array[prediction_result] << std::endl;

    }
    return 0;
}