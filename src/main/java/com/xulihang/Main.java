package com.xulihang;

import ai.onnxruntime.OrtException;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class Main {
    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("Hello world!");
        ImageLanguageClassfier classifier =
                new ImageLanguageClassfier("inference.onnx");
        Mat img = Imgcodecs.imread("zh.jpg");
        // 预测图像
        ImageLanguageClassfier.PredictionResult result =
                classifier.predict(img, 1);
        classifier.printResult(result);
        System.out.println(result.getTimeStats().getLoadMs());
        System.out.println(result.getTimeStats().getInferenceMs());
        System.out.println(result.getTimeStats().getTotalMs());
        System.out.println(result.getPredictedLanguage());
    }
}