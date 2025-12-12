package com.xulihang;

import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.stream.Collectors;

public class ImageLanguageClassfier {

    // 语言标签定义
    private static final Map<Integer, String> LANGUAGE_LABELS = new HashMap<Integer, String>() {{
        put(0, "Arabic");
        put(1, "chinese_cht");
        put(2, "cyrillic");
        put(3, "devanagari");
        put(4, "Japanese");
        put(5, "ka");
        put(6, "Korean");
        put(7, "ta");
        put(8, "te");
        put(9, "Latin");
    }};

    // 语言中文名称
    private static final Map<Integer, String> LANGUAGE_NAMES = new HashMap<Integer, String>() {{
        put(0, "阿拉伯语");
        put(1, "繁体中文");
        put(2, "西里尔文");
        put(3, "天城文");
        put(4, "日语");
        put(5, "卡纳达文");
        put(6, "韩语");
        put(7, "泰米尔文");
        put(8, "泰卢固文");
        put(9, "拉丁文");
    }};

    private OrtSession session;
    private String inputName;
    private long[] inputShape;
    private String outputName;

    /**
     * 初始化语言分类ONNX推理器
     * @param onnxPath ONNX模型文件路径
     */
    public ImageLanguageClassfier(String onnxPath) throws OrtException {
        System.out.println("=".repeat(50));
        System.out.println("正在加载ONNX模型...");

        // 创建ONNX Runtime环境
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

        // 创建会话
        this.session = env.createSession(onnxPath, sessionOptions);

        // 获取输入输出信息
        NodeInfo inputInfo = session.getInputInfo().values().iterator().next();
        this.inputName = inputInfo.getName();
        TensorInfo inputTensorInfo = (TensorInfo) inputInfo.getInfo();
        this.inputShape = inputTensorInfo.getShape();

        NodeInfo outputInfo = session.getOutputInfo().values().iterator().next();
        this.outputName = outputInfo.getName();

        System.out.println("ONNX模型加载成功!");
        System.out.println("模型输入: " + inputName + ", 形状: " + Arrays.toString(inputShape));
        System.out.println("模型输出: " + outputName);
        System.out.println("支持的分类: " + LANGUAGE_LABELS.size() + " 种语言");
        System.out.println("=".repeat(50));
    }

    /**
     * 预处理图像
     */
    private OnnxTensor preprocess(Mat image) throws OrtException {
        // 转换为RGB (OpenCV默认是BGR)
        Mat rgb = new Mat();
        Imgproc.cvtColor(image, rgb, Imgproc.COLOR_BGR2RGB);

        // 调整大小到160x80 (根据Python代码)
        Mat resized = new Mat();
        Imgproc.resize(rgb, resized, new Size(160, 80));

        // 转换为float并归一化
        resized.convertTo(resized, CvType.CV_32FC3, 1.0 / 255.0);

        // ImageNet归一化参数
        Scalar mean = new Scalar(0.485, 0.456, 0.406);
        Scalar std = new Scalar(0.229, 0.224, 0.225);

        // 分割通道并归一化
        List<Mat> channels = new ArrayList<>();
        Core.split(resized, channels);

        for (int i = 0; i < channels.size(); i++) {
            Core.subtract(channels.get(i), new Scalar(mean.val[i]), channels.get(i));
            Core.divide(channels.get(i), new Scalar(std.val[i]), channels.get(i));
        }

        // 合并通道
        Core.merge(channels, resized);

        // 准备ONNX输入 (NCHW格式)
        int height = resized.rows();
        int width = resized.cols();
        int channelsNum = resized.channels();

        // 创建FloatBuffer
        float[] floatArray = new float[1 * channelsNum * height * width];
        float[] data = new float[height * width * channelsNum];
        resized.get(0, 0, data);

        // 转换为NCHW格式
        int idx = 0;
        for (int c = 0; c < channelsNum; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int srcIdx = h * width * channelsNum + w * channelsNum + c;
                    floatArray[idx++] = data[srcIdx];
                }
            }
        }

        // 创建ONNX Tensor
        long[] shape = {1, channelsNum, height, width};
        FloatBuffer buffer = FloatBuffer.wrap(floatArray);

        return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), buffer, shape);
    }

    /**
     * 从文件加载图像
     */
    private Mat loadImage(String imagePath) throws IOException {
        File file = new File(imagePath);
        if (!file.exists()) {
            throw new IOException("图像文件不存在: " + imagePath);
        }

        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            // 如果OpenCV无法读取，尝试使用ImageIO
            BufferedImage bufferedImage = ImageIO.read(file);
            if (bufferedImage == null) {
                throw new IOException("无法加载图像: " + imagePath);
            }

            // 将BufferedImage转换为Mat
            image = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(),
                    CvType.CV_8UC3);
            byte[] data = ((java.awt.image.DataBufferByte) bufferedImage.getRaster()
                    .getDataBuffer()).getData();
            image.put(0, 0, data);

            // 如果是ARGB，需要转换为RGB
            if (bufferedImage.getType() == BufferedImage.TYPE_4BYTE_ABGR ||
                    bufferedImage.getType() == BufferedImage.TYPE_INT_ARGB) {
                Imgproc.cvtColor(image, image, Imgproc.COLOR_BGRA2BGR);
            }
        }

        return image;
    }

    /**
     * 从BufferedImage加载图像
     */
    private Mat loadImage(BufferedImage bufferedImage) {
        Mat image = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(),
                CvType.CV_8UC3);

        byte[] data = ((java.awt.image.DataBufferByte) bufferedImage.getRaster()
                .getDataBuffer()).getData();
        image.put(0, 0, data);

        // 如果是ARGB，需要转换为RGB
        if (bufferedImage.getType() == BufferedImage.TYPE_4BYTE_ABGR ||
                bufferedImage.getType() == BufferedImage.TYPE_INT_ARGB) {
            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGRA2BGR);
        }

        return image;
    }

    /**
     * 预测图像的语言类型
     */
    public PredictionResult predict(String imagePath, int topK) throws Exception {
        // 加载图像
        Mat image = loadImage(imagePath);
        return predict(image, topK);
    }

    private List<Float> floatArrayToList(float[] array) {
        List<Float> list = new ArrayList<>(array.length);
        for (float value : array) {
            list.add(value);
        }
        return list;
    }
    /**
     * 预测图像的语言类型 (使用Mat对象)
     */
    public PredictionResult predict(Mat image, int topK) throws Exception {
        long startTime = System.currentTimeMillis();

        // 预处理
        OnnxTensor inputTensor = preprocess(image);
        long preprocessTime = System.currentTimeMillis();

        // 推理
        OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));
        long inferenceTime = System.currentTimeMillis();

        // 获取输出
        float[][] predictions = ((float[][]) result.get(0).getValue());
        float[] scores = predictions[0];

        // 获取top-k预测结果
        List<Integer> topIndices = getTopKIndices(scores, topK);

        // 构建结果
        List<Prediction> topPredictions = new ArrayList<>();
        for (int i = 0; i < topIndices.size(); i++) {
            int idx = topIndices.get(i);
            float confidence = scores[idx];

            Prediction pred = new Prediction();
            pred.setClassId(idx);
            pred.setLabel(LANGUAGE_LABELS.getOrDefault(idx, "未知(" + idx + ")"));
            pred.setLabelCn(LANGUAGE_NAMES.getOrDefault(idx, "未知"));
            pred.setConfidence(confidence);
            pred.setConfidencePercent(String.format("%.2f%%", confidence * 100));

            topPredictions.add(pred);
        }

        // 时间统计
        TimeStats timeStats = new TimeStats();
        timeStats.setPreprocessMs(preprocessTime - startTime);
        timeStats.setInferenceMs(inferenceTime - preprocessTime);
        timeStats.setTotalMs(inferenceTime - startTime);

        // 返回结果
        PredictionResult predictionResult = new PredictionResult();
        predictionResult.setTopPredictions(topPredictions);
        predictionResult.setAllScores(floatArrayToList(scores));
        predictionResult.setTimeStats(timeStats);
        predictionResult.setPredictedLanguage(topPredictions.get(0).getLabel());
        predictionResult.setPredictedLanguageCn(topPredictions.get(0).getLabelCn());
        predictionResult.setConfidence(topPredictions.get(0).getConfidence());

        // 清理资源
        inputTensor.close();
        result.close();

        return predictionResult;
    }

    /**
     * 获取top-k索引
     */
    private List<Integer> getTopKIndices(float[] scores, int k) {
        List<Map.Entry<Integer, Float>> entries = new ArrayList<>();
        for (int i = 0; i < scores.length; i++) {
            entries.add(new AbstractMap.SimpleEntry<>(i, scores[i]));
        }

        // 按置信度降序排序
        entries.sort((a, b) -> Float.compare(b.getValue(), a.getValue()));

        // 取前k个
        k = Math.min(k, entries.size());
        List<Integer> topIndices = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            topIndices.add(entries.get(i).getKey());
        }

        return topIndices;
    }

    /**
     * 批量预测
     */
    public List<PredictionResult> predictBatch(List<String> imagePaths, int topK) throws Exception {
        List<PredictionResult> results = new ArrayList<>();
        long totalTime = 0;

        for (int i = 0; i < imagePaths.size(); i++) {
            System.out.printf("处理第 %d/%d 张图像...%n", i + 1, imagePaths.size());

            try {
                PredictionResult result = predict(imagePaths.get(i), topK);
                results.add(result);
                totalTime += result.getTimeStats().getTotalMs();
            } catch (Exception e) {
                System.out.printf("图像 %d 处理失败: %s%n", i + 1, e.getMessage());
                PredictionResult errorResult = new PredictionResult();
                errorResult.setError(e.getMessage());
                results.add(errorResult);
            }
        }

        if (!results.isEmpty()) {
            double avgTime = totalTime / (double) results.size();
            System.out.printf("%n批量处理完成，平均处理时间: %.1fms/张%n", avgTime);
        }

        return results;
    }

    /**
     * 打印预测结果
     */
    public void printResult(PredictionResult result) {
        if (result.getError() != null) {
            System.out.printf("错误: %s%n", result.getError());
            return;
        }

        System.out.println("\n" + "=".repeat(50));
        System.out.println("语言分类结果");
        System.out.println("=".repeat(50));

        System.out.printf("%n预测语言: %s (%s)%n",
                result.getPredictedLanguageCn(), result.getPredictedLanguage());
        System.out.printf("置信度: %.2f%%%n", result.getConfidence() * 100);

        System.out.println("\nTop 预测结果:");
        System.out.println("-".repeat(60));
        System.out.printf("%-4s %-8s %-15s %-10s %-12s%n",
                "排名", "语言ID", "语言(英文)", "语言(中文)", "置信度");
        System.out.println("-".repeat(60));

        for (int i = 0; i < result.getTopPredictions().size(); i++) {
            Prediction pred = result.getTopPredictions().get(i);
            System.out.printf("%-4d %-8d %-15s %-10s %-12s%n",
                    i + 1, pred.getClassId(), pred.getLabel(),
                    pred.getLabelCn(), pred.getConfidencePercent());
        }

        System.out.println("\n时间统计:");
        TimeStats timeStats = result.getTimeStats();
        if (timeStats.getLoadMs() > 0) {
            System.out.printf("  加载: %.1fms%n", timeStats.getLoadMs());
        }
        System.out.printf("  预处理: %.1fms%n", timeStats.getPreprocessMs());
        System.out.printf("  推理: %.1fms%n", timeStats.getInferenceMs());
        System.out.printf("  总计: %.1fms%n", timeStats.getTotalMs());
        System.out.println("=".repeat(50));
    }

    /**
     * 关闭会话
     */
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }

    // 数据类
    public static class Prediction {
        private int classId;
        private String label;
        private String labelCn;
        private float confidence;
        private String confidencePercent;

        // getters and setters
        public int getClassId() { return classId; }
        public void setClassId(int classId) { this.classId = classId; }

        public String getLabel() { return label; }
        public void setLabel(String label) { this.label = label; }

        public String getLabelCn() { return labelCn; }
        public void setLabelCn(String labelCn) { this.labelCn = labelCn; }

        public float getConfidence() { return confidence; }
        public void setConfidence(float confidence) { this.confidence = confidence; }

        public String getConfidencePercent() { return confidencePercent; }
        public void setConfidencePercent(String confidencePercent) {
            this.confidencePercent = confidencePercent;
        }
    }

    public static class TimeStats {
        private double loadMs;
        private double preprocessMs;
        private double inferenceMs;
        private double totalMs;

        // getters and setters
        public double getLoadMs() { return loadMs; }
        public void setLoadMs(double loadMs) { this.loadMs = loadMs; }

        public double getPreprocessMs() { return preprocessMs; }
        public void setPreprocessMs(double preprocessMs) { this.preprocessMs = preprocessMs; }

        public double getInferenceMs() { return inferenceMs; }
        public void setInferenceMs(double inferenceMs) { this.inferenceMs = inferenceMs; }

        public double getTotalMs() { return totalMs; }
        public void setTotalMs(double totalMs) { this.totalMs = totalMs; }
    }

    public static class PredictionResult {
        private List<Prediction> topPredictions;
        private List<Float> allScores;
        private TimeStats timeStats;
        private String predictedLanguage;
        private String predictedLanguageCn;
        private float confidence;
        private String error;

        // getters and setters
        public List<Prediction> getTopPredictions() { return topPredictions; }
        public void setTopPredictions(List<Prediction> topPredictions) {
            this.topPredictions = topPredictions;
        }

        public List<Float> getAllScores() { return allScores; }
        public void setAllScores(List<Float> allScores) { this.allScores = allScores; }

        public TimeStats getTimeStats() { return timeStats; }
        public void setTimeStats(TimeStats timeStats) { this.timeStats = timeStats; }

        public String getPredictedLanguage() { return predictedLanguage; }
        public void setPredictedLanguage(String predictedLanguage) {
            this.predictedLanguage = predictedLanguage;
        }

        public String getPredictedLanguageCn() { return predictedLanguageCn; }
        public void setPredictedLanguageCn(String predictedLanguageCn) {
            this.predictedLanguageCn = predictedLanguageCn;
        }

        public float getConfidence() { return confidence; }
        public void setConfidence(float confidence) { this.confidence = confidence; }

        public String getError() { return error; }
        public void setError(String error) { this.error = error; }
    }
}
