package com.qizhenghao.imageprocessing;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import static org.opencv.core.CvType.CV_32FC3;

public class MainActivity extends AppCompatActivity {

    public int[] resIds = new int[]{R.drawable.img_9374, R.drawable.img_9524, R.drawable.img_9530, R.drawable.img_9544, R.drawable.img_9549, R.drawable.img_9568, R.drawable.img_9571, R.drawable.img_9574};
    public int srcIndex = 0;

    TextView tv;
    Button btnProcess0;
    Button btnReset;
    Button btnProcess1;
    Button btnProcess2;
    Button btnProcess3;
    Button btnProcess4;
    Button btnProcess5;
    Button btnProcess6;
    Button btnProcess7;
    Button btnProcess8;
    Button btnProcess9;
    Bitmap srcBitmap;
    Bitmap resultBitmap;
    Bitmap resultBitmapTemp;
    ImageView originIv, resultIv;

    static {
        System.loadLibrary("native-lib");
    }

    private Mat originMat, lastResultMat;
    private Point rightTopPoint, leftTopPoint, leftBottomPoint, rightBottomPoint;
    private Point[] originPoints, transformPoints;

    private MainActivity mActivity;
    private EditText blurEdit;
    private EditText threshEdit;
    private EditText cannyEdit;

    List<MatOfPoint> contourList = new ArrayList<>();
    private MatOfPoint2f targetApproxCurve = null;
    double xMin = 0, xMax = 0, yMin = 0, yMax = 0;
    private int width, height;

    public native String stringFromJNI();

    //OpenCV库加载并初始化成功后的回调函数
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS:
                    Log.i("bruce", "成功加载");
                    initMat();
                    break;
                default:
                    super.onManagerConnected(status);
                    Log.i("bruce", "加载失败");
                    break;
            }

        }
    };

    private void initMat() {
        originMat = new Mat();
        lastResultMat = new Mat();
        Utils.bitmapToMat(srcBitmap, originMat);
        originMat.copyTo(lastResultMat);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mActivity = this;
        setContentView(R.layout.activity_main);
        initUI();


//        tv.setText(stringFromJNI());

        setOnClickListeners();
    }

    public void initUI() {
        tv = findViewById(R.id.sample_text);
        blurEdit = findViewById(R.id.blur_edit);
        threshEdit = findViewById(R.id.binary_edit);
        cannyEdit = findViewById(R.id.canny_edit);
        resultIv = findViewById(R.id.reslut_iv);
        originIv = findViewById(R.id.origin_iv);
        btnProcess0 = findViewById(R.id.btn_process);
        btnReset = findViewById(R.id.btn_reset);
        btnProcess1 = findViewById(R.id.btn_gray_process1);
        btnProcess2 = findViewById(R.id.btn_gray_process2);
        btnProcess3 = findViewById(R.id.btn_gray_process3);
        btnProcess4 = findViewById(R.id.btn_gray_process4);
        btnProcess5 = findViewById(R.id.btn_gray_process5);
        btnProcess6 = findViewById(R.id.btn_gray_process6);
        btnProcess7 = findViewById(R.id.btn_gray_process7);
        btnProcess8 = findViewById(R.id.btn_gray_process8);
        btnProcess9 = findViewById(R.id.btn_gray_process9);
    }

    private void setOnClickListeners() {
        srcBitmap = BitmapFactory.decodeResource(getResources(), resIds[srcIndex % resIds.length]);
        resultBitmap = Bitmap.createBitmap(srcBitmap.getWidth(), srcBitmap.getHeight(), Bitmap.Config.RGB_565);
        originIv.setImageBitmap(srcBitmap);
        btnReset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                srcBitmap = BitmapFactory.decodeResource(getResources(), resIds[srcIndex % resIds.length]);
                resultBitmap = Bitmap.createBitmap(srcBitmap.getWidth(), srcBitmap.getHeight(), Bitmap.Config.RGB_565);
                originIv.setImageBitmap(srcBitmap);
                initMat();
                resultIv.setImageBitmap(null);
            }
        });

        originIv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                srcIndex++;
                srcBitmap = BitmapFactory.decodeResource(getResources(), resIds[srcIndex % resIds.length]);
                originIv.setImageBitmap(srcBitmap);
                initMat();
                resultIv.setImageBitmap(null);
            }
        });
        btnProcess0.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final int blurSize = TextUtils.isEmpty(blurEdit.getText().toString()) ? 19 : Integer.valueOf(blurEdit.getText().toString());
                if (blurSize % 2 == 0) {
                    Toast.makeText(mActivity, "滤波只能是奇数", Toast.LENGTH_LONG).show();
                    return;
                }
                final int threshSize = TextUtils.isEmpty(threshEdit.getText().toString()) ? 70 : Integer.valueOf(threshEdit.getText().toString());
                final int thresholdSize = TextUtils.isEmpty(cannyEdit.getText().toString()) ? 3 : Integer.valueOf(cannyEdit.getText().toString());
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        Mat originMat = new Mat();

                        Mat GaussianMat = new Mat();
                        Mat medianBlurMat = new Mat();
                        Mat cvtColorMat = new Mat();
                        Mat thresholdMat = new Mat();
                        Mat cannyMat = new Mat();

                        Utils.bitmapToMat(srcBitmap, originMat);
                        long timeStamp = System.currentTimeMillis();


                        Imgproc.GaussianBlur(originMat, GaussianMat, new Size(blurSize, blurSize), 0, 0);
                        lastResultMat = GaussianMat;

//                        Imgproc.medianBlur(originMat, medianBlurMat, 19);
//                        lastResultMat = medianBlurMat;

                        Imgproc.cvtColor(lastResultMat, cvtColorMat, Imgproc.COLOR_RGB2GRAY);
                        lastResultMat = cvtColorMat;

                        Imgproc.threshold(lastResultMat, thresholdMat, threshSize, 255, Imgproc.THRESH_BINARY);
                        lastResultMat = thresholdMat;

                        Imgproc.Canny(lastResultMat, cannyMat, thresholdSize, thresholdSize * 2);
                        lastResultMat = cannyMat;

                        List<MatOfPoint> contourList = new ArrayList<>();
                        Mat hierarchyMat = new Mat();
                        Imgproc.findContours(lastResultMat, contourList, hierarchyMat, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

                        Log.d("Bruce", "contourList.size() = " + contourList.size());
                        rightTopPoint = new Point();
                        leftTopPoint = new Point();
                        leftBottomPoint = new Point();
                        rightBottomPoint = new Point();
                        originPoints = null;
                        if (contourList.size() > 0) {
                            Collections.sort(contourList, new Comparator<MatOfPoint>() {
                                @Override
                                public int compare(MatOfPoint o1, MatOfPoint o2) {
                                    return Imgproc.contourArea(o1) > Imgproc.contourArea(o2) ? 1 : 0;
                                }
                            });

                            targetApproxCurve = null;
                            for (MatOfPoint point : contourList) {
                                MatOfPoint2f curve = new MatOfPoint2f(point.toArray());
                                double peri = Imgproc.arcLength(curve, true);
                                double area = Imgproc.contourArea(curve);
                                if ((area > 2000000 && area < 11000000) || peri > 6000) {
                                    MatOfPoint2f approxCurve = new MatOfPoint2f();
                                    Imgproc.approxPolyDP(curve, approxCurve, 15, true);
                                    int length = approxCurve.toArray().length;
                                    if (length == 4 || length == 3) {
                                        targetApproxCurve = approxCurve;
                                        break;
                                    }
                                }
                            }

                            Log.d("Bruce", "targetApproxCurve = " + targetApproxCurve);
                            if (targetApproxCurve != null) {
                                Mat approxMat = new Mat();
                                approxMat.create(lastResultMat.rows(), lastResultMat.cols(), CvType.CV_8UC3);
                                //Drawing corners on a new image
                                xMin = 0;
                                xMax = 0;
                                yMin = 0;
                                yMax = 0;
                                Log.d("Bruce", targetApproxCurve.cols() + ", " + targetApproxCurve.rows());
                                originPoints = new Point[targetApproxCurve.rows()];
                                StringBuilder stringBuilder = new StringBuilder("四角定位：");
                                for (int i = 0; i < targetApproxCurve.rows(); i++) {
                                    double[] vec = targetApproxCurve.get(i, 0);
                                    double x = vec[0], y = vec[1];
                                    if (xMin > x || xMin == 0) xMin = x;
                                    if (xMax < x) xMax = x;
                                    if (yMin > y || yMin == 0) yMin = y;
                                    if (yMax < y) yMax = y;
                                    Point point = new Point(x, y);
                                    originPoints[i] = point;
                                    Imgproc.circle(approxMat, point, 100, new Scalar(255, 255, 255), 10);
                                    stringBuilder.append(x).append(", ").append(y).append("    ");
                                }
                                Log.d("Bruce", stringBuilder.toString());
                                showToast(stringBuilder.toString());
                            } else {
                                showToast("没有找到轮廓-1");
                                return;
                            }
                        } else {
                            showToast("没有找到轮廓-2");
                            return;
                        }

                        width = (int) (xMax - xMin);
                        height = (int) (yMax - yMin);
                        Rect rect = new Rect((int) xMin, (int) yMin, (int) (xMax - xMin), (int) (yMax - yMin));
                        lastResultMat = originMat.submat(rect);
                        transformPoints = new Point[originPoints.length];
                        for (int i = 0; i < originPoints.length; i++) {
                            Point p = originPoints[i];
                            double x = p.x - xMin, y = p.y - yMin;
                            originPoints[i] = new Point(x, y);
                            transformPoints[i] = new Point(x < 500 ? 0 : width, y < 500 ? 0 : height);
                            Log.d("Bruce", "x , y = " + x + ", " + y + "      " + transformPoints[i].x + ", " + transformPoints[i].y);
                        }

                        Mat resultMat = new Mat();
                        resultMat.create(lastResultMat.rows(), lastResultMat.cols(), CvType.CV_8UC3);
                        Mat src = lastResultMat, dst = new Mat();
                        List<Point> listSrcs = Arrays.asList(originPoints);
                        Mat srcPoints = Converters.vector_Point_to_Mat(listSrcs, CvType.CV_32F);
                        List<Point> listDsts = Arrays.asList(transformPoints);
                        Mat dstPoints = Converters.vector_Point_to_Mat(listDsts, CvType.CV_32F);
                        Mat perspectiveMat = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
                        Imgproc.warpPerspective(src, dst, perspectiveMat, src.size(), Imgproc.INTER_LINEAR);
                        lastResultMat = dst;


                        final String spendTime = String.valueOf(System.currentTimeMillis() - timeStamp);
                        final Bitmap processedImage = Bitmap.createBitmap(lastResultMat.cols(), lastResultMat.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(lastResultMat, processedImage);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                btnProcess0.setText(String.format("一键顺序执行：%sms", spendTime));
                                resultIv.setImageBitmap(processedImage);
                            }
                        });
                    }
                }).start();

            }
        });
        btnProcess1.setText("GaussianB");
        btnProcess1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final int blurSize = TextUtils.isEmpty(blurEdit.getText().toString()) ? 19 : Integer.valueOf(blurEdit.getText().toString());
                if (blurSize % 2 == 0) {
                    Toast.makeText(mActivity, "只能是奇数", Toast.LENGTH_LONG).show();
                    return;
                }

                Mat resultMat = new Mat();
                Imgproc.GaussianBlur(lastResultMat, resultMat, new Size(blurSize, blurSize), 0, 0);
//                Imgproc.medianBlur(originMat, resultMat, 1);
                lastResultMat = resultMat;
                Utils.matToBitmap(resultMat, resultBitmap);
                resultIv.setImageBitmap(resultBitmap);
            }
        });
        btnProcess2.setText("cvtColor");
        btnProcess2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Mat resultMat = new Mat();
                Imgproc.cvtColor(lastResultMat, resultMat, Imgproc.COLOR_RGB2GRAY);//rgbMat to gray grayMat
                lastResultMat = resultMat;
                Utils.matToBitmap(resultMat, resultBitmap);
                resultIv.setImageBitmap(resultBitmap);
            }
        });
        btnProcess3.setText("threshold");
        btnProcess3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final int threshSize = TextUtils.isEmpty(threshEdit.getText().toString()) ? 70 : Integer.valueOf(threshEdit.getText().toString());
                Mat resultMat = new Mat();
                Imgproc.threshold(lastResultMat, resultMat, threshSize, 255, Imgproc.THRESH_BINARY);
                lastResultMat = resultMat;
                Utils.matToBitmap(resultMat, resultBitmap);
                resultIv.setImageBitmap(resultBitmap);
            }
        });
//        Imgproc.morphologyEx(getLastResultMat(), resultMat, Imgproc.MORPH_GRADIENT, Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(15, 15)));
        btnProcess4.setText("Canny");
        btnProcess4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final int thresholdSize = TextUtils.isEmpty(cannyEdit.getText().toString()) ? 3 : Integer.valueOf(cannyEdit.getText().toString());
                Mat resultMat = new Mat();
                Imgproc.Canny(lastResultMat, resultMat, thresholdSize, thresholdSize * 2);
                lastResultMat = resultMat;
                Utils.matToBitmap(resultMat, resultBitmap);
                resultIv.setImageBitmap(resultBitmap);
            }
        });

        btnProcess5.setText("findContours");
        btnProcess5.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Mat hierarchyMat = new Mat();
                contourList.clear();
                Imgproc.findContours(lastResultMat, contourList, hierarchyMat, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                //Drawing contours on a new image
                Mat contours = new Mat();
                contours.create(lastResultMat.rows(), lastResultMat.cols(), CvType.CV_8UC3);
                Random r = new Random();
                for (int i = 0; i < contourList.size(); i++) {
                    Imgproc.drawContours(contours, contourList, i, new Scalar(r.nextInt(255), r.nextInt(255), r.nextInt(255)), -1);
                }
                Utils.matToBitmap(contours, resultBitmap);
                resultIv.setImageBitmap(resultBitmap);
            }
        });
        btnProcess6.setText("四角定位");
        btnProcess6.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d("Bruce", "contourList.size() = " + contourList.size());
                rightTopPoint = new Point();
                leftTopPoint = new Point();
                leftBottomPoint = new Point();
                rightBottomPoint = new Point();
                originPoints = null;
                if (contourList.size() > 0) {
                    Collections.sort(contourList, new Comparator<MatOfPoint>() {
                        @Override
                        public int compare(MatOfPoint o1, MatOfPoint o2) {
                            return Imgproc.contourArea(o1) > Imgproc.contourArea(o2) ? 1 : 0;
                        }
                    });

                    targetApproxCurve = null;
                    for (MatOfPoint point : contourList) {
                        MatOfPoint2f curve = new MatOfPoint2f(point.toArray());
                        double peri = Imgproc.arcLength(curve, true);
                        double area = Imgproc.contourArea(curve);
                        if ((area > 2000000 && area < 11000000) || peri > 6000) {
                            MatOfPoint2f approxCurve = new MatOfPoint2f();
                            Imgproc.approxPolyDP(curve, approxCurve, 15, true);
                            int length = approxCurve.toArray().length;
                            if (length == 4 || length == 3) {
                                targetApproxCurve = approxCurve;
                                break;
                            }
                        }
                    }

                    Log.d("Bruce", "targetApproxCurve = " + targetApproxCurve);
                    if (targetApproxCurve != null) {
                        Mat approxMat = new Mat();
                        approxMat.create(lastResultMat.rows(), lastResultMat.cols(), CvType.CV_8UC3);
                        //Drawing corners on a new image
                        xMin = 0;
                        xMax = 0;
                        yMin = 0;
                        yMax = 0;
                        Log.d("Bruce", targetApproxCurve.cols() + ", " + targetApproxCurve.rows());
                        originPoints = new Point[targetApproxCurve.rows()];
                        for (int i = 0; i < targetApproxCurve.rows(); i++) {
                            double[] vec = targetApproxCurve.get(i, 0);
                            double x = vec[0], y = vec[1];
                            if (xMin > x || xMin == 0) xMin = x;
                            if (xMax < x) xMax = x;
                            if (yMin > y || yMin == 0) yMin = y;
                            if (yMax < y) yMax = y;
                            Point point = new Point(x, y);
                            originPoints[i] = point;
                            Imgproc.circle(approxMat, point, 100, new Scalar(255, 255, 255), 10);
                            Log.d("Bruce", "四角定位：" + x + ", " + y);
                        }

                        Utils.matToBitmap(approxMat, resultBitmap);
                        resultIv.setImageBitmap(resultBitmap);
                    } else {
                        Toast.makeText(mActivity, "没有找到合适的轮廓", Toast.LENGTH_LONG).show();
                    }
                }
            }
        });
//        HoughLines功能代码
//        Mat lineMat = new Mat();
//        Imgproc.HoughLinesP(lastResultMat, lineMat, 1, Math.PI/180, 150, 0, 0);
//        Log.d("Bruce", "lineMat.rows() = " + lineMat.rows());
//        for (int x = 0; x < lineMat.rows(); x++) {
//            double[] vec = lineMat.get(x, 0);
//            Imgproc.line(lastResultMat, new Point(vec[0], vec[1]), new Point(vec[2], vec[3]), new Scalar(255,0,0), 3);
//        }
        btnProcess7.setText("截取原图");
        btnProcess7.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                width = (int) (xMax - xMin);
                height = (int) (yMax - yMin);
                Rect rect = new Rect((int) xMin, (int) yMin, (int) (xMax - xMin), (int) (yMax - yMin));
                lastResultMat = originMat.submat(rect);
                transformPoints = new Point[originPoints.length];

                for (int i = 0; i < originPoints.length; i++) {
                    Point p = originPoints[i];
                    double x = p.x - xMin, y = p.y - yMin;
                    originPoints[i] = new Point(x, y);
                    transformPoints[i] = new Point(x < 500 ? 0 : width, y < 500 ? 0 : height);
                    Log.d("Bruce", "x , y = " + x + ", " + y + "      " + transformPoints[i].x + ", " + transformPoints[i].y);
                }

                resultBitmapTemp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(lastResultMat, resultBitmapTemp);
                resultIv.setImageBitmap(resultBitmapTemp);
            }
        });

        btnProcess8.setText("透视转换");
        btnProcess8.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Mat resultMat = new Mat();
                resultMat.create(lastResultMat.rows(), lastResultMat.cols(), CvType.CV_8UC3);

                Mat src = lastResultMat;
                Mat dst = new Mat();
                List<Point> listSrcs = Arrays.asList(originPoints);
                Mat srcPoints = Converters.vector_Point_to_Mat(listSrcs, CvType.CV_32F);
                List<Point> listDsts = Arrays.asList(transformPoints);
                Mat dstPoints = Converters.vector_Point_to_Mat(listDsts, CvType.CV_32F);
                Mat perspectiveMmat = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
                Imgproc.warpPerspective(src, dst, perspectiveMmat, src.size(), Imgproc.INTER_LINEAR);
                lastResultMat = dst;

                int width = (int) (xMax - xMin), height = (int) (yMax - yMin);
                resultBitmapTemp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(lastResultMat, resultBitmapTemp);
                resultIv.setImageBitmap(resultBitmapTemp);
            }
        });

        btnProcess9.setText("图像增强");
        btnProcess9.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                Mat equalizeHistDMat = new Mat();
//                Imgproc.equalizeHist(lastResultMat, equalizeHistDMat);
//                lastResultMat = equalizeHistDMat;

                //1、白平衡
                Mat dstMat = new Mat();
                List<Mat> matChannels = new ArrayList<>();
                Core.split(lastResultMat, matChannels);// 分离通道
                Mat imageBlueChannel = matChannels.get(0);
                Mat imageGreenChannel = matChannels.get(1);
                Mat imageRedChannel = matChannels.get(2);
                double imageBlueChannelAvg = Core.mean(imageBlueChannel).val[0];// 求各通道的平均值
                double imageGreenChannelAvg = Core.mean(imageGreenChannel).val[0];
                double imageRedChannelAvg = Core.mean(imageRedChannel).val[0];
                double K = (imageRedChannelAvg + imageGreenChannelAvg + imageRedChannelAvg) / 3;// 求出各通道所占增益
                double Kb = K / imageBlueChannelAvg;
                double Kg = K / imageGreenChannelAvg;
                double Kr = K / imageRedChannelAvg;
                Log.d("Bruce", "kr, kg, kb = " + Kr + ", " + Kg + ", " + Kb);
                 Core.addWeighted(imageBlueChannel, Kb, imageBlueChannel, 0, 0, imageBlueChannel);
                 Core.addWeighted(imageGreenChannel, Kg, imageGreenChannel, 0, 0, imageGreenChannel);
                 Core.addWeighted(imageRedChannel, Kr, imageRedChannel, 0, 0, imageRedChannel);
                Core.merge(matChannels, dstMat);
                lastResultMat = dstMat;

                //2、对数增强
//                dstMat = new Mat(lastResultMat.size(), CV_32FC3);
//                for (int i = 0; i < lastResultMat.rows(); i++) {
//                    for (int j = 0; j < lastResultMat.cols(); j++) {
//                        dstMat.get(i,j)[0] = Math.log(1 + lastResultMat.get(i, j)[0]);
//                        dstMat.get(i,j)[1] = Math.log(1 + lastResultMat.get(i, j)[1]);
//                        dstMat.get(i,j)[2] = Math.log(1 + lastResultMat.get(i, j)[2]);
//                    }
//                }
//                Core.normalize(dstMat, dstMat, 0, 255,  Core.NORM_MINMAX);//归一化到0~255
//                Core.convertScaleAbs(dstMat, lastResultMat);//转换成8bit图像显示

                //3、增加亮度、对比度
                dstMat = new Mat(lastResultMat.size(), lastResultMat.type());
                Mat blackMat = Mat.zeros(lastResultMat.size(), lastResultMat.type());
//                Core.add(lastResultMat, new Scalar(30,30,30), dstMat);
//                Core.multiply(dstMat, new Scalar(0.1, 0.1, 0.1), lastResultMat);

                Core.addWeighted(lastResultMat, 1.2, blackMat, -0.2, 0, dstMat);
                lastResultMat = dstMat;


                Utils.matToBitmap(lastResultMat, resultBitmapTemp);
                resultIv.setImageBitmap(resultBitmapTemp);
                resultBitmap = resultBitmapTemp;
            }
        });
    }

    private void showToast(final String s) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(mActivity, s, Toast.LENGTH_LONG).show();
                String text = s + "\n";
                tv.append(text, 0, text.length());
            }
        });
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("bruce", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d("bruce", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
}