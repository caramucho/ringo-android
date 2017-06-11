package com.ringo.opencvuvctest;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;
import android.util.SparseArray;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.THRESH_BINARY;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;
import com.googlecode.tesseract.android.TessBaseAPI;


/**
 * Created by zhaoliang on 2017/05/21.
 */

public class RoiProcessor {


    private Bitmap mFrame;
    private Bitmap mMask;
    private Bitmap mTextRoi;
    private Bitmap mOrigFrame;
    private Bitmap mFrameWithrect;
    private AssetManager mAssetManager;
    private Context mContext;

    private String modelFilename = "file:///android_asset/net5_2.5.pb";
    private String[] outputNames = new String[] {"output"};
    private int imageMean = 0;
    private float imageStd = 1;
    private String outputName = "output";
    private boolean logStats = false;
//    private static int counter = 0;

    public RoiProcessor(Context context){
        mContext = context;
    }

    public void SetAssetManager(AssetManager assetManager){
        mAssetManager = assetManager;
    }

//    public void SetFrame(ByteBuffer frame, int width , int height){
//        mFrame = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
//        mFrame.copyPixelsFromBuffer(frame);
//        counter += 1;
//    }

    public void SetFrame(Bitmap frame){
//        mFrame = frame.copy(frame.getConfig(), true);
        mFrame = frame;

    }

    public void SetOrigFrame(Bitmap frame){
//        mFrame = frame.copy(frame.getConfig(), true);
        mOrigFrame = frame;

    }


//    public Bitmap GetMask(){
//        return mMask;
//    }

    public Bitmap GetROI(){
        return mTextRoi;
    }

    public  Bitmap GetOrigWithMask(){
        return mFrameWithrect;
    }

    public void CalculateMask(){
        if (mFrame == null) {
            throw new IllegalArgumentException("mFrame is null");
        }
        TensorFlowInferenceInterface inferenceInterface;

        int width = mFrame.getWidth();
        int height = mFrame.getHeight();
        int pixels[] = new int[width * height];
        float[] floatValues = new float[width * height * 3];
        float[] outputs = new float[width * height * 2];

        int[] mask = new int[width * height];
//        ByteBuffer mask;


        mFrame.getPixels(pixels, 0, mFrame.getWidth(), 0, 0, mFrame.getWidth(), mFrame.getHeight());
        for (int i = 0; i < pixels.length; ++i) {
            final int val = pixels[i];


            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        for (int i = 0; i < floatValues.length; ++i) {
            final float val = floatValues[i];
            floatValues[i] = val / 255;
        }

        inferenceInterface = new TensorFlowInferenceInterface(mAssetManager, modelFilename);
        inferenceInterface.feed("input", floatValues, 1, height, width, 3);
        inferenceInterface.run(outputNames , logStats);
        inferenceInterface.fetch(outputName, outputs);


        for (int i = 0; i < mask.length; ++i) {
            mask[i] = outputs[2*i] > outputs[2*i+1] ? Color.argb(255,0,0,0) : Color.argb(255,255,255,255);
        }


        // You are using RGBA that's why Config is ARGB.8888

        mMask = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        // vector is your int[] of ARGB
//        bitmap_mask.copyPixelsFromBuffer(IntBuffer.wrap(mask));
        mMask.setPixels(mask, 0, width, 0, 0, width, height);

    }



    public void CalculateRoi(){

        System.loadLibrary("opencv_java3");

        int mwidth = mMask.getWidth();
        int mheight = mMask.getHeight();
        Mat mask = new Mat (mwidth, mheight, CvType.CV_8UC1);
        Utils.bitmapToMat(mMask, mask);

        Mat frame = new Mat (mFrame.getWidth(), mFrame.getHeight(), CvType.CV_8UC1);
        Utils.bitmapToMat(mFrame, frame);

        Mat Origframe = new Mat (mOrigFrame.getWidth(), mOrigFrame.getHeight(), CvType.CV_8UC1);
        Utils.bitmapToMat(mOrigFrame, Origframe);
        int numberOfContours = 100;
//        String file_path = "./masks/";

//        Mat mask = Imgcodecs.imread(file_path+"002211.png",Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE); //mask image
//        Mat frame = Imgcodecs.imread(file_path + "002211.jpg"); //frame image
        Imgproc.cvtColor(mask , mask, Imgproc.COLOR_RGB2GRAY);

		Mat threshold = new Mat(mask.rows(),mask.cols(),mask.type());

        Imgproc.threshold(mask, threshold, 0 , 255 , THRESH_BINARY);
        // structure for storing contours, the elements of which are integer type (x,y) points
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>(numberOfContours);


        Mat hierarchy = new Mat(mask.rows(),mask.cols(), CvType.CV_8UC1,new Scalar(0));
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);

//        Mat for drawing contours, shapes upon
//        Mat drawing=Mat.zeros(mask.size(), CvType.CV_8UC3);

        //Number of contours detected
//        System.out.println("contours.size():"+contours.size());
        if(contours.size() == 0){
            throw new IllegalArgumentException("contours.size() is 0");
//            return;
        }

        double max_area = 0;
        int index_max = 0; //integer for storing the index of contour with the largest area
        double contour_area = 0;

        //Dectect the contour with the largest area,
        for(int i=0;i<contours.size();i++){
            contour_area = Imgproc.contourArea(contours.get(i));
            System.out.println("contour[" + i + "] area: " + contour_area);
            if (contour_area > max_area){
                max_area = contour_area;
                index_max = i;
            }
        }

        //Draw the contour with the largest area onto Mat drawing
//        Imgproc.drawContours(drawing, contours, index_max, new Scalar(255,0,0,255),1);
        RotatedRect r= Imgproc.minAreaRect(new MatOfPoint2f(contours.get(index_max).toArray()));

        //Print out the parameters of the slant minimum slant bounding Rectangle
//        System.out.println("RotatedRect r = Imgproc.minAreaRect()");
//        System.out.println("----------");
//        System.out.println("r: "+r);
//        System.out.println("max_area" + max_area);
//        System.out.println("r.boundingRect().height"+r.boundingRect().height+"r.boundingRect().width"+r.boundingRect().width);
//        System.out.println("r.boundingRect().area()"+r.boundingRect().area());
//        System.out.println("r.center: "+r.center);
//        System.out.println("r.size: "+r.size.height + r.size.width);
//        System.out.println("r.height: "+r.size.height + "r.width: " + r.size.width);
//        System.out.println("r.angle: "+r.angle);
//        System.out.println("----------");

        //The coordinates of the four vertice points of miniRect r
//        Point[] allPoint=new Point[4];
//        r.points(allPoint);
//        System.out.println("Coordinates of the vertices of miniBoundingRect r");
//        System.out.println("p0="+allPoint[0].x+","+allPoint[0].y);
//        System.out.println("p1="+allPoint[1].x+","+allPoint[1].y);
//        System.out.println("p2="+allPoint[2].x+","+allPoint[2].y);
//        System.out.println("p3="+allPoint[3].x+","+allPoint[3].y);
//
        //'Draw' the the miniRect into a Mat variable drawing
//        for (int j = 0; j < 4; j++){
//            Imgproc.line(drawing, allPoint[j], allPoint[(j+1)%4], new Scalar(0,255,0),2);
//        }

        //Save the drawing (with contour and bounding rectangle)
//        Imgcodecs.imwrite(file_path + "test_002211_contours.png", drawing);
//        System.out.println("----------");

        //RotationMatrix2D for the following affine transformation
        Mat rotMat=new Mat(2,3,CvType.CV_32FC1);

        //Mat for storing the transformed image
        Mat destination=new Mat(Origframe.rows(),Origframe.cols(),Origframe.type());

        //Rotation center and rotation angle
        Point center=r.center;
        double angle = r.angle;
//        if (angle < -45)
//            angle = angle + 90.0;
        double scale = 1.0;

        if ((r.boundingRect().height > r.boundingRect().width && angle > -45) ||(r.boundingRect().height < r.boundingRect().width && angle < -45))
            angle = angle + 90.0;

        //Calculate the rotation matrix, and apply the Affine transformation
        rotMat=Imgproc.getRotationMatrix2D(new Point(center.x * 2.5, center.y * 2.5), angle, scale);
        Imgproc.warpAffine(Origframe, destination, rotMat, destination.size());

        //Save the rotated image
//        Imgcodecs.imwrite(file_path + "test_002211_rotate.png", destination);

        //The box area to crop
        int px = (int)((r.center.x - 0.5* r.size.height) * 2.5);
        int py = (int)((r.center.y - 0.5* r.size.width)* 2.5);
        int width = (int)((r.size.height+0.5)* 2.5);
        int height = (int)((r.size.width+0.5)* 2.5);

        //Crop out the ROI(Region of interest)
        Rect rectCrop = new Rect(px, py, width, height);
        Mat destinationROI = destination.submat(rectCrop);

        mTextRoi = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(destinationROI, mTextRoi);

        //text, 原圖片
//source，mask區域
        Mat EQHist=new Mat(Origframe.rows(),Origframe.cols(),Origframe.type());
        List<Mat> bgrList = new ArrayList<Mat>(3);
        Core.split(Origframe, bgrList);
        Core.addWeighted(bgrList.get(1), 1, mask, 1, 10, bgrList.get(1)); //bgrList.get(1)*1 + source*1 + 10 for G channel(green channel doubled)
        Core.merge(bgrList, EQHist);

        mFrameWithrect = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(Origframe,mFrameWithrect);

    }

//    public String RecognizeText(){
//        TextRecognizer textRecognizer = new TextRecognizer.Builder(mContext).build();
//        Frame frame = new Frame.Builder().setBitmap(mTextRoi).build();
//        SparseArray<TextBlock> origTextBlocks = textRecognizer.detect(frame);
//
//        List<TextBlock> textBlocks = new ArrayList<>();
//        for (int i = 0; i < origTextBlocks.size(); i++) {
//            TextBlock textBlock = origTextBlocks.valueAt(i);
//            textBlocks.add(textBlock);
//        }
//        StringBuilder detectedText = new StringBuilder();
//
//        try {
//            for (TextBlock textBlock : textBlocks) {
//                if (textBlock != null && textBlock.getValue() != null) {
//                    detectedText.append(textBlock.getValue());
//                    detectedText.append("\n");
//                }
//            }
//        } catch (final Exception e) {
//
//        }
//        return detectedText.toString();
//    }

    public String RecognizeText(){
        return recognizeText(mContext, mTextRoi);
    }


    public static String recognizeText(Context context, Bitmap bitmap) {
        Bitmap converted = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        TessBaseAPI baseApi = new TessBaseAPI();
        baseApi.init(getTessdataDir(context).getParentFile().getAbsolutePath(),"eng");
//        baseApi.init(context.getAssets(),"eng");
        baseApi.setImage(converted);
        String recognizedText = baseApi.getUTF8Text();
        baseApi.end();
        return recognizedText;
    }

    private static File getTessdataDir(Context context) {
        File dir = new File(context.getExternalFilesDir("tesseract"), "tessdata");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new RuntimeException("cannot create file dir.");
        }
        return dir;
    }

    public void ReleaseAll(){
        mFrame.recycle();
        mFrame = null;
        mMask.recycle();
        mMask = null;
        mTextRoi.recycle();
        mTextRoi = null;
        mOrigFrame.recycle();
        mOrigFrame = null;
    }


}
