package com.ringo.opencvuvctest;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.hardware.usb.UsbDevice;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.serenegiant.common.BaseActivity;
import com.serenegiant.opencv.ImageProcessor;
import com.serenegiant.usb.CameraDialog;
import com.serenegiant.usb.USBMonitor;
import com.serenegiant.usb.USBMonitor.OnDeviceConnectListener;

import com.serenegiant.usbcameracommon.UVCCameraHandlerMultiSurface;
import com.serenegiant.widget.UVCCameraTextureView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;


public class MainActivity extends BaseActivity implements CameraDialog.CameraDialogParent{
    private ToggleButton mCameraButton;

    private USBMonitor mUSBMonitor;
    private UVCCameraTextureView mUVCCameraView;
    private UVCCameraHandlerMultiSurface mCameraHandler;
    private static final boolean USE_SURFACE_ENCODER = false;
    private static final int PREVIEW_WIDTH = 640;
    private static final int PREVIEW_HEIGHT = 480;
    private static final int PREVIEW_MODE = 1;

    private static final boolean DEBUG = true;	// TODO set false on release
    private static final String TAG = "MainActivity";
    protected ImageProcessor mImageProcessor;
    protected SurfaceView mResultView;
    private TextView mTextView;
    private TextView mResultTextView;
    private String modelFilename = "file:///android_asset/net5_2.5.pb";


    private static final String EXSTORAGE_PATH = String.format("%s/%s", Environment.getExternalStorageDirectory().toString(), "OpencvUVCTest/");
    private static final String TESSDATA_PATH = String.format("%s/%s", EXSTORAGE_PATH, "tessdata/");
    private static final String TRAIN_LANG = "eng";
    private static final String TRAINEDDATA = String.format("%s.traineddata", TRAIN_LANG);


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        mCameraButton = (ToggleButton)findViewById(R.id.camera_button);
        mCameraButton.setOnCheckedChangeListener(mOnCheckedChangeListener);

        mUVCCameraView = (UVCCameraTextureView)findViewById(R.id.camera_view);
        mUVCCameraView.setAspectRatio(PREVIEW_WIDTH / (float)PREVIEW_HEIGHT);

        mResultView = (SurfaceView)findViewById(R.id.surfaceView);

        mTextView = (TextView) findViewById(R.id.textView);
        mResultTextView = (TextView) findViewById(R.id.textView2);

        mUSBMonitor = new USBMonitor(this, mOnDeviceConnectListener);
        mCameraHandler = UVCCameraHandlerMultiSurface.createHandler(this, mUVCCameraView,
                USE_SURFACE_ENCODER ? 0 : 1, PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_MODE);

//        Mytest();
//        try {
//            prepareTrainedFileIfNotExist();
//        } catch (Exception e) {
//            Log.e(TAG, e.toString());
//        }
    }

    /**
     * traineddataを assetsから外部ストレージにコピーする。TessBaseAPIクラスがアセットを直接扱えないため。
     * @throws Exception アセットから外部ストレージへのコピーに失敗
     */
//    private void prepareTrainedFileIfNotExist() throws Exception {
//
//        // MEMO : Manifestの android.permission.WRITE_EXTERNAL_STORAGEを忘れずに
//
//        String paths[] = {EXSTORAGE_PATH, EXSTORAGE_PATH + "/tessdata"};
//        for (String path : paths) {
//            File dir = new File(path);
//            if (!dir.exists()) {
//                if (!dir.mkdirs()) {
//                    throw new Exception("ディレクトリ生成に失敗");
//                }
//            }
//        }
//
//        String traineddata_path = String.format("%s%s", TESSDATA_PATH, TRAINEDDATA);
//
//        if ( (new File(traineddata_path).exists()))
//            return;
//
//        try {
//            InputStream   in = getAssets().open(TRAINEDDATA);
//            OutputStream out = new FileOutputStream(traineddata_path);
//
//            byte[] buf = new byte[1024];
//            int len;
//            while ((len = in.read(buf)) > 0) {
//                out.write(buf, 0, len);
//            }
//            in.close();
//            out.close();
//        } catch (IOException e) {
//            Log.e(TAG, e.toString());
//            throw new Exception("アセットのコピーに失敗");
//        }
//    }

    private Bitmap bitmap;


    @Override
    protected void onStart() {
        super.onStart();
        mUSBMonitor.register();
    }

    @Override
    protected void onStop() {
        stopPreview();
        mCameraHandler.close();
        super.onStop();
    }

    @Override
    public void onDestroy() {
        if (mCameraHandler != null) {
            mCameraHandler.release();
            mCameraHandler = null;
        }
        if (mUSBMonitor != null) {
            mUSBMonitor.destroy();
            mUSBMonitor = null;
        }
        mUVCCameraView = null;
        super.onDestroy();
    }

    private final CompoundButton.OnCheckedChangeListener mOnCheckedChangeListener
            = new CompoundButton.OnCheckedChangeListener() {
        @Override
        public void onCheckedChanged(
                final CompoundButton compoundButton, final boolean isChecked) {

            switch (compoundButton.getId()) {
                case R.id.camera_button:
                    if (isChecked && !mCameraHandler.isOpened()) {
                        CameraDialog.showDialog(MainActivity.this);
                    } else {
                        stopPreview();
                    }
                    break;
            }
        }
    };

    private void setCameraButton(final boolean isOn) {
        if (DEBUG) Log.v(TAG, "setCameraButton:isOn=" + isOn);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (mCameraButton != null) {
                    try {
                        mCameraButton.setOnCheckedChangeListener(null);
                        mCameraButton.setChecked(isOn);
                    } finally {
                        mCameraButton.setOnCheckedChangeListener(mOnCheckedChangeListener);
                    }
                }
//                if (!isOn && (mCaptureButton != null)) {
//                    mCaptureButton.setVisibility(View.INVISIBLE);
//                }
            }
        }, 0);

//        updateItems();
    }

    private int mPreviewSurfaceId;

    private void startPreview() {
        if (DEBUG) Log.v(TAG, "startPreview:");
        mUVCCameraView.resetFps();
        mCameraHandler.startPreview();
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                try {
                    final SurfaceTexture st = mUVCCameraView.getSurfaceTexture();
                    if (st != null) {
                        final Surface surface = new Surface(st);
                        mPreviewSurfaceId = surface.hashCode();
                        mCameraHandler.addSurface(mPreviewSurfaceId, surface, false);
                    }
                    startImageProcessor(PREVIEW_WIDTH, PREVIEW_HEIGHT);
                } catch (final Exception e) {
                    Log.w(TAG, e);
                }
            }
        });
//        updateItems();
    }

    private void stopPreview() {
        if (DEBUG) Log.v(TAG, "stopPreview:");
        stopImageProcessor();
        if (mPreviewSurfaceId != 0) {
            mCameraHandler.removeSurface(mPreviewSurfaceId);
            mPreviewSurfaceId = 0;
        }
        mCameraHandler.close();
    }

    private final OnDeviceConnectListener mOnDeviceConnectListener
            = new OnDeviceConnectListener() {

        @Override
        public void onAttach(final UsbDevice device) {
            Toast.makeText(MainActivity.this,
                    "USB_DEVICE_ATTACHED", Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onConnect(final UsbDevice device,
                              final USBMonitor.UsbControlBlock ctrlBlock, final boolean createNew) {

            if (DEBUG) Log.v(TAG, "onConnect:");
            mCameraHandler.open(ctrlBlock);
            startPreview();
//            updateItems();
        }

        @Override
        public void onDisconnect(final UsbDevice device,
                                 final USBMonitor.UsbControlBlock ctrlBlock) {

            if (DEBUG) Log.v(TAG, "onDisconnect:");
            if (mCameraHandler != null) {
                queueEvent(new Runnable() {
                    @Override
                    public void run() {
                        stopPreview();
                    }
                }, 0);
//                updateItems();
            }
        }
        @Override
        public void onDettach(final UsbDevice device) {
            Toast.makeText(MainActivity.this,
                    "USB_DEVICE_DETACHED", Toast.LENGTH_SHORT).show();
        }

        public void onCancel(final UsbDevice device) {
        }
    };

    /**
     * to access from CameraDialog
     * @return
     */
    @Override
    public USBMonitor getUSBMonitor() {
        return mUSBMonitor;
    }

    @Override
    public void onDialogResult(boolean canceled) {
        if (DEBUG) Log.v(TAG, "onDialogResult:canceled=" + canceled);
        if (canceled) {
            setCameraButton(false);
        }

    }

    //================================================================================
    private volatile boolean mIsRunning;
    private int mImageProcessorSurfaceId;

    /**
     * start image processing
     * @param processing_width
     * @param processing_height
     */
    protected void startImageProcessor(final int processing_width, final int processing_height) {
        if (DEBUG) Log.v(TAG, "startImageProcessor:");
        mIsRunning = true;
        if (mImageProcessor == null) {
            mImageProcessor = new ImageProcessor(PREVIEW_WIDTH, PREVIEW_HEIGHT,	// src size
                    new MyImageProcessorCallback(processing_width, processing_height, this));	// processing size
            mImageProcessor.start(processing_width, processing_height);	// processing size
            final Surface surface = mImageProcessor.getSurface();
            mImageProcessorSurfaceId = surface != null ? surface.hashCode() : 0;
            if (mImageProcessorSurfaceId != 0) {
                mCameraHandler.addSurface(mImageProcessorSurfaceId, surface, false);
            }
        }
    }

    /**
     * stop image processing
     */
    protected void stopImageProcessor() {
        if (DEBUG) Log.v(TAG, "stopImageProcessor:");
        if (mImageProcessorSurfaceId != 0) {
            mCameraHandler.removeSurface(mImageProcessorSurfaceId);
            mImageProcessorSurfaceId = 0;
        }
        if (mImageProcessor != null) {
            mImageProcessor.release();
            mImageProcessor = null;
        }
    }

    /**
     * callback listener from `ImageProcessor`
     */
    protected class MyImageProcessorCallback implements ImageProcessor.ImageProcessorCallback {
        private final int width, height;
        private final Matrix matrix = new Matrix();
        private Bitmap mFrame;

        private Context mContext;
        private RoiProcessor rp;
//        Bitmap bitmap;
        protected MyImageProcessorCallback(
                final int processing_width, final int processing_height, Context context) {

            width = processing_width;
            height = processing_height;
            mContext = context;
            rp = new RoiProcessor(context);


        }
//        @Override
//        public void onFrame(final ByteBuffer frame) {
//            if (mResultView != null) {
//                final SurfaceHolder holder = mResultView.getHolder();
//                if ((holder == null)
//                        || (holder.getSurface() == null)
//                        || (frame == null)) return;
//
////--------------------------------------------------------------------------------
//// Using SurfaceView and Bitmap to draw resulted images is inefficient way,
//// but functions onOpenCV are relatively heavy and expect slower than source
//// frame rate. So currently just use the way to simply this sample app.
//// If you want to use much efficient way, try to use as same way as
//// UVCCamera class use to receive images from UVC camera.
////--------------------------------------------------------------------------------
//                if (mFrame == null) {
//                    mFrame = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
//                    final float scaleX = mResultView.getWidth() / (float) width;
//                    final float scaleY = mResultView.getHeight() / (float) height;
//                    matrix.reset();
//                    matrix.postScale(scaleX, scaleY);
//                }
//                    try {
//                        frame.clear();
//                        mFrame.copyPixelsFromBuffer(frame);
////                        rp.SetFrame(frame,width,height);
////                        int frameCounter = rp.getCounter();
////                        mTextView.setText("Counter = " + frameCounter);
//                        mTextView.setText(" OnFrame ");
////                    rp.CalculateMask();
////                    rp.GetRoi();
//
//                        // write something
//
//
//
//                        final Canvas canvas = holder.lockCanvas();
//                        if (canvas != null) {
//                            try {
//                                canvas.drawBitmap(mFrame, matrix, null);
//                            } catch (final Exception e) {
//                                Log.w(TAG, e);
//                            } finally {
//                                holder.unlockCanvasAndPost(canvas);
//                            }
//                        }
//                    } catch (final Exception e) {
//                        Log.w(TAG, e);
//                        Toast.makeText(MainActivity.this,
//                                "Onframe error", Toast.LENGTH_SHORT).show();
//                    }
//                }
//
//        }
        @Override
        public void onFrame(final ByteBuffer frame){

//            mTextView = (TextView) findViewById(R.id.textView);
            Bitmap bmptmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

            final SurfaceHolder holder = mResultView.getHolder();
            if ((holder == null)
                    || (holder.getSurface() == null)
                    || (frame == null)) return;

            if (mFrame == null) {
                mFrame = Bitmap.createBitmap(256, 192, Bitmap.Config.ARGB_8888);
                final float scaleX = mResultView.getWidth() / (float) PREVIEW_WIDTH;
                final float scaleY = mResultView.getHeight() / (float) PREVIEW_HEIGHT;
                matrix.reset();
                matrix.postScale(scaleX, scaleY);
            }
            try {
//                frame.clear();
                mUVCCameraView.getBitmap(bmptmp);
                rp.SetOrigFrame(bmptmp);
//                bmptmp.copyPixelsFromBuffer(frame);
                mFrame = Bitmap.createScaledBitmap(bmptmp,
                        256, 192, true);
//                bmptmp.recycle();


                rp.SetFrame(mFrame);
//                rp.SetFrame(bitmap);


                rp.SetAssetManager(getAssets());
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        mTextView.setText("Calculating mask...");

                    }
                });

                rp.CalculateMask();


                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        mTextView.setText("Recognizing ROI area...");

                    }
                });


                rp.CalculateRoi();

//                Bitmap roi = rp.GetROI();
                Bitmap OrigWithMask = rp.GetOrigWithMask();

                final Canvas canvas = holder.lockCanvas();
                if (canvas != null) {
                    try {
                        canvas.drawBitmap(OrigWithMask, matrix, null);
                    } catch (final Exception e) {
                        Log.w(TAG, e);
                    } finally {
                        holder.unlockCanvasAndPost(canvas);
                    }
                }
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        mTextView.setText("Recognizing text...");


                    }
                });

                String str = rp.RecognizeText();

                if (str!=null && !str.isEmpty()) {
                    mResultTextView.setText(str);

//                    runOnUiThread(new Runnable() {
//                        @Override
//                        public void run() {
//                        }
//                    });

                }

            }catch (final Exception e) {
                        Log.w(TAG, e);
                        Toast.makeText(MainActivity.this,
                                "Onframe error", Toast.LENGTH_SHORT).show();
            } finally {
                rp.ReleaseAll();
            }
        }

        @Override
        public void onResult(final int type, final float[] result) {
            // do something
//            mTextView.setText("OnResult");
        }

    }




}
