package com.example.objectdetection
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.core.content.ContextCompat
import com.example.objectdetection.ml.Model4
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {

    lateinit var labels:List<String>
    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    val paint = Paint()
    lateinit var imageProcessor: ImageProcessor
    lateinit var bitmap:Bitmap
    lateinit var imageView: ImageView
    lateinit var cameraDevice: CameraDevice
    lateinit var handler: Handler
    lateinit var cameraManager: CameraManager
    lateinit var textureView: TextureView
    lateinit var model:Model4

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permission()

        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()
        model = Model4.newInstance(this)
        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()

        handler = Handler(handlerThread.looper)
        imageView = findViewById(R.id.imageView)
        textureView = findViewById(R.id.textureView)

        textureView.surfaceTextureListener = object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera()
            }
            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!
                var image = TensorImage(DataType.FLOAT32)
                image.load(bitmap)
                image = imageProcessor.process(image)

                val outputs = model.process(image.tensorBuffer)
                val bbox = outputs.outputFeature0AsTensorBuffer.floatArray
                val classProbs = outputs.outputFeature1AsTensorBuffer.floatArray
//                Log.d("ModelOutput", "BBox: ${bbox.joinToString()}")
                Log.d("BBox Raw", "Raw BBox: ${bbox.joinToString()}")
                val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutableBitmap)

                val w = mutableBitmap.width
                val h = mutableBitmap.height

                Log.d("ImageSize", "Width: $w, Height: $h")

                val paint = Paint().apply {
                    color = Color.RED
                    style = Paint.Style.STROKE
                    strokeWidth = 4f
                    isAntiAlias = true
                }
                val textPaint = Paint().apply {
                    color = Color.YELLOW
                    textSize = 32f
                    isAntiAlias = true
                }

                val maxProb = classProbs.maxOrNull() ?: 0f
                val classIndex = classProbs.indices.maxByOrNull { classProbs[it] } ?: -1

                val threshold = 0.9f

                if (classIndex != -1 && maxProb > threshold && bbox.size == 4) {

                    // Convert [x_center, y_center, width, height] to box corners

                    val xCenter = bbox[0] * w
                    val yCenter = bbox[1] * h
                    val boxWidth = bbox[2] * w
                    val boxHeight = bbox[3] * h

                    val xMin = (xCenter - boxWidth / 2)
                    val yMin = (yCenter - boxHeight / 2)
                    val xMax = (xCenter + boxWidth / 2)
                    val yMax = (yCenter + boxHeight / 2)

                    Log.d("BBox", "xMin=$xMin, yMin=$yMin, xMax=$xMax, yMax=$yMax")

                    canvas.drawRect(xMin, yMin, xMax, yMax, paint)

                    val label = if (labels.size > classIndex && classIndex >= 0) {
                        "${labels[classIndex]} (${String.format("%.2f", maxProb)})"
                    } else {
                        "Class $classIndex (${String.format("%.2f", maxProb)})"
                    }

                    canvas.drawText(label, xMin, yMin - 10, textPaint)

                    // Optionally draw the center point
                    canvas.drawCircle(xCenter * (w / 256f), yCenter * (h / 256f), 8f, Paint().apply {
                        color = Color.GREEN
                        style = Paint.Style.FILL
                    })
                }

                imageView.setImageBitmap(mutableBitmap)
            }
        }

        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    @SuppressLint("MissingPermission")
    fun open_camera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0], object:CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0

                var surfaceTexture = textureView.surfaceTexture
                var surface = Surface(surfaceTexture)

                var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {

            }

            override fun onError(p0: CameraDevice, p1: Int) {

            }
        }, handler)
    }

    fun get_permission(){
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 101)
        }
    }
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED){
            get_permission()
        }
    }
}