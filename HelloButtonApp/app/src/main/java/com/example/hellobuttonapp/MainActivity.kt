package com.example.hellobuttonapp

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.gms.location.LocationServices
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity(), SensorEventListener {

    companion object {
        // Keep for adb reverse or emulator. For a real LAN phone replace with "http://<your_pc_ip>:5001"
        private const val SERVER_URL_MANUAL = "http://127.0.0.1:5001"
        private val JSON = "application/json; charset=utf-8".toMediaType()
        private const val LOG_TAG = "SMARTLOC"
    }

    // base url (auto uses emulator alias when applicable)
    private val BASE_URL: String by lazy {
        if (isRunningOnEmulator()) "http://10.0.2.2:5001" else SERVER_URL_MANUAL
    }

    private val client by lazy {
        OkHttpClient.Builder()
            .callTimeout(8, TimeUnit.SECONDS)
            .connectTimeout(6, TimeUnit.SECONDS)
            .readTimeout(8, TimeUnit.SECONDS)
            .build()
    }

    // UI
    private lateinit var btnGps: Button
    private lateinit var btnModel: Button
    private lateinit var tvGpsResult: TextView
    private lateinit var tvModelResult: TextView
    private lateinit var tvStatus: TextView

    // sensors + streaming
    private lateinit var sensorManager: SensorManager
    private var streamingJob: Job? = null

    // latest sensor values (volatile to be safe across threads)
    @Volatile private var ax = 0f
    @Volatile private var ay = 0f
    @Volatile private var az = 0f
    @Volatile private var gx = 0f
    @Volatile private var gy = 0f
    @Volatile private var gz = 0f
    @Volatile private var mx = 0f
    @Volatile private var my = 0f
    @Volatile private var mz = 0f

    // location manager
    private var locationManager: LocationManager? = null
    private var lastLocation: Location? = null
    private val fusedLocationClient by lazy { LocationServices.getFusedLocationProviderClient(this) }

    // permission launcher
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { perms ->
            val ok = perms[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
                    perms[Manifest.permission.ACCESS_COARSE_LOCATION] == true
            if (!ok) {
                Toast.makeText(this, "Location permission required to include GPS labels.", Toast.LENGTH_SHORT).show()
            } else {
                startLocationListener()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // bind UI
        btnGps = findViewById(R.id.btnGps)
        btnModel = findViewById(R.id.btnModel)
        tvGpsResult = findViewById(R.id.tvGpsResult)
        tvModelResult = findViewById(R.id.tvModelResult)
        tvStatus = findViewById(R.id.tvStatus)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(LOCATION_SERVICE) as LocationManager

        // request location permission if needed
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
        } else {
            startLocationListener()
        }

        btnGps.setOnClickListener {
            tvStatus.text = "Requesting GPS from phone + server..."
            requestPhoneLocationAndSendGps()
        }

        btnModel.setOnClickListener {
            tvStatus.text = "Asking server for ML prediction..."
            sendModelRequest()
        }

        // quick reachability check
        lifecycleScope.launch { updateServerStatus() }
    }

    override fun onResume() {
        super.onResume()

        // register sensors
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        // start streaming loop (~10Hz)
        if (streamingJob == null) {
            streamingJob = lifecycleScope.launch {
                while (true) {
                    sendStreamPacket()
                    delay(100)
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()
        streamingJob?.cancel(); streamingJob = null
        sensorManager.unregisterListener(this)
        stopLocationListener()
    }

    override fun onDestroy() {
        super.onDestroy()
        streamingJob?.cancel()
    }

    private fun isRunningOnEmulator(): Boolean {
        return (Build.FINGERPRINT.startsWith("generic")
                || Build.FINGERPRINT.lowercase().contains("vbox")
                || Build.FINGERPRINT.lowercase().contains("test-keys")
                || Build.MODEL.contains("Emulator")
                || Build.MODEL.contains("Android SDK built for x86"))
    }

    // SensorEventListener
    override fun onSensorChanged(event: android.hardware.SensorEvent?) {
        if (event == null) return
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> { ax = event.values[0]; ay = event.values[1]; az = event.values[2] }
            Sensor.TYPE_GYROSCOPE -> { gx = event.values[0]; gy = event.values[1]; gz = event.values[2] }
            Sensor.TYPE_MAGNETIC_FIELD -> { mx = event.values[0]; my = event.values[1]; mz = event.values[2] }
        }
    }
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { /* ignore */ }

    // Location helpers
    private fun startLocationListener() {
        try {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                return
            }
            locationManager?.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, locationListener)
            locationManager?.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 1000L, 0f, locationListener)
            // seed lastLocation from last known providers if available
            lastLocation = try { locationManager?.getLastKnownLocation(LocationManager.GPS_PROVIDER) } catch (e: Exception) { null }
            if (lastLocation == null) lastLocation = try { locationManager?.getLastKnownLocation(LocationManager.NETWORK_PROVIDER) } catch (e: Exception) { null }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun stopLocationListener() {
        try { locationManager?.removeUpdates(locationListener) } catch (e: Exception) { /* ignore */ }
    }

    private val locationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) { lastLocation = location }
        override fun onProviderEnabled(provider: String) {}
        override fun onProviderDisabled(provider: String) {}
        override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    }

    // ---------------------------
    // Networking: stream IMU packet
    // ---------------------------
    private suspend fun sendStreamPacket() = withContext(Dispatchers.IO) {
        try {
            val json = JSONObject()
            val sample = JSONObject()
            sample.put("ax", ax.toDouble()); sample.put("ay", ay.toDouble()); sample.put("az", az.toDouble())
            sample.put("gx", gx.toDouble()); sample.put("gy", gy.toDouble()); sample.put("gz", gz.toDouble())
            sample.put("mx", mx.toDouble()); sample.put("my", my.toDouble()); sample.put("mz", mz.toDouble())
            json.put("sample", sample)
            json.put("timestamp", System.currentTimeMillis() / 1000.0)

            lastLocation?.let {
                val gps = JSONObject()
                gps.put("lat", it.latitude); gps.put("lon", it.longitude)
                json.put("gps", gps)
            }

            val body = json.toString().toRequestBody(JSON)
            val req = Request.Builder().url("$BASE_URL/stream").post(body).build()
            client.newCall(req).execute().use { resp -> /* ignore response here */ }
        } catch (e: IOException) {
            // network problems: ignore streaming failure silently (keeps UI responsive)
        } catch (e: Exception) {
            // ignore other streaming exceptions
        }
    }

    // ---------------------------
    // GPS button: get phone location then POST to /gps
    // ---------------------------
    private fun requestPhoneLocationAndSendGps() {
        try {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Location permission required", Toast.LENGTH_SHORT).show()
                return
            }

            fusedLocationClient.lastLocation.addOnSuccessListener { loc: Location? ->
                if (loc != null) {
                    lastLocation = loc
                    lifecycleScope.launch(Dispatchers.IO) { sendGpsRequestWithLocation(loc.latitude, loc.longitude) }
                } else {
                    lifecycleScope.launch(Dispatchers.IO) { sendGpsRequestWithLocation(null, null) }
                }
            }.addOnFailureListener {
                lifecycleScope.launch(Dispatchers.IO) { sendGpsRequestWithLocation(null, null) }
            }
        } catch (e: Exception) {
            lifecycleScope.launch(Dispatchers.IO) { sendGpsRequestWithLocation(null, null) }
        }
    }

    private suspend fun sendGpsRequestWithLocation(lat: Double?, lon: Double?) {
        withContext(Dispatchers.IO) {
            val url = "$BASE_URL/gps"
            try {
                val json = JSONObject()
                if (lat != null && lon != null) {
                    val gps = JSONObject(); gps.put("lat", lat); gps.put("lon", lon)
                    json.put("gps", gps)
                }
                val reqBody = json.toString().toRequestBody(JSON)
                val req = Request.Builder().url(url).post(reqBody).build()
                client.newCall(req).execute().use { resp ->
                    val text = resp.body?.string()
                    withContext(Dispatchers.Main) {
                        tvGpsResult.text = "GPS RAW: ${text ?: "(empty)"}"
                        val parsed = parseGpsFromResponse(text)
                        if (parsed != null) {
                            tvStatus.text = "GPS OK"
                            tvGpsResult.text = parsed
                        } else {
                            tvStatus.text = if (resp.code == 204) "GPS: no signal (204)" else "GPS: no coords"
                        }
                    }
                }
            } catch (e: IOException) {
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Network error (gps)"
                    Toast.makeText(this@MainActivity, "Network error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    tvStatus.text = "GPS error"
                    Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    // ---------------------------
    // MODEL request (predict) - robust parsing + UI update
    // ---------------------------
    private fun sendModelRequest() {
        lifecycleScope.launch(Dispatchers.IO) {
            val url = "$BASE_URL/predict"
            val reqBody = "{}".toRequestBody(JSON)
            val req = Request.Builder().url(url).post(reqBody).build()

            try {
                client.newCall(req).execute().use { resp ->
                    val raw = resp.body?.string()
                    android.util.Log.i(LOG_TAG, "MODEL RAW SERVER RESPONSE: ${raw ?: "(empty)"}")

                    // Try to parse coordinates robustly
                    var parsedLat: Double? = null
                    var parsedLon: Double? = null
                    var statusMessage: String? = null

                    if (!raw.isNullOrBlank()) {
                        try {
                            val jo = JSONObject(raw)
                            if (jo.has("lat") && jo.has("lon")) {
                                val lat = jo.optDouble("lat", Double.NaN)
                                val lon = jo.optDouble("lon", Double.NaN)
                                if (!lat.isNaN() && !lon.isNaN()) {
                                    parsedLat = lat; parsedLon = lon
                                }
                            }
                            if (parsedLat == null && jo.has("gps")) {
                                val g = jo.optJSONObject("gps")
                                if (g != null && g.has("lat") && g.has("lon")) {
                                    val lat = g.optDouble("lat", Double.NaN)
                                    val lon = g.optDouble("lon", Double.NaN)
                                    if (!lat.isNaN() && !lon.isNaN()) {
                                        parsedLat = lat; parsedLon = lon
                                    }
                                }
                            }
                            // fallback to status/message
                            if (parsedLat == null) {
                                val st = jo.optString("status", "")
                                val msg = jo.optString("message", "")
                                if (msg.isNotBlank() || st.isNotBlank()) {
                                    statusMessage = if (msg.isNotBlank()) "$st — $msg" else st
                                }
                            }
                        } catch (je: Exception) {
                            android.util.Log.w(LOG_TAG, "JSON parse failed: ${je.message}")
                        }
                    }

                    // Regex fallback if JSON didn't yield coords
                    if (parsedLat == null && !raw.isNullOrBlank()) {
                        val latRegex = """"lat"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)""".toRegex()
                        val lonRegex = """"lon"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)""".toRegex()
                        val latMatch = latRegex.find(raw)
                        val lonMatch = lonRegex.find(raw)
                        if (latMatch != null && lonMatch != null) {
                            val lat = latMatch.groupValues[1].toDoubleOrNull()
                            val lon = lonMatch.groupValues[1].toDoubleOrNull()
                            if (lat != null && lon != null) {
                                parsedLat = lat; parsedLon = lon
                            }
                        }
                    }

                    // UI update
                    withContext(Dispatchers.Main) {
                        if (parsedLat != null && parsedLon != null) {
                            val formatted = "Model: $parsedLat , $parsedLon"
                            tvModelResult.text = formatted
                            tvStatus.text = "Model OK"
                            android.util.Log.i(LOG_TAG, "PARSED MODEL RESPONSE: $formatted")
                            Toast.makeText(this@MainActivity, "Model predicted: ${parsedLat}, ${parsedLon}", Toast.LENGTH_SHORT).show()
                        } else {
                            // show parsed status message if present, otherwise show raw for debugging
                            if (!statusMessage.isNullOrBlank()) {
                                tvModelResult.text = "Model status: $statusMessage"
                                tvStatus.text = "Model: status"
                            } else {
                                tvModelResult.text = "Model raw: ${raw ?: "(empty)"}"
                                tvStatus.text = if (resp.isSuccessful) "Model OK (no coords parsed)" else "Model request failed: ${resp.code}"
                            }
                            android.util.Log.w(LOG_TAG, "Could not parse coords from model response. Raw -> ${raw ?: "(empty)"}")
                        }
                    }
                }
            } catch (e: IOException) {
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Network error (model)"
                    Toast.makeText(this@MainActivity, "Network error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Model request error"
                    Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private suspend fun updateServerStatus() = withContext(Dispatchers.IO) {
        val url = "$BASE_URL/status"
        val req = Request.Builder().url(url).get().build()
        try {
            client.newCall(req).execute().use { resp ->
                val text = resp.body?.string()
                withContext(Dispatchers.Main) {
                    tvStatus.text = if (resp.isSuccessful && !text.isNullOrBlank()) "Server: reachable" else "Server: unreachable"
                }
            }
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                tvStatus.text = "Server: network error"
            }
        }
    }

    // Parsers helpers used for GPS button
    private fun parseGpsFromResponse(body: String?): String? {
        if (body.isNullOrBlank()) return null
        return try {
            val jo = JSONObject(body)
            val lat = when {
                jo.has("lat") -> jo.getDouble("lat")
                jo.has("gps") -> jo.optJSONObject("gps")?.optDouble("lat") ?: Double.NaN
                else -> Double.NaN
            }
            val lon = when {
                jo.has("lon") -> jo.getDouble("lon")
                jo.has("gps") -> jo.optJSONObject("gps")?.optDouble("lon") ?: Double.NaN
                else -> Double.NaN
            }
            if (!lat.isNaN() && !lon.isNaN()) "GPS: $lat , $lon" else null
        } catch (e: Exception) { null }
    }
}
