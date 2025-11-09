package com.example.a9nov

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Scaffold
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.Task
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

suspend fun <T> Task<T>.awaitOrNull(): T? {
    return try {
        await()
    } catch (e: Exception) {
        null
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            WearLocationApp()
        }
    }
}

@Composable
fun WearLocationApp() {
    // Simple material theme for Wear
    MaterialTheme {
        Scaffold(
            timeText = { TimeText() }, // shows small time at top (Wear style)
        ) {
            LocationScreen()
        }
    }
}

@Composable
fun LocationScreen() {
    val context = LocalContext.current
    val fusedClient = remember { LocationServices.getFusedLocationProviderClient(context) }

    // UI state
    var locationText by remember { mutableStateOf("Press the button to get location") }
    var permissionGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    // Permission launcher
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { perms ->
        val fineGranted = perms[Manifest.permission.ACCESS_FINE_LOCATION] ?: false
        permissionGranted = fineGranted
        if (!fineGranted) {
            locationText = "Location permission denied"
        }
    }

    // Coroutine scope for fetching location
    val coroutineScope = rememberCoroutineScope()

    // Layout
    Box(
        modifier = Modifier.fillMaxSize().padding(12.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth().background(MaterialTheme.colors.surface).padding(10.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Predicted location",
                    style = MaterialTheme.typography.title1,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = locationText,
                    style = MaterialTheme.typography.body2,
                    textAlign = TextAlign.Center
                )
            }

            // The button (Chip) — modern and large for touch
            Chip(
                onClick = {
                    if (!permissionGranted) {
                        // Ask permissions
                        launcher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
                    } else {
                        // Fetch location
                        locationText = "Getting location..."
                        coroutineScope.launch {
                            try {
                                // Try last known location first
                                val last = fusedClient.lastLocation.awaitOrNull()
                                if (last != null) {
                                    locationText = "Lat: ${"%.5f".format(last.latitude)}, Lon: ${"%.5f".format(last.longitude)}"
                                } else {
                                    // Request current high-accuracy location
                                    val current = fusedClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null).awaitOrNull()
                                    if (current != null) {
                                        locationText = "Lat: ${"%.5f".format(current.latitude)}, Lon: ${"%.5f".format(current.longitude)}"
                                    } else {
                                        locationText = "Couldn't get location. Is GPS on?"
                                    }
                                 }
                            } catch (e: Exception) {
                                locationText = "Error: ${e.message}"
                            }
                        }
                    }
                },
                label = { Text("Predicted location") },
                modifier = Modifier.padding(top = 8.dp)
            )
        }
    }
}