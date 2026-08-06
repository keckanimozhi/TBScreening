package com.kongu.ai4tb.wrapper

import android.os.Bundle
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity

/**
 * Thin WebView wrapper around the AI4TB-Mobile Streamlit app
 * (TB screening/app/app.py) -- the original Task 14 deliverable, kept as
 * a secondary screen after Task 18 made the native on-device screening
 * (MainActivity) the app's primary/launcher experience. Useful for the
 * full multi-modality dashboard + GIS surveillance view, which the
 * native screen does not attempt to replicate.
 *
 * NOT an offline native app -- the phone must be on the same network as
 * a machine running `.venv/Scripts/streamlit.exe run app/app.py`, and
 * reach it at the address entered below (Streamlit prints this as
 * "Network URL" on startup). The address is editable in-app and
 * persisted across launches since a laptop's LAN IP changes between
 * networks.
 */
class WebViewActivity : AppCompatActivity() {

    private val prefsName = "ai4tb_wrapper_prefs"
    private val urlKey = "server_url"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_webview)

        val webView = findViewById<WebView>(R.id.webView)
        val urlInput = findViewById<EditText>(R.id.serverUrlInput)
        val connectButton = findViewById<Button>(R.id.connectButton)

        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.webViewClient = WebViewClient()

        val prefs = getSharedPreferences(prefsName, MODE_PRIVATE)
        val savedUrl = prefs.getString(urlKey, getString(R.string.default_server_url))
        urlInput.setText(savedUrl)
        savedUrl?.let { webView.loadUrl(it) }

        connectButton.setOnClickListener {
            val url = urlInput.text.toString().trim()
            if (url.isNotEmpty()) {
                prefs.edit().putString(urlKey, url).apply()
                webView.loadUrl(url)
            }
        }
    }

    override fun onBackPressed() {
        val webView = findViewById<WebView>(R.id.webView)
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }
}
