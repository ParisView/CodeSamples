package com.code.sample.readmrz

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.widget.TextView


class InfoActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_info)
        val textView = findViewById<TextView>(R.id.textView)
        textView.movementMethod = ScrollingMovementMethod()
    }
}

