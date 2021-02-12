# RTPR_CSharp
RTPR_Csharp is a class library to easily facilitate real-time electromyography based machine learning systems.  Input signals are split into two categories (EMG or Generic) with seperate feature mapping pipelines.

# EMG Features
* Integrated EMG
* Mean Absolute Value
* Mean Modified Absolute Value 1
* Mean Modified Absolute Value 2
* Simple Square Integral
* Variance
* Zero Crossings
* Slope Sign Changes
* Waveform Length
* Wilson Amplitude
* Raw Value

# Generic Features
Meav Value
Raw Value

# Models
* SVM - training and real-time output
* LDA - training and real-time output
* Open Neural Network Exchange (ONNX) - real-time output

# Dependencies
* Accord.Net v3.8.0
* ML.Net v1.5.4
