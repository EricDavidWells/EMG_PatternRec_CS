# RTPR_CSharp
RTPR_Csharp is a class library to easily facilitate real-time electromyography based machine learning systems.  Input signals are split into two categories (EMG or Generic) with seperate feature mapping, scaling, and filtering pipelines.

# Feature Extraction Techniques
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
* Mean Value
* Raw Value

# Filter Implementations
* Notch Filter
* High-pass Butterworth Filter
* Low-pass Butterworth Filter
* Moving Average Filter

# Pre-processing Techniques
* Min-Max Scaling
* Zero-Shifting

# Post-processing Techniques
* Majority Voting
* Velocity Ramping

# Prediction Models
* SVM - training and real-time output
* LDA - training and real-time output
* Open Neural Network Exchange (ONNX) - real-time output

# Dependencies
* .NET Framework 4.7.2
* Accord.Net v3.8.0
* ML.Net v1.5.4
* NWaves v0.9.4

# Build Instructions
* Clone repository
* Open solution file in Visual Studio 2017 or higher
* In Solution Explorer, right click on solution, select "Restore NuGet Packages"
* Right click on ExampleUse project, select "Set as StartUp Project"
* Click "Start" button to run tests
