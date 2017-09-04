# Smartphone-Software-Retina
iOS application to simulate human retina vision

## Requirements to install application
- iOS 10.1 or newer
- iPhone 5 or newer

## Requirements to develop application
- Xcode 8.0 or newer
- OpenCV 3.1 or newer (including contrib)
- Apple Developer Account

## Set-up Components

### 1. Setting up environment

- Add files to project =>
  <app_project_path>/opencv2.framework

- File -> Add File to <Project> => Supporting files folder for <images>

### 2. Configure the project

- <Project> General -> Deployment info
  - Hide status bar
  - Requires full screen

- Info.plist
  - +View Controller-based status bar appearance -> NO
  - Required device capabilities
    - item 1 -> video-camera
  - +Privacy - Photo Library Usage Description -> $(PRODUCT_NAME) photo use
  - +Privacy - Camera Usage Description -> $(PRODUCT_NAME) camera use

- <Project> Build Phases
  - Link Binary With Libaries
    - +Accelerate.framework
    - +AssetsLibrary.framework
    - +CoreGraphics.framework
    - +CoreMedia.framework
    - +CoreVideo.framework
    - +Photos.framework
    - +QuartzCore.framework
    - +Social.framework
    - +UIKit.framework
    - opencv2.framework
    
- <Project> Build Settings
  - Apple LLVM <no.> - Language => Compile Source as Objective C++
  - Apple LLVM <no.> - Preprocessing -> Preprocessor Macro
    - Debug WITH_OPENCV_CONTRIB DEBUG=1
    - Release WITH_OPENCV_CONTRIB Any Architecture | Any SDK

## Versions

- RetinaVision
	- most basic retina version with no optmisations

- RetinaVisionPrep
	- retina vision with preprocessed files (generated offline) to reduce runtime computation

- RetinaVisionMovableCortex
	- Similar setup as to RetinaVisionPrep
	- Added functionality for gaze control


