//
//  ViewController.m
//  RetinaVision
//
//  Created by Ryan Wong on 2017/07/04.
//  Copyright © 2017 Ryan Wong. All rights reserved.
//

#import <Photos/Photos.h>

#import <opencv2/core.hpp>
#import <opencv2/imgcodecs.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/imgproc.hpp>

#import "ViewController.h"
#import "VideoCamera.h"

@interface ViewController () <CvVideoCameraDelegate> {
    cv::Mat originalStillMat;
    cv::Mat updatedStillMatRetina;
    cv::Mat updatedStillMatRGBA;
    cv::Mat updatedVideoMatRetina;
    cv::Mat updatedVideoMatRGBA;
}

@property IBOutlet UIImageView *imageView;
@property IBOutlet UIActivityIndicatorView *activityIndicatorView;
@property IBOutlet UIToolbar *toolbar;

@property VideoCamera *videoCamera;
@property BOOL saveNextFrame;
@property int viewMode;
@property NSMutableArray* loc;
@property NSMutableArray* coeff;
@property NSMutableArray* L;
@property NSMutableArray* R;



- (IBAction)onTapToSetPointOfInterest:(UITapGestureRecognizer *)tapGesture;
- (IBAction)onColorModeSelected:(UISegmentedControl *)segmentedControl;
- (IBAction)onSwitchCameraButtonPressed;
- (IBAction)onSaveButtonPressed;

- (void)refresh;
- (void)processImage:(cv::Mat &)mat;
- (void)processImageHelper:(cv::Mat &)mat;
- (void)saveImage:(UIImage *)image;
- (void)showSaveImageFailureAlertWithMessage:(NSString *)message;
- (void)showSaveImageSuccessAlertWithImage:(UIImage *)image;
- (void)startBusyMode;
- (void)stopBusyMode;
- (void)LRsplit;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    UIImage *originalStillImage = [UIImage imageNamed:@"home.png"];
    UIImageToMat(originalStillImage, originalStillMat);
    
    self.videoCamera = [[VideoCamera alloc] initWithParentView:self.imageView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureSessionPreset  = AVCaptureSessionPresetHigh;
//    self.videoCamera.defaultFPS = 30;
    self.videoCamera.letterboxPreview = YES;

// ====================================================================================
    //    Reading loc file
    NSString* filePath = @"locFile";
    NSString* fileRoot = [[NSBundle mainBundle] pathForResource:filePath ofType:@"txt"];
    //
    NSString* fileContents = [NSString stringWithContentsOfFile:fileRoot encoding:NSUTF8StringEncoding error:nil];
    
    // array of lines
    NSArray* allLinedStrings = [fileContents componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    
    self.loc = [[NSMutableArray alloc]init];
    
    for (NSString *elements in allLinedStrings){
        NSArray* singleStrs = [elements componentsSeparatedByCharactersInSet: [NSCharacterSet characterSetWithCharactersInString:@"#"]];
        [self.loc addObject:singleStrs];
    }
    [self.loc removeLastObject];
//    NSLog(@"Test 1: %@",self.loc);
// ====================================================================================
    // Reading coeff file
    NSString* filePath2 = @"coeffFile";
    NSString* fileRoot2 = [[NSBundle mainBundle] pathForResource:filePath2 ofType:@"txt"];
    //
    NSString* fileContents2 = [NSString stringWithContentsOfFile:fileRoot2 encoding:NSUTF8StringEncoding error:nil];
    
    // array of lines
    NSArray* allLinedStrings2 = [fileContents2 componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    
    self.coeff = [[NSMutableArray alloc]init];
    
    NSMutableArray* temp = [[NSMutableArray alloc]init];
    
    for (NSString *elements in allLinedStrings2){
        if([elements isEqualToString:@"$"]){
            
            NSMutableArray *temp2 = [temp mutableCopy];
            [self.coeff addObject:temp2];
            [temp removeAllObjects];
        }
        else{
            NSArray* singleStrs = [elements componentsSeparatedByCharactersInSet: [NSCharacterSet characterSetWithCharactersInString:@"#"]];
            [temp addObject:singleStrs];
        }
    }
    
 //   NSLog(@"Test 2: %@",self.coeff);
    [self LRsplit];
    
}

-(void)LRsplit{
    self.L = [[NSMutableArray alloc]init];
    self.R = [[NSMutableArray alloc]init];
    
    
    for(int i=0;i<[self.loc count];i++){
        NSString *strFromI = [NSString stringWithFormat:@"%d",i];
//        [[self.loc objectAtIndex:0] replaceObjectAtIndex:2 withObject:strFromI];
        self.loc[i][2] = strFromI;
        if([self.loc[i][0] floatValue] < 0){
            [self.L addObject:[self.loc[i] subarrayWithRange:NSMakeRange(0,3)]];
        }
        else{
            [self.R addObject:[self.loc[i] subarrayWithRange:NSMakeRange(0,3)]];
        }
    }
//    NSLog(@"Left: %@",self.L);
//    NSLog(@"Right: %@",self.R);

}

-(void)viewDidLayoutSubviews{
    [super viewDidLayoutSubviews];
    
    switch ([UIDevice currentDevice].orientation){
        case UIDeviceOrientationPortraitUpsideDown:
            self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortraitUpsideDown;
            break;
        case UIDeviceOrientationLandscapeLeft:
            self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeLeft;
            break;
        case UIDeviceOrientationLandscapeRight:
            self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeRight;
            break;
        default:
            self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
            break;
    }
    
    [self refresh];
}

-(IBAction)onTapToSetPointOfInterest:(UITapGestureRecognizer *)tapGesture{
    if (tapGesture.state == UIGestureRecognizerStateEnded){
        if(self.videoCamera.running){
            CGPoint tapPoint = [tapGesture locationInView:self.imageView];
            [self.videoCamera setPointOfInterestInParentViewSpace:tapPoint];
        }
    }
}

-(IBAction)onColorModeSelected:(UISegmentedControl *)segmentedControl{
    switch (segmentedControl.selectedSegmentIndex){
        case 0:
            self.viewMode = 0;
            break;
        default:
            self.viewMode = 1;
            break;
    }
    [self refresh];
}

-(IBAction)onSwitchCameraButtonPressed  {

    if (self.videoCamera.running){
        switch(self.videoCamera.defaultAVCaptureDevicePosition){
            case AVCaptureDevicePositionFront:
                self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
                [self refresh];
                break;
            default:
                [self.videoCamera stop];
                [self refresh];
                break;
        }
    } else {
        // Hide the still image.
        self.imageView.image = nil;

        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
        [self.videoCamera start];
    }
}

-(void)refresh{
    if (self.videoCamera.running){
        // Hide the still image.
        self.imageView.image = nil;
        
        // Restart the video.
        [self.videoCamera stop];
        [self.videoCamera start];
    }
    else{
        // Refresh the still image.
        UIImage *image;
        //cv::cvtColor(originalStillMat, updatedStillMatRetina, cv::COLOR_RGBA2GRAY);
        originalStillMat.copyTo(updatedStillMatRetina);
        // processimage to update it.
        [self processImage:updatedStillMatRetina];
        image = MatToUIImage(updatedStillMatRetina);
        
        self.imageView.image = image;
    }
}

-(void)processImage:(cv::Mat &)mat {
    if (self.videoCamera.running) {
        switch (self.videoCamera.defaultAVCaptureVideoOrientation){
            case AVCaptureVideoOrientationLandscapeLeft:
            case AVCaptureVideoOrientationLandscapeRight:
                // The landscape video is captured upside-down.
                // Rotate it by 180 degrees
                cv::flip(mat, mat,-1);
                break;
            default:
                break;
        }
    }
    
    [self processImageHelper:mat];

    if (self.saveNextFrame){
        // The video frame, 'mat', is not safe for long running
        // operations such as saving to file. Thus, we copy its
        // data to another cv::Mat first.
        UIImage *image;
        if (self.viewMode == 0){
            mat.copyTo(updatedStillMatRetina);
            image = MatToUIImage(updatedStillMatRetina);
        } else {
            // TODO: Change to processImageHelper
            cv::cvtColor(mat, updatedVideoMatRGBA, cv::COLOR_BGRA2RGBA);
            
            image = MatToUIImage(updatedVideoMatRGBA);
        }
        [self saveImage:image];
        self.saveNextFrame = NO;
    }
}

-(void)processImageHelper:(cv::Mat &)mat{
    if (self.viewMode == 1){
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2GRAY);
    } else {
        
    }
}

-(void)startBusyMode{
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.activityIndicatorView startAnimating];
        for (UIBarItem *item in self.toolbar.items){
            item.enabled = NO;
        }
    });
}

-(void)stopBusyMode {
    dispatch_async(dispatch_get_main_queue(),^{
        [self.activityIndicatorView stopAnimating];
        for (UIBarItem *item in self.toolbar.items){
            item.enabled=YES;
        }
    });
}

-(IBAction)onSaveButtonPressed{
    [self startBusyMode];
    if(self.videoCamera.running){
        self.saveNextFrame = YES;
    } else {
        [self saveImage:self.imageView.image];
    }
}

-(void)saveImage:(UIImage *)image{
    // Try to save the image to a temporary file.
    NSString *outputPath = [NSString stringWithFormat:@"%@%@", NSTemporaryDirectory(),@"output.png"];
    if(![UIImagePNGRepresentation(image) writeToFile:outputPath atomically:YES]){
        // Show an alert describing the failure.
        [self showSaveImageFailureAlertWithMessage:@"The image could not be saved to the temporary directory."];
        return;
    }
    
    // Try to add the image to the Photos library.
    NSURL *outputURL = [NSURL URLWithString:outputPath];
    PHPhotoLibrary *photoLibrary = [PHPhotoLibrary sharedPhotoLibrary];
    [photoLibrary performChanges:^{[PHAssetChangeRequest creationRequestForAssetFromImageAtFileURL:outputURL];}  completionHandler:^(BOOL success, NSError *error){
        if (success) {
            // Show an alert describing the success, with sharing options
            [self showSaveImageSuccessAlertWithImage:image];
        } else {
            // Show an alert describing the failure.
            [self showSaveImageFailureAlertWithMessage:error.localizedDescription];
        }
    }];
}

-(void)showSaveImageFailureAlertWithMessage:(NSString *)message {
    UIAlertController* alert = [UIAlertController alertControllerWithTitle:@"Failed to save image" message:message preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction* okAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action){
        [self stopBusyMode];
    }];
    [alert addAction:okAction];
    [self presentViewController:alert animated:YES completion:nil];
}

-(void)showSaveImageSuccessAlertWithImage:(UIImage *)image {
    // Create a "Saved image" alert.
    UIAlertController* alert = [UIAlertController alertControllerWithTitle:@"Saved image" message :@"The image has been added to your Photos library." preferredStyle:UIAlertControllerStyleAlert];
    
    
    UIAlertAction* okAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action){
        [self stopBusyMode];
    }];
    [alert addAction:okAction];

    // Show the alert.
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}





@end
