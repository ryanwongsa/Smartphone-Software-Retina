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
    cv::Mat L;
    cv::Mat R;
    cv::Mat L_loc;
    cv::Mat R_loc;
}

@property IBOutlet UIImageView *imageView;
@property IBOutlet UIActivityIndicatorView *activityIndicatorView;
@property IBOutlet UIToolbar *toolbar;

@property VideoCamera *videoCamera;
@property BOOL saveNextFrame;
@property int viewMode;
@property NSMutableArray* loc;
@property NSMutableArray* coeff;
//@property NSMutableArray* L;
//@property NSMutableArray* R;



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
- (void)cort_map:(int)alpha;
- (double)cdist:(cv::Mat &)mat;

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
    
    [self cort_map:15];
    
}

-(void)LRsplit{
    NSMutableArray* left = [[NSMutableArray alloc]init];
    NSMutableArray* right = [[NSMutableArray alloc]init];
    
    
    for(int i=0;i<[self.loc count];i++){
        NSString *strFromI = [NSString stringWithFormat:@"%d",i];
//        [[self.loc objectAtIndex:0] replaceObjectAtIndex:2 withObject:strFromI];
        self.loc[i][2] = strFromI;
        if([self.loc[i][0] floatValue] < 0){
            [left addObject:[self.loc[i] subarrayWithRange:NSMakeRange(0,3)]];
            
        }
        else{
            [right addObject:[self.loc[i] subarrayWithRange:NSMakeRange(0,3)]];
            
        }
    }
    L = cv::Mat([left count],3, CV_32F);
    for(int i=0;i<[left count];i++){
        for(int j=0;j<3;j++){
            L.at<float>(i,j)=[left[i][j] floatValue];
        }
    }
//    print(L);
    
    R = cv::Mat([right count],3, CV_32F);
    for(int i=0;i<[right count];i++){
        for(int j=0;j<3;j++){
            R.at<float>(i,j)=[right[i][j] floatValue];
        }
    }
    
}



-(void)cort_map:(int)alpha{
    NSLog(@"Alpha value is: %d",alpha);
    
    // Calculating L_r
    cv::Mat L_r;
    cv::Mat L0 = L.col(0)-alpha;
    cv::pow(L0, 2, L0);
    cv::Mat L1;
    L.col(1).copyTo(L1);
    cv::pow(L1, 2, L1);
    cv::add(L0,L1,L_r);
    cv::sqrt(L_r, L_r);
    
    
    // Calculating R_r
    cv::Mat R_r;
    cv::Mat R0 = R.col(0)+alpha;
    cv::pow(R0, 2, R0);
    cv::Mat R1;
    R.col(1).copyTo(R1);
    cv::pow(R1, 2, R1);
    cv::add(R0,R1,R_r);
    cv::sqrt(R_r, R_r);
//    print(R_r);
    
    //  Calculating L_theta
    cv::Mat L_theta;

    cv::phase(L.col(0)-alpha, L.col(1), L_theta);
    L_theta = L_theta - M_PI;
//    print(L_theta);

    //  Calculating R_theta
    cv::Mat R_theta;
    
    cv::phase(R.col(0)+alpha, R.col(1), R_theta);
    
    for(int i =0;i<R_theta.rows;i++){
        while(R_theta.at<float>(i) > M_PI/2){
            R_theta.at<float>(i)=R_theta.at<float>(i)-M_PI;
        }
    }
    cv::hconcat(L_theta, L_r, L_loc);
    cv::hconcat(R_theta, R_r, R_loc);
    
//    print(L_loc);
    
    // Calculating x (theta)
    cv::Mat L_theta2;
    L_loc.copyTo(L_theta2);
    cv::Mat R_theta2;
    R_loc.copyTo(R_theta2);
    
    L_theta2.col(1)=0;
    R_theta2.col(1)=0;
    
    
    double L_theta_mean = [self cdist:L_theta2];
    NSLog(@"Left Mean: %f",L_theta_mean);
    
    
    double R_theta_mean = [self cdist:R_theta2];
    NSLog(@"Right Mean: %f",R_theta_mean);
    
    double xd = (L_theta_mean+R_theta_mean)/2;
    
    // Calculating y (r)
    cv::Mat L_r2;
    L_loc.copyTo(L_r2);
    cv::Mat R_r2;
    R_loc.copyTo(R_r2);
    
    L_r2.col(0)=0;
    R_r2.col(0)=0;
    
    double L_r_mean = [self cdist:L_r2];
    NSLog(@"Left Mean: %f",L_r_mean);

    double R_r_mean = [self cdist:R_r2];
    NSLog(@"Right Mean: %f",R_r_mean);
    
    double yd = (L_r_mean+R_r_mean)/2;
    
    // Scale theta (x)
    L_loc.col(0)*= yd/xd;
    R_loc.col(0)*= yd/xd;
    
//    print(R_loc);
    
}

-(double)cdist:(cv::Mat &)mat{
    double sum=0;
    for(int i=0;i<mat.rows;i++){
        for(int j=0;j<mat.rows;j++){
            double result = cv::norm(mat.row(i), mat.row(j));
            sum+=result;
        }
    }
    double mean = sum / (mat.rows*mat.rows);
    return mean;
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
