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
#import <opencv2/highgui/highgui.hpp>
#import <opencv2/xfeatures2d.hpp>

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
    cv::Mat G[10][10];
    cv::Mat GI;
    cv::Mat loc;
    cv::Mat coeff[8192]; // Hardcoded value, note need to change this depending on coeff file.
}

@property IBOutlet UIImageView *imageView;
@property IBOutlet UIActivityIndicatorView *activityIndicatorView;
@property IBOutlet UIToolbar *toolbar;

@property VideoCamera *videoCamera;
@property BOOL saveNextFrame;
@property int viewMode;
@property NSMutableArray* cort_size;
@property BOOL rotated;
@property BOOL alreadyLoaded;
@property BOOL started;
@property int x;
@property int y;
@property int retinaRadius;
// if 0 then it was previously going left, if 1 then it was previously going right.
@property int gazePrev;
@property float xd;
@property float yd;
@property float l_min0;
@property float l_min1;
@property float r_min0;
@property float r_min1;

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
- (cv::Mat)retina_sample:(int)x y:(int)y mat:(cv::Mat &)mat;
- (cv::Mat)cort_img:(cv::Mat &)V k_width:(int)k_width sigma:(float)sigma;
- (cv::Mat)inverse:(cv::Mat &)V x:(int)x y:(int)y shape0:(int)shape0 shape1:(int)shape1;
- (void)gauss_norm_img:(int)x y:(int)y shape0:(int)shape0 shape1:(int)shape1;
- (void)create_new_focal_point:(cv::Mat &)cortImg mat:(cv::Mat &) mat;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    UIImage *originalStillImage = [UIImage imageNamed:@"leopard.jpg"];
    UIImageToMat(originalStillImage, originalStillMat);
    
    self.videoCamera = [[VideoCamera alloc] initWithParentView:self.imageView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureSessionPreset  = AVCaptureSessionPreset1280x720;
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
    
    
    NSString* strsInOneLine = [allLinedStrings objectAtIndex:0];
    NSArray* singleStr = [strsInOneLine componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    int shape0 = [singleStr[0] intValue];
    int shape1 = [singleStr[1] intValue];
    NSLog(@"%d %d", shape0, shape1);
    loc = cv::Mat(shape0, shape1, CV_32F,0.0);
    
    for(int i=1;i< (int)([allLinedStrings count]-1);i++){
        NSString* element = allLinedStrings[i];
        NSArray* singleStrs = [element componentsSeparatedByCharactersInSet: [NSCharacterSet characterSetWithCharactersInString:@"#"]];
        for(int j=0; j<(int)([singleStrs count]-1); j++){
            loc.at<float>(i-1,j) = [singleStrs[j] floatValue];
        }
    }
    
    
    NSString* filePath2 = @"coeffFile";
    NSString* fileRoot2 = [[NSBundle mainBundle] pathForResource:filePath2 ofType:@"txt"];
    NSString* fileContents2 = [NSString stringWithContentsOfFile:fileRoot2 encoding:NSUTF8StringEncoding error:nil];
    //
    //        // array of lines
    NSArray* allLinedStrings2 = [fileContents2 componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    int countCoeff=0;
    for(int i=1;i< (int)([allLinedStrings2 count]);i++){
        NSString* elements = allLinedStrings2[i];
        if([elements containsString:@"$"]){
            NSArray* singleStrs = [elements componentsSeparatedByCharactersInSet: [NSCharacterSet characterSetWithCharactersInString:@"$"]];
            //                NSLog(@"%@",singleStrs[1]);
            int matSize = [singleStrs[1] intValue];
            cv::Mat tempMat;
            tempMat = cv::Mat(matSize, matSize, CV_32F,0.0);
            int count=0;
            for(int j=i+1;j<=i+matSize;j++){
                elements = allLinedStrings2[j];
                NSArray* rows = [elements componentsSeparatedByCharactersInSet: [NSCharacterSet characterSetWithCharactersInString:@"#"]];
                //                    NSLog(@"%@", rows);
                for(int k=0;k<(int)[rows count];k++){
                    tempMat.at<float>(count,k) = [rows[k] floatValue];
                }
                count++;
            }
            i=count+i-1;
            tempMat.copyTo(coeff[countCoeff]);
            countCoeff++;
        }
    }
    
    //        L file
    
    NSString* filePath3 = @"L_file";
    NSString* fileRoot3 = [[NSBundle mainBundle] pathForResource:filePath3 ofType:@"txt"];
    NSString* fileContents3 = [NSString stringWithContentsOfFile:fileRoot3 encoding:NSUTF8StringEncoding error:nil];
    //
    //        // array of lines
    NSArray* allLinedStrings3 = [fileContents3 componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSString* strsInOneLine3 = [allLinedStrings3 objectAtIndex:0];
    NSArray* singleStr3 = [strsInOneLine3 componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    shape0 = [singleStr3[0] intValue];
    shape1 = [singleStr3[1] intValue];
    
    L = cv::Mat(shape0,shape1,CV_32F ,0.0);
    
    for(int i=1;i< (int)([allLinedStrings3 count]-1);i++){
        NSString* elements = allLinedStrings3[i];
        //            NSLog(@"%@", elements);
        NSArray* singleStrs = [elements componentsSeparatedByCharactersInSet: [NSCharacterSet characterSetWithCharactersInString:@"#"]];
        //            NSLog(@"%@",singleStrs);
        for(int j=0;j<(int)[singleStrs count];j++){
            L.at<float>(i-1,j) = [singleStrs[j] floatValue];
        }
    }
    
    // R file
    
    NSString* filePath4 = @"R_file";
    NSString* fileRoot4 = [[NSBundle mainBundle] pathForResource:filePath4 ofType:@"txt"];
    NSString* fileContents4 = [NSString stringWithContentsOfFile:fileRoot4 encoding:NSUTF8StringEncoding error:nil];
    //
            // array of lines
    NSArray* allLinedStrings4 = [fileContents4 componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSString* strsInOneLine4 = [allLinedStrings4 objectAtIndex:0];
    NSArray* singleStr4 = [strsInOneLine4 componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    shape0 = [singleStr4[0] intValue];
    shape1 = [singleStr4[1] intValue];
    
    R = cv::Mat(shape0,shape1,CV_32F ,0.0);
    
    for(int i=1;i< (int)([allLinedStrings4 count]-1);i++){
        NSString* elements = allLinedStrings4[i];
        //            NSLog(@"%@", elements);
        NSArray* singleStrs = [elements componentsSeparatedByCharactersInSet: [NSCharacterSet characterSetWithCharactersInString:@"#"]];
        //            NSLog(@"%@",singleStrs);
        for(int j=0;j<(int)[singleStrs count];j++){
            R.at<float>(i-1,j) = [singleStrs[j] floatValue];
        }
    }
    
    //        print(R);
    
    
    // L loc file
    
    NSString* filePath5 = @"L_loc_file";
    NSString* fileRoot5 = [[NSBundle mainBundle] pathForResource:filePath5 ofType:@"txt"];
    NSString* fileContents5 = [NSString stringWithContentsOfFile:fileRoot5 encoding:NSUTF8StringEncoding error:nil];
    //
    // array of lines
    NSArray* allLinedStrings5 = [fileContents5 componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSString* strsInOneLine5 = [allLinedStrings5 objectAtIndex:0];
    NSArray* singleStr5 = [strsInOneLine5 componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    shape0 = [singleStr5[0] intValue];
    shape1 = [singleStr5[1] intValue];
    
    strsInOneLine5 = [allLinedStrings5 objectAtIndex:1];
    singleStr5 = [strsInOneLine5 componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    
    self.yd = [singleStr5[0] floatValue];
    self.xd = [singleStr5[1] floatValue];
    
    strsInOneLine5 = [allLinedStrings5 objectAtIndex:2];
    singleStr5 = [strsInOneLine5 componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    
    self.l_min0 = [singleStr5[0] floatValue];
    self.l_min1 = [singleStr5[1] floatValue];
    
    
    L_loc = cv::Mat(shape0,shape1,CV_32F ,0.0);
    
    for(int i=3;i< (int)([allLinedStrings5 count]-1);i++){
        NSString* elements = allLinedStrings5[i];
        //            NSLog(@"%@", elements);
        NSArray* singleStrs = [elements componentsSeparatedByCharactersInSet: [NSCharacterSet characterSetWithCharactersInString:@"#"]];
        //            NSLog(@"%@",singleStrs);
        for(int j=0;j<(int)[singleStrs count];j++){
            L_loc.at<float>(i-3,j) = [singleStrs[j] floatValue];
        }
    }
    //        print(L_loc);
    
    // R loc file
    
    NSString* filePath6 = @"R_loc_file";
    NSString* fileRoot6 = [[NSBundle mainBundle] pathForResource:filePath6 ofType:@"txt"];
    NSString* fileContents6 = [NSString stringWithContentsOfFile:fileRoot6 encoding:NSUTF8StringEncoding error:nil];
    //
    //        // array of lines
    NSArray* allLinedStrings6 = [fileContents6 componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSString* strsInOneLine6 = [allLinedStrings6 objectAtIndex:0];
    NSArray* singleStr6 = [strsInOneLine6 componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    shape0 = [singleStr6[0] intValue];
    shape1 = [singleStr6[1] intValue];
    
    
    strsInOneLine6 = [allLinedStrings6 objectAtIndex:2];
    singleStr6 = [strsInOneLine6 componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    
    self.r_min0 = [singleStr6[0] floatValue];
    self.r_min1 = [singleStr6[1] floatValue];
    NSLog(@"%f %f %f %f %f %f", self.yd, self.xd,self.l_min0, self.l_min1, self.r_min0, self.r_min1);
    
    R_loc = cv::Mat(shape0,shape1,CV_32F ,0.0);
    
    for(int i=3;i< (int)([allLinedStrings6 count]-1);i++){
        NSString* elements = allLinedStrings6[i];
        //            NSLog(@"%@", elements);
        NSArray* singleStrs = [elements componentsSeparatedByCharactersInSet: [NSCharacterSet characterSetWithCharactersInString:@"#"]];
        //            NSLog(@"%@",singleStrs);
        for(int j=0;j<(int)[singleStrs count];j++){
            R_loc.at<float>(i-3,j) = [singleStrs[j] floatValue];
        }
    }
    

    
    
    // cort_size
    NSString* filePath8 = @"cort_size_file";
    NSString* fileRoot8 = [[NSBundle mainBundle] pathForResource:filePath8 ofType:@"txt"];
    NSString* fileContents8 = [NSString stringWithContentsOfFile:fileRoot8 encoding:NSUTF8StringEncoding error:nil];
    
    NSArray* allLinedStrings8 = [fileContents8 componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    
    self.cort_size = [[NSMutableArray alloc] init];
    self.cort_size[0] = allLinedStrings8[0];
    self.cort_size[1] = allLinedStrings8[1];
    NSLog(@"%@", self.cort_size);
    
    // G
    NSString* filePath9 = @"G_file";
    NSString* fileRoot9 = [[NSBundle mainBundle] pathForResource:filePath9 ofType:@"txt"];
    NSString* fileContents9 = [NSString stringWithContentsOfFile:fileRoot9 encoding:NSUTF8StringEncoding error:nil];
    
    NSArray* allLinedStrings9 = [fileContents9 componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    
    NSString* strsInOneLine9 = [allLinedStrings9 objectAtIndex:0];
    NSArray* singleStr9 = [strsInOneLine9 componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    shape0 = [singleStr9[2] intValue];
    shape1 = [singleStr9[3] intValue];
    
    int first = -1;
    int second = 0;
    cv::Mat tempMat;
    int counter=0;
    
    for(int i=1;i< (int)([allLinedStrings9 count]-1);i++){
        NSString* elements = allLinedStrings9[i];

        if([elements isEqualToString:@"$"]){
            first++;
            second=0;
            tempMat = cv::Mat(shape0,shape1,CV_32F,0.0);
        }
        
        if([elements isEqualToString:@"&"]){
//            NSLog(@"%d %d", first, second);
            tempMat.copyTo(G[first][second]);
            second++;
            counter=0;
        }
        
        
        
        NSArray* parts = [elements componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"#"]];
        for(int j=0;j<(int)[parts count];j++){
            tempMat.at<float>(counter,j) = [parts[j] floatValue];
        }
        
        if(![elements isEqualToString:@"&"] && ![elements isEqualToString:@"$"]){
            counter++;
        }
        
        
    }
    
    self.started=false;
    
    
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
        case 1:
            self.viewMode = 1;
            break;
        case 2:
            self.viewMode = 2;
            break;
        case 3:
            self.viewMode = 3;
            break;
        default:
            self.viewMode = 0;
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
        
        NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
        for (AVCaptureDevice *device in devices)
        {
            if ([device position] == AVCaptureDevicePositionBack) {
                if ([device lockForConfiguration:nil]) {
                    device.autoFocusRangeRestriction = AVCaptureAutoFocusRangeRestrictionNone;
                    [device unlockForConfiguration];
                }
            }
        }
        
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

-(void)gauss_norm_img:(int)x y:(int)y shape0:(int)shape0 shape1:(int)shape1{
    GI = cv::Mat(shape0,shape1,CV_32F,0.0);
    
    int s = (int)loc.rows;
    cv::Mat X = loc.col(0)+x;
    cv::Mat Y = loc.col(1)+y;
    
    for(int i=s-1;i>-1;i--){
        float loci6 = loc.at<float>(i,6);
        float yi = Y.at<float>(i);
        float xi = X.at<float>(i);
        int y1 = yi-loci6/2+0.5;
        if(y1<0){
            y1=0;
        } else if(y1>=shape0){
            y1=shape0-1;
        }
        int y2 = (yi+loci6/2+0.5);
        if(y2<0){
            y2=0;
        } else if(y2>=shape0){
            y2=shape0-1;
        }
        
        int x1 = (xi-loci6/2+0.5);
        if(x1<0){
            x1=0;
        } else if(x1>=shape1){
            x1=shape1-1;
        }
        
        int x2 = (xi+loci6/2+0.5);
        if(x2<0){
            x2=0;
        } else if(x2>=shape1){
            x2=shape1-1;
        }
        
        //        NSLog(@"%d %d %d %d", x1, x2, y1, y2);
        cv::Mat GI_roi = GI(cv::Rect(x1,y1,x2-x1,y2-y1));
        
        
        int shape0_coeff =coeff[i].rows;
        int shape1_coeff =coeff[i].cols;
        
        int begX = roundf(shape1_coeff/2.0-(x2-x1)/2.0);
        int endX = roundf(shape1_coeff/2.0+(x2-x1)/2.0);
        
        int begY = shape0_coeff/2.0-(y2-y1)/2.0;
        int endY = shape0_coeff/2.0+(y2-y1)/2.0;
        
        //        NSLog(@"%d %d %d %d", begX, endX, begY, endY);
        
        cv::Mat coeff_roi =coeff[i](cv::Rect(begX,begY,endX-begX,endY-begY));
        
        cv::add(GI_roi,coeff_roi,GI_roi);
    }
}

-(void)create_new_focal_point:(cv::Mat &)cortImg mat:(cv::Mat &)mat{
    std::vector<cv::KeyPoint> keypointsLeft;
    cv::Mat cortLeft_roi = cortImg(cv::Rect(15,15,cortImg.cols/2-30,cortImg.rows-30));
    cv::FAST(cortLeft_roi, keypointsLeft, 20);
    
    
    
    std::vector<cv::KeyPoint> keypointsRight;
    cv::Mat cortRight_roi = cortImg(cv::Rect(cortImg.cols/2+15,15,cortImg.cols/2-30,cortImg.rows-30));
    cv::FAST(cortRight_roi, keypointsRight, 20);
    
    
    // - reponse for left kp's
    for(auto &kp : keypointsLeft){
        kp.response = -1*kp.response;
        kp.pt.x = kp.pt.x+15;
        kp.pt.y = kp.pt.y+15;
        
        // distance
        float dx = kp.pt.x - cortImg.cols/2;
        float dy = kp.pt.y - cortImg.rows/2;
        float dist = sqrt(dx*dx + dy*dy);
//        NSLog(@"Distance is : %f",dist);
//        kp.response=-1*(std::abs(kp.response)+dist);
        kp.response*=dist;
        
    }
    
    for(auto &kp : keypointsRight){
        kp.pt.x = kp.pt.x+15;
        kp.pt.y = kp.pt.y+15;
        
        // distance
        float dx = kp.pt.x - cortImg.cols/2;
        float dy = kp.pt.y - cortImg.rows/2;
        float dist = sqrt(dx*dx + dy*dy);
//        NSLog(@"Distance is : %f",dist);
//        kp.response=kp.response+dist;
        kp.response*=dist;
    }
    
    std::vector<cv::KeyPoint> keypointsAll;
    keypointsAll.reserve(keypointsLeft.size()+keypointsRight.size());
    keypointsAll.insert(keypointsAll.end(),keypointsLeft.begin(),keypointsLeft.end());
    keypointsAll.insert(keypointsAll.end(),keypointsRight.begin(),keypointsRight.end());
    
    
    std::sort(keypointsAll.begin(), keypointsAll.end(), [](cv::KeyPoint a, cv::KeyPoint b) { return std::abs(a.response) > std::abs(b.response);});
    
    BOOL found=false;
    
    // using probability to switch gaze direction
    int prob = arc4random() % 100;
    if(prob <= 25){
        NSLog(@"%@",@"Changing gaze direction");
        if(self.gazePrev==1){
            self.gazePrev=0;
        }else if(self.gazePrev==0){
            self.gazePrev=1;
        }
    }
    
    // starting HERE
    int leftorright=-1;
    for(cv::KeyPoint kp : keypointsAll){
//    cv::KeyPoint kp = keypointsAll[0];
    
    // adding point here for seeing it
//    kp.pt.x = 10;//cortImg.cols/2-15;
//    kp.pt.y = 0;//cortImg.rows/4;
//    kp.response=-1;
    
        
        cv::cvtColor(cortImg, cortImg, cv::COLOR_GRAY2RGB);

        cv::Mat locHere;
        if(kp.response<=0){
            NSLog(@"%@",@"LEFT");
            locHere=L_loc;
            leftorright=0;
            cv::Vec3b keyVal = cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x);
            keyVal[0]=255;
            keyVal[1]=0;
            keyVal[2]=0;
            cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x) = keyVal;
        }
        else{
            NSLog(@"%@",@"RIGHT");
            locHere=R_loc;
            leftorright=1;
            cv::Vec3b keyVal = cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x+cortImg.cols/2.0);
            keyVal[0]=255;
            keyVal[1]=0;
            keyVal[2]=0;
            cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x+cortImg.cols/2.0) = keyVal;
        }
        
        
//        for(int i=0;i<locHere.rows;i++){
//            int xLocI_ceil =(int)ceilf(locHere.at<float>(i,0));
//            int yLocI_ceil =(int)ceilf(locHere.at<float>(i,1));
//            int xLocI_floor =(int)floorf(locHere.at<float>(i,0));
//            int yLocI_floor =(int)floorf(locHere.at<float>(i,1));
//        int xVal = (int)kp.pt.x;
//        int yVal = (int)kp.pt.y;
    kp.pt.y = -kp.pt.y;
        NSLog(@"%d ===== %f %f", leftorright, kp.pt.x, kp.pt.y);

        cv::Point2f center((cortImg.cols/2.0-1)/2.0, -(cortImg.rows-1)/2.0);
    NSLog(@"Center is : %f %f",  center.x, center.y);
        cv::Point2f trans_pt = kp.pt - center;
    NSLog(@"new point wrt center is : %f %f",  trans_pt.x, trans_pt.y);
        
    float xTemp;
    float yTemp;
        if(leftorright==0){
//            rotation = -1.0/2.0*M_PI;
            xTemp = trans_pt.y;
            yTemp = -trans_pt.x;
            NSLog(@"check with next: %f %f", trans_pt.y, -trans_pt.x);
        }
        else{
            xTemp = -trans_pt.y;
            yTemp = trans_pt.x;
//            rotation = 3.0/2.0*M_PI;
        }
//        float xTemp = std::cos(rotation) * trans_pt.x - std::sin(rotation) * trans_pt.y;
//        float yTemp = std::sin(rotation) * trans_pt.x - std::cos(rotation) * trans_pt.y;
        cv::Point2f rot_p(xTemp, yTemp);
    NSLog(@" rotated %f %f", rot_p.x, rot_p.y);
        cv::Point2f fin_pt = rot_p + center;
    fin_pt.y=-fin_pt.y;
        NSLog(@"%f %f", fin_pt.x, fin_pt.y);
    
//    fin_pt.x = 24.4552818404;//cortImg.cols/2-15;
//    fin_pt.y = 5.577756237;//cortImg.rows-3;//cortImg.rows/4;
    NSLog(@"TEST 1 === %f %f",fin_pt.x, fin_pt.y);
    // scaling factor fixing
        fin_pt = fin_pt*2.0;
        NSLog(@"%f %f", fin_pt.x, fin_pt.y);
    NSLog(@"TEST 2 === %f %f",fin_pt.x, fin_pt.y);
    // k_width fixing
        fin_pt.x = fin_pt.x - 7;
        fin_pt.y = fin_pt.y - 7;
        NSLog(@"%f %f", fin_pt.x, fin_pt.y);
    
    NSLog(@"TEST 3 === %f %f",fin_pt.x, fin_pt.y);
    if(leftorright==0){
        fin_pt.y = self.l_min1 + fin_pt.y;
        fin_pt.x = self.l_min0 + fin_pt.x;
    }
    else{
        NSLog(@"Ry_MIN : %f", self.r_min1);
        fin_pt.y = self.r_min1 + fin_pt.y;
        fin_pt.x = self.r_min0 + fin_pt.x;
    }
    NSLog(@"%f %f", fin_pt.x, fin_pt.y);
    NSLog(@"TEST 4 === %f %f",fin_pt.x, fin_pt.y);
    
    fin_pt.y = - fin_pt.y;
    NSLog(@"TEST 5 === %f %f",fin_pt.x, fin_pt.y);
    fin_pt.x = fin_pt.x * self.xd / self.yd;
    NSLog(@"xdyd fixed: %f %f", fin_pt.x, fin_pt.y);
    NSLog(@"TEST 6 === %f %f",fin_pt.x, fin_pt.y);
    
    
    float theta=0;
    float r = 0;
        float finalX=0;
        float finalY=0;
    if(leftorright==0){
        if(fin_pt.x>0){
            NSLog(@"We are in the 3rd quadrant %@",@"3");
            theta= fin_pt.x+M_PI;
//            r =fin_pt.y;
//            NSLog(@"theta, r: %f %f", theta, r);
//            
//            float tan_theta = std::tan(theta);
//            NSLog(@"theta: %f", tan_theta);
//            
//            float finalX = -std::sqrt((r*r)/(1+tan_theta*tan_theta));//+15;
////            finalX = -(self.retinaRadius + finalX);
//            float finalY = tan_theta*(finalX);//-15);
////            NSLog(@"FINAL: %f %f", finalX, -finalY);
//            finalX = -(self.retinaRadius + finalX+15);
//            finalY = finalY-15;
//            
//            
//            NSLog(@"FINAL: %f %f", finalX, -finalY);
//            self.x = self.x + finalX;
//            self.y = self.y - finalY;
        }
        else{
            NSLog(@"We are in the 2rd quadrant %@",@"2");
            theta= fin_pt.x-M_PI;
            
        }
        r =fin_pt.y;
        NSLog(@"theta, r: %f %f", theta, r);
        
        float tan_theta = std::tan(theta);
        NSLog(@"theta: %f", tan_theta);
        
        finalX = -std::sqrt((r*r)/(1+tan_theta*tan_theta));//+15;
        //            finalX = -(self.retinaRadius + finalX);
        finalY = tan_theta*(finalX);//-15);
        finalX = finalX+15;
        NSLog(@"FINAL: %f %f", finalX, finalY);
        //            finalX = -(self.retinaRadius + finalX+15);
        
        
        
        NSLog(@"FINAL: %f %f", finalX, finalY);
//        self.x = self.x + finalX;
//        self.y = self.y - finalY;
    }
    else{
        theta = fin_pt.x;
        r =fin_pt.y;
        NSLog(@"theta, r: %f %f", theta, r);
        
        if(fin_pt.x>0){
            NSLog(@"We are in the 1st quadrant %@",@"1");
            
        }
        else{
            NSLog(@"We are in the 4th quadrant %@",@"4");
//            float tan_theta = std::tan(theta);
//            NSLog(@"theta: %f", tan_theta);
//            float finalX = std::sqrt((r*r)/(1+tan_theta*tan_theta))-15;
//            float finalY = tan_theta*(finalX+15);
//            
//            NSLog(@"FINAL: %f %f", finalX, -finalY);
//            self.x = self.x + finalX;
//            self.y = self.y - finalY;
            
        }
        
        float tan_theta = std::tan(theta);
        NSLog(@"theta: %f", tan_theta);
        finalX = std::sqrt((r*r)/(1+tan_theta*tan_theta));//-15;
        finalY = tan_theta*(finalX);
        finalX = finalX-15;
        
        NSLog(@"FINAL: %f %f", finalX, finalY);
//        self.x = self.x + finalX;
//        self.y = self.y - finalY;
    }
//          self.x = self.x + finalX; // remove this
//          self.y = self.y + finalY; // remove this
        int tempInverseX = self.x + finalX;
        int tempInverseY = self.y + finalY;
        NSLog(@"FINAL ON ORIGINAL: %d %d %d %d", self.x, self.y ,tempInverseX, tempInverseY);
        if((tempInverseX>self.retinaRadius)&&(tempInverseY>self.retinaRadius)&&(tempInverseX<mat.cols-self.retinaRadius) && (tempInverseY<mat.rows-self.retinaRadius) ){
            NSLog(@"WE HAVE MADE IT TO BEFORE THE END! %@",@"");
                            int tempGaze;
                            // determine gaze direction
                            if( kp.response < 0 ){
                                tempGaze = 0;
                            }
                            else{
                                tempGaze = 1;
                            }
            NSLog(@"WE HAVE MADE IT TO SLIGHTLY BEFORE THE END! %@",@"");

        
                            if((tempGaze==self.gazePrev) ||(self.gazePrev==-1)){
        
                                self.x = tempInverseX;
                                self.y = tempInverseY;
                                found=true;
                                self.gazePrev =tempGaze;
                                NSLog(@"WE HAVE MADE IT TO THE END! %@",@"");
                                break;
                            }
        }
    }
    
    if(!found){
        if(self.gazePrev==1){
            self.gazePrev=0;
        }else if(self.gazePrev==0){
            self.gazePrev=1;
        }
    }
    
//        for(int i=0;i<locHere.rows;i++){
//            int xLocI_ceil =(int)ceilf(locHere.at<float>(i,0));
//            int yLocI_ceil =(int)ceilf(locHere.at<float>(i,1));
//            int xLocI_floor =(int)floorf(locHere.at<float>(i,0));
//            int yLocI_floor =(int)floorf(locHere.at<float>(i,1));
//            int xVal = (int)kp.pt.x;
//            int yVal = (int)kp.pt.y;
//            if(( (xLocI_ceil==xVal) && (yLocI_ceil==yVal) )||( (xLocI_ceil==xVal) && (yLocI_floor==yVal) )||( (xLocI_floor==xVal) && (yLocI_ceil==yVal) )||( (xLocI_floor==xVal) && (yLocI_floor==yVal) )){
////                // Determine location of it in inverse image
////                int tempInverseX = self.x + locHere.at<float>(i,2);
////                int tempInverseY = self.y + locHere.at<float>(i,3);
//                NSLog(@"Does it match here: %f %f", locHere.at<float>(i,2),locHere.at<float>(i,3));
//            }
//        }
//
//                // if retina is within image
//                if((tempInverseX>self.retinaRadius)&&(tempInverseY>self.retinaRadius)&&(tempInverseX<mat.cols-self.retinaRadius) && (tempInverseY<mat.rows-self.retinaRadius) ){
//                    
//                    int tempGaze;
//                    // determine gaze direction
//                    if( kp.response < 0 ){
//                        tempGaze = 0;
//                    }
//                    else{
//                        tempGaze = 1;
//                    }
//                    
//                    if((tempGaze==self.gazePrev) ||(self.gazePrev==-1)){
//                        
//                        self.x = tempInverseX;
//                        self.y = tempInverseY;
//                        found=true;
//                        self.gazePrev =tempGaze;
//                        break;
//                    }
//                    
//                }
//                
//            }
//        }
//        if(found){break;}
//    }
    
    if(!found){
// random reset
//        NSLog(@"%@",@"Reseting retina position");
//        self.x  = (int) (self.retinaRadius+15 + arc4random() % (mat.cols-2*self.retinaRadius-15));
//        self.y  = (int) (self.retinaRadius+15 + arc4random() % (mat.rows-2*self.retinaRadius-15));
        if(self.gazePrev==1){
            self.gazePrev=0;
        }else if(self.gazePrev==0){
            self.gazePrev=1;
        }
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
    if (self.viewMode == 0){
        self.started=false;
        
    } else {
        int shape0 = mat.rows;
        int shape1 = mat.cols;
        
        
        // Corner Detection Here
        if(!self.started){
            self.x = (int) shape1/2;
            self.y = (int) shape0/2;
            self.started=true;
            double minVal0;
            double maxVal0;
            double minVal1;
            double maxVal1;
            cv::minMaxLoc(loc.col(0), &minVal0, &maxVal0);
            cv::minMaxLoc(loc.col(1), &minVal1, &maxVal1);
            
            self.retinaRadius = (int)(std::max(maxVal0,maxVal1));
            NSLog(@"Retina Radius is: %d", self.retinaRadius);
            
            self.gazePrev=-1;
        }
        
        
        
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2GRAY);
        
        cv::Mat V = [self retina_sample:self.x y:self.y mat:mat];
        
        if (self.viewMode == 1){
            cv::Mat cortImg =[self cort_img:V k_width:7 sigma:0.8];
            cortImg.convertTo(cortImg,CV_8U);
            mat = cortImg;
        }
        else if(self.viewMode==2){
            // Inverse Image
            [self gauss_norm_img:self.x y:self.y shape0:originalStillMat.rows shape1:originalStillMat.cols ];
            cv::Mat inverse = [self inverse:V x:self.x y:self.y shape0:shape0 shape1:shape1];
            
            
            
            inverse.convertTo(inverse,CV_8U);
            mat = inverse;
        }
        else if(self.viewMode==3){
            // creating cortical image
            int oldselfx = self.x;
            int oldselfy = self.y;
            
            cv::Mat cortImg =[self cort_img:V k_width:7 sigma:0.8];
            cortImg.convertTo(cortImg,CV_8U);
            
            [self create_new_focal_point:cortImg mat:mat];
            
            [self gauss_norm_img:oldselfx y:oldselfy shape0:originalStillMat.rows shape1:originalStillMat.cols ];
            cv::Mat inverse = [self inverse:V x:oldselfx y:oldselfy shape0:shape0 shape1:shape1];
            inverse.convertTo(inverse,CV_8U);
            
            // can convert to rgb inverse here
            cv::cvtColor(inverse, inverse, cv::COLOR_GRAY2RGB);
//            cv::cvtColor(cortImg, cortImg, cv::COLOR_GRAY2RGB);
            
            NSLog(@"%d %d",self.y,self.x);
            for(int i=self.y-3;i<self.y+3;i++){
                for(int j=self.x-3;j<self.x+3;j++){
                    cv::Vec3b keyVal = inverse.at<cv::Vec3b>(i,j);
                    keyVal[0]=255;
                    keyVal[1]=0;
                    keyVal[2]=0;
                    inverse.at<cv::Vec3b>(i,j) = keyVal;
                }
            }
            
            
            cv::Mat cortImgPadded;
            cv::copyMakeBorder(cortImg, cortImgPadded, (inverse.rows-cortImg.rows)/2+1, (inverse.rows-cortImg.rows)/2, 0, 0, cv::BORDER_CONSTANT,cv::Scalar(0));
        
            
            cv::hconcat(inverse, cortImgPadded, mat);
        }

    }
}

-(cv::Mat)inverse:(cv::Mat &)V x:(int)x y:(int)y shape0:(int)shape0 shape1:(int)shape1{
    cv::Mat vTemp;
    V.copyTo(vTemp);
    cv::Mat I1 = cv::Mat(shape0,shape1,CV_32F,0.0);
    cv::Mat I;
    
    int s = loc.rows;
    cv::Mat X = loc.col(0) + x;
    cv::Mat Y = loc.col(1) + y;
    
    for(int i=s-1;i>=0;i--){
        float loci6 = loc.at<float>(i,6);
        float yi = Y.at<float>(i);
        float xi = X.at<float>(i);
        int y1 = yi-loci6/2+0.5;
        if(y1<0){
            y1=0;
        } else if(y1>=shape0){
            y1=shape0-1;
        }
        int y2 = (yi+loci6/2+0.5);
        if(y2<0){
            y2=0;
        } else if(y2>=shape0){
            y2=shape0-1;
        }
        
        int x1 = (xi-loci6/2+0.5);
        if(x1<0){
            x1=0;
        } else if(x1>=shape1){
            x1=shape1-1;
        }
        
        int x2 = (xi+loci6/2+0.5);
        if(x2<0){
            x2=0;
        } else if(x2>=shape1){
            x2=shape1-1;
        }
        
        //        NSLog(@"%d %d %d %d", x1, x2, y1, y2);
        cv::Mat I1_roi = I1(cv::Rect(x1,y1,x2-x1,y2-y1));
        I1_roi.convertTo(I1_roi,CV_32F);
        cv::Mat multied;
        
        int shape0_coeff =coeff[i].rows;
        int shape1_coeff =coeff[i].cols;
        
        int begX = roundf(shape1_coeff/2.0-(x2-x1)/2.0);
        int endX = roundf(shape1_coeff/2.0+(x2-x1)/2.0);
        
        int begY = shape0_coeff/2.0-(y2-y1)/2.0;
        int endY = shape0_coeff/2.0+(y2-y1)/2.0;
        
        //        NSLog(@"%d %d %d %d", begX, endX, begY, endY);
        
        cv::Mat coeff_roi =coeff[i](cv::Rect(begX,begY,endX-begX,endY-begY));
        //        NSLog(@"%d %d %d %d", extract.rows, extract.cols, coeff[i].rows, coeff[i].cols);
        //        cv::multiply(extract,coeff[i],multied);
//        cv::multiply(extract,coeff_roi,multied);
//        NSLog(@"%d %d", coeff_roi.type(), I1_roi.type());
//        NSLog(@"%f",V.at<float>(i));
        
        if(V.at<float>(i)!=0.0){
            cv::Mat tempMat = V.at<float>(i) * coeff_roi;
            cv::Mat tempMat2;
            tempMat.convertTo(tempMat2, CV_32F);
            cv::add(I1_roi,tempMat2,I1_roi);
        }
//        NSLog(@"%@",@"GOTHERE");
        
    }
    cv::divide(I1, GI, I);
    
    
    return I;
}

-(cv::Mat)retina_sample:(int)x y:(int)y mat:(cv::Mat &)mat{
    cv::Mat copyMat;
    mat.copyTo(copyMat);
    int shape0 = copyMat.rows;
    int shape1 = copyMat.cols;
    
    int s = loc.rows;

    cv::Mat V = cv::Mat(1,s, CV_32F,0.0);
    cv::Mat X = loc.col(0)+x;
    cv::Mat Y = loc.col(1)+y;
    
    
    for(int i=0;i<s;i++){
        float loci6 = loc.at<float>(i,6);
        float yi = Y.at<float>(i);
        float xi = X.at<float>(i);
        int y1 = yi-loci6/2+0.5;
        if(y1<0){
            y1=0;
        } else if(y1>=shape0){
            y1=shape0-1;
        }
        int y2 = (yi+loci6/2+0.5);
        if(y2<0){
            y2=0;
        } else if(y2>=shape0){
            y2=shape0-1;
        }
        
        int x1 = (xi-loci6/2+0.5);
        if(x1<0){
            x1=0;
        } else if(x1>=shape1){
            x1=shape1-1;
        }
        
        int x2 = (xi+loci6/2+0.5);
        if(x2<0){
            x2=0;
        } else if(x2>=shape1){
            x2=shape1-1;
        }
        
//        NSLog(@"%d %d %d %d", x1, x2, y1, y2);
        cv::Mat extract = copyMat(cv::Rect(x1,y1,x2-x1,y2-y1));
        extract.convertTo(extract, CV_32F);
        
        cv::Mat multied;
        
        int shape0_coeff =coeff[i].rows;
        int shape1_coeff =coeff[i].cols;
        
        int begX = roundf(shape1_coeff/2.0-(x2-x1)/2.0);
        int endX = roundf(shape1_coeff/2.0+(x2-x1)/2.0);
        
        int begY = shape0_coeff/2.0-(y2-y1)/2.0;
        int endY = shape0_coeff/2.0+(y2-y1)/2.0;
        
//        NSLog(@"%d %d %d %d", begX, endX, begY, endY);
        
        cv::Mat coeff_roi =coeff[i](cv::Rect(begX,begY,endX-begX,endY-begY));
//        NSLog(@"%d %d %d %d", extract.rows, extract.cols, coeff[i].rows, coeff[i].cols);
//        cv::multiply(extract,coeff[i],multied);
        cv::multiply(extract,coeff_roi,multied);
//        NSLog(@"%f", cv::sum(multied)[0]);
        V.at<float>(i) = cv::sum(multied)[0];

    }
    return V;
}

-(cv::Mat)cort_img:(cv::Mat &)V k_width:(int)k_width sigma:(float)sigma{
    cv::Mat cortImg;
    int shape0 = [self.cort_size[0] intValue];
    int shape1 = [self.cort_size[1] intValue];

    // Project the cortical images
    cv::Mat L_img  = cv::Mat(shape0,shape1, CV_32F, 0.0);
    cv::Mat R_img  = cv::Mat(shape0,shape1, CV_32F, 0.0);
    cv::Mat L_gimg = cv::Mat(shape0,shape1, CV_32F, 0.0);
    cv::Mat R_gimg = cv::Mat(shape0,shape1, CV_32F, 0.0);

 
    // L
    for(int p=0;p<(int)(L_loc.rows);p++){
        float p0 =L_loc.at<float>(p,0);
        float p1 =L_loc.at<float>(p,1);
        float p2 =L.at<float>(p,2);
        int x = (int)(roundf(p0));
        int y = (int)(roundf(p1));
        
        // coords of kernel in img array
        int y1=0;
        int y2=0;
        int x1=0;
        int x2=0;
        
        if ((y - k_width/2)>0){
            y1 = y - k_width/2;
        }
        if ((y + k_width/2 + 1)>shape0){
            y2 = shape0;
        } else{
            y2 =y + k_width/2 + 1;
        }
        
        if ((x - k_width/2) > 0){
            x1 = x - k_width/2;
        }
        if ((x + k_width/2 + 1)>shape1){
            x2 = shape1;
        } else{
            x2 = x + k_width/2 + 1;
        }
        
        // coords into the 10x10 gaussian filters array (used floor instead)
        
        int dx = (int)(roundf(10*((roundf(p0*10)/10) - floor(p0)) ));
        if(dx==10){
            dx=0;
        }
        int dy = (int)(roundf(10*((roundf(p1*10)/10) - floor(p1)) ));
        if(dy==10){
            dy=0;
        }
        
        
        
        cv::Mat g = G[dx][dy](cv::Rect(0,0,x2-x1,y2-y1));
        
//        NSLog(@"%d %d %d %d", x1,x2,y1,y2);
        
        cv::Mat L_img_roi = L_img(cv::Rect(x1,y1,x2-x1,y2-y1));
        cv::Mat L_gimg_roi = L_gimg(cv::Rect(x1,y1,x2-x1,y2-y1));
//        cv::Mat gV = (g*V.at<float>((int)(p2)));
//        NSLog(@"%d %d %d %d",gV.rows, gV.cols, L_img_roi.rows, L_img_roi.cols);
        
        cv::add(L_img_roi,g*V.at<float>((int)(p2)),L_img_roi);
        cv::add(L_gimg_roi,g,L_gimg_roi);
//
    }
    
    cv::Mat left;// = cv::Mat(L_img.rows,L_img.cols,CV_8U);
    cv::divide(L_img, L_gimg, left);

//    print(left);
//
//    
    // R
    for(int p=0;p<(int)(R_loc.rows);p++){
        float p0 =R_loc.at<float>(p,0);
        float p1 =R_loc.at<float>(p,1);
        float p2 =R.at<float>(p,2);
        int x = (int)(roundf(p0));
        int y = (int)(roundf(p1));
        
        // coords of kernel in img array
        int y1=0;
        int y2=0;
        int x1=0;
        int x2=0;
        
        if ((y - k_width/2)>0){
            y1 = y - k_width/2;
        }
        if ((y + k_width/2 + 1)>shape0){
            y2 = shape0;
        } else{
            y2 =y + k_width/2 + 1;
        }
        
        if ((x - k_width/2) > 0){
            x1 = x - k_width/2;
        }
        if ((x + k_width/2 + 1)>shape1){
            x2 = shape1;
        } else{
            x2 = x + k_width/2 + 1;
        }
        
        // coords into the 10x10 gaussian filters array (used floor instead)
        
        int dx = (int)(roundf(10*((roundf(p0*10)/10) - floor(p0)) ));
        if(dx==10){
            dx=0;
        }
        int dy = (int)(roundf(10*((roundf(p1*10)/10) - floor(p1)) ));
        if(dy==10){
            dy=0;
        }
        
        
        cv::Mat g = G[dx][dy](cv::Rect(0,0,x2-x1,y2-y1));
        
        //        NSLog(@"%d %d %d %d", x1,x2,y1,y2);
        
        cv::Mat R_img_roi = R_img(cv::Rect(x1,y1,x2-x1,y2-y1));
        cv::Mat R_gimg_roi = R_gimg(cv::Rect(x1,y1,x2-x1,y2-y1));
        //        cv::Mat gV = (g*V.at<float>((int)(p2)));
        //        NSLog(@"%d %d %d %d",gV.rows, gV.cols, L_img_roi.rows, L_img_roi.cols);
        
        cv::add(R_img_roi,g*V.at<float>((int)(p2)),R_img_roi);
        cv::add(R_gimg_roi,g,R_gimg_roi);
        //
    }
    
    cv::Mat right;// = cv::Mat(L_img.rows,L_img.cols,CV_8U);
    cv::divide(R_img, R_gimg, right);
    
//    print(right);
    
    
//    NSLog(@"%@",@"Completed cortical image");
    cv::rotate(left, left, 2);
    cv::rotate(right, right, 0);
    cv::hconcat(left,right, cortImg);
//    NSLog(@"check check %d %d %d %d %d %d",left.rows, left.cols,right.rows, right.cols,cortImg.rows, cortImg.cols);
    return cortImg;
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
