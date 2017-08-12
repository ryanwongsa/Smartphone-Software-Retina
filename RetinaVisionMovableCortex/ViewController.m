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
#import <opencv2/calib3d/calib3d.hpp>

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
    cv::Mat mask;
}

@property IBOutlet UIImageView *imageView;
@property IBOutlet UIActivityIndicatorView *activityIndicatorView;
@property IBOutlet UIToolbar *toolbar;

@property VideoCamera *videoCamera;
@property BOOL saveNextFrame;
@property int viewMode;
@property NSMutableArray* cort_size;
@property BOOL started;
@property int x;
@property int y;
@property int retinaRadius;
@property int gazePrev;
@property float xd;
@property float yd;
@property float l_min0;
@property float l_min1;
@property float r_min0;
@property float r_min1;
@property BOOL record;
@property NSString* fileContents;

- (IBAction)onTapToSetPointOfInterest:(UITapGestureRecognizer *)tapGesture;
- (IBAction)onColorModeSelected:(UISegmentedControl *)segmentedControl;
- (IBAction)onSwitchCameraButtonPressed;
- (IBAction)onSaveButtonPressed;

- (void)refresh;
- (void)processImage:(cv::Mat &)mat;
- (void)processImageHelper:(cv::Mat &)mat;
- (void)saveImage:(UIImage *)image;
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
    
    //    Reading loc file
    NSString* filePath = @"locFile";
    NSString* fileRoot = [[NSBundle mainBundle] pathForResource:filePath ofType:@"txt"];
    
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

    NSString* filePath12 = @"mask_file";
    NSString* fileRoot12 = [[NSBundle mainBundle] pathForResource:filePath12 ofType:@"txt"];
    NSString* fileContents12 = [NSString stringWithContentsOfFile:fileRoot12 encoding:NSUTF8StringEncoding error:nil];
    
    NSArray* allLinedStrings12 = [fileContents12 componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];

    NSString* strsInOneLine12 = [allLinedStrings12 objectAtIndex:0];
    NSArray* singleStr12= [strsInOneLine12 componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@" "]];
    shape0 = [singleStr12[0] intValue];
    shape1 = [singleStr12[1] intValue];
    mask = cv::Mat(shape0,shape1,CV_8U);
    for(int i=1;i< (int)([allLinedStrings12 count]);i++){
        NSString* elements = allLinedStrings12[i];
        
        NSArray* parts = [elements componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"#"]];
        for(int j=0;j< shape1;j++){
            mask.at<unsigned char>(i-1,j) = (unsigned char)[parts[j] intValue];
        }
    }
    
    self.started=false;
    
    self.fileContents=@"";
    
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
            self.record=true;
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
        
        cv::Mat GI_roi = GI(cv::Rect(x1,y1,x2-x1,y2-y1));
        
        
        int shape0_coeff =coeff[i].rows;
        int shape1_coeff =coeff[i].cols;
        
        int begX = roundf(shape1_coeff/2.0-(x2-x1)/2.0);
        int endX = roundf(shape1_coeff/2.0+(x2-x1)/2.0);
        
        int begY = shape0_coeff/2.0-(y2-y1)/2.0;
        int endY = shape0_coeff/2.0+(y2-y1)/2.0;
        
        
        cv::Mat coeff_roi =coeff[i](cv::Rect(begX,begY,endX-begX,endY-begY));
        
        cv::add(GI_roi,coeff_roi,GI_roi);
    }
}

-(void)create_new_focal_point:(cv::Mat &)cortImg mat:(cv::Mat &)mat{
    
    std::vector<cv::KeyPoint> keypointsAll;
//    cv::FAST(cortImg, keypointsAll, 20);
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(50);
    detector->detect(cortImg, keypointsAll, mask);
//    sift.operator
    
    
    for(cv::KeyPoint kp : keypointsAll){
        
        cv::Vec3b keyVal = cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x);
        keyVal[0]=0;
        keyVal[1]=0;
        keyVal[2]=255;
        cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x) = keyVal;
        
    }
    
    // 1= first quad, 2 = second quad, 3 = third quad, 4 = forth quad
    for(auto &kp : keypointsAll){
        // disable the effect of responses greater than 1 affecting the stuff
        float dx = std::abs(kp.pt.x - cortImg.cols/2);
        dx = dx/(cortImg.cols/2.0);
        kp.response = kp.response*dx;
        
        
        // FIRST QUADRANT
        int quadrant = 0;
        if((kp.pt.x <= cortImg.cols/2) && (kp.pt.y >= cortImg.rows/2)){
            quadrant=3;
        }
        // SECOND QUADRANT
        else if((kp.pt.x <= cortImg.cols/2) && (kp.pt.y < cortImg.rows/2)){
            quadrant=2;
        }
        else{
            kp.pt.x = kp.pt.x - cortImg.cols/2;
            // THIRD QUADRANT
            if(kp.pt.y >= cortImg.rows/2){
                quadrant=4;
            }
            // FORTH QUADRANT
            else{
                quadrant=1;
            }
            
        }
        kp.class_id = quadrant;
        
    }
    
    
    std::sort(keypointsAll.begin(), keypointsAll.end(), [](cv::KeyPoint a, cv::KeyPoint b) { return a.response > b.response;});
    
    BOOL found=false;
    
    // using probability to switch gaze direction
    
    // Choose a random start
    if(self.gazePrev==-1){
        self.gazePrev = ( arc4random() % 4 ) +1;
    }
    else
    {
        int prob = arc4random() % 100;
        if(prob <= 50){
        } else if(prob <= 70){
            self.gazePrev = (self.gazePrev+1)%4;
        } else if(prob <= 90){
            self.gazePrev = (self.gazePrev-1)%4;
        } else{
            self.gazePrev = (self.gazePrev+2)%4;
        }
    
        if(self.gazePrev==0){
            self.gazePrev=4;
        }
    }
    
    
    // Could remove this part, just for visualisation
    for(cv::KeyPoint kp : keypointsAll){
        if((kp.class_id ==2)||(kp.class_id==3)){
            cv::Vec3b keyVal = cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x);
            keyVal[0]=255;
            keyVal[1]=0;
            keyVal[2]=0;
            cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x) = keyVal;
        }
        else{
            cv::Vec3b keyVal = cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x+cortImg.cols/2.0);
            keyVal[0]=255;
            keyVal[1]=0;
            keyVal[2]=0;
            cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x+cortImg.cols/2.0) = keyVal;
        }

    }
    
    
    for(cv::KeyPoint kp : keypointsAll){
        // skip this one if it doesnt match gaze point
        if(self.gazePrev!=kp.class_id){
            continue;
        }
        
        if((kp.class_id ==2)||(kp.class_id==3)){
            cv::Vec3b keyVal = cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x);
            keyVal[0]=0;
            keyVal[1]=0;
            keyVal[2]=255;
            cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x) = keyVal;
        }
        else{
            cv::Vec3b keyVal = cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x+cortImg.cols/2.0);
            keyVal[0]=0;
            keyVal[1]=0;
            keyVal[2]=255;
            cortImg.at<cv::Vec3b>(kp.pt.y,kp.pt.x+cortImg.cols/2.0) = keyVal;
        }
        float yPoint = kp.pt.y;
        float xPoint = kp.pt.x;
        
        kp.pt.y = -kp.pt.y;

        cv::Point2f center((cortImg.cols/2.0-1)/2.0, -(cortImg.rows-1)/2.0);
        cv::Point2f trans_pt = kp.pt - center;
        
        float xTemp;
        float yTemp;
        if((kp.class_id ==2)||(kp.class_id==3)){
            xTemp = trans_pt.y;
            yTemp = -trans_pt.x;
        }
        else{
            xTemp = -trans_pt.y;
            yTemp = trans_pt.x;
        }
        cv::Point2f rot_p(xTemp, yTemp);
        cv::Point2f fin_pt = rot_p + center;
        fin_pt.y=-fin_pt.y;
    
        // scaling factor fixing
        fin_pt = fin_pt*2.0;
        // k_width fixing
        fin_pt.x = fin_pt.x - 7;
        fin_pt.y = fin_pt.y - 7;
    
        if((kp.class_id ==2)||(kp.class_id==3)){
            fin_pt.y = self.l_min1 + fin_pt.y;
            fin_pt.x = self.l_min0 + fin_pt.x;
        }
        else{
            fin_pt.y = self.r_min1 + fin_pt.y;
            fin_pt.x = self.r_min0 + fin_pt.x;
        }
    
        fin_pt.y = - fin_pt.y;
        fin_pt.x = fin_pt.x * self.xd / self.yd;
        
        float theta=0;
        float r = 0;
        float finalX=0;
        float finalY=0;
        
        if((kp.class_id ==2)||(kp.class_id==3)){
             
            if(fin_pt.x>0){
                theta= fin_pt.x+M_PI;
            }
            else{
                theta= fin_pt.x-M_PI;
            }
            r =fin_pt.y;
        
            float tan_theta = std::tan(theta);
        
            finalX = -std::sqrt((r*r)/(1+tan_theta*tan_theta));
            finalY = tan_theta*(finalX);//-15);
            finalX = finalX+15;
        
        }
        else{
            theta = fin_pt.x;
            r =fin_pt.y;
        
        
            float tan_theta = std::tan(theta);
            finalX = std::sqrt((r*r)/(1+tan_theta*tan_theta));//-15;
            finalY = tan_theta*(finalX);
            finalX = finalX-15;
        
            
        }
        int tempInverseX = self.x + finalX;
        int tempInverseY = self.y + finalY;
        
        if((tempInverseX>self.retinaRadius)&&(tempInverseY>self.retinaRadius)&&(tempInverseX<mat.cols-self.retinaRadius) && (tempInverseY<mat.rows-self.retinaRadius) ){
            int tempGaze = kp.class_id;

            if((tempGaze==self.gazePrev) ||(self.gazePrev==-1)){
                self.x = tempInverseX;
                self.y = tempInverseY;
                found=true;
                self.gazePrev =tempGaze;
                if((kp.class_id ==2)||(kp.class_id==3)){
                    cv::Vec3b keyVal = cortImg.at<cv::Vec3b>(yPoint,xPoint);
                    keyVal[0]=0;
                    keyVal[1]=255;
                    keyVal[2]=0;
                    cortImg.at<cv::Vec3b>(yPoint,xPoint) = keyVal;
                }
                else{
                    cv::Vec3b keyVal = cortImg.at<cv::Vec3b>(yPoint,xPoint+cortImg.cols/2.0);
                    keyVal[0]=0;
                    keyVal[1]=255;
                    keyVal[2]=0;
                    cortImg.at<cv::Vec3b>(yPoint,xPoint+cortImg.cols/2.0) = keyVal;
                }
                break;
            }
        }
    }
    
    if(!found){
        self.gazePrev = (self.gazePrev+2)%4;
        if(self.gazePrev==0){
            self.gazePrev=4;
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
            
            if(self.record){
                std::stringstream origImg;
                origImg << mat;
                NSString *stringImgVersion = [NSString stringWithCString:origImg.str().c_str() encoding:NSASCIIStringEncoding];
                
                self.fileContents = [NSString stringWithFormat:@"%@%@\n", self.fileContents, stringImgVersion];
                
                std::stringstream ss;
                ss << V;
                NSString *stringVersion = [NSString stringWithCString:ss.str().c_str() encoding:NSASCIIStringEncoding];
                
//                [self.fileContents appendString:stringVersion];
                self.fileContents = [NSString stringWithFormat:@"%@%d %d\n", self.fileContents,
                                     self.x, self.y ];
                self.fileContents = [NSString stringWithFormat:@"%@%@\n", self.fileContents,
                                      stringVersion];
                NSLog(@"Contents now: %lu",(unsigned long)self.fileContents.length);
            }
            
            
            
            // creating cortical image
            int oldselfx = self.x;
            int oldselfy = self.y;
            
            cv::Mat cortImg =[self cort_img:V k_width:7 sigma:0.8];
            cortImg.convertTo(cortImg,CV_8U);
            cv::cvtColor(cortImg, cortImg, cv::COLOR_GRAY2RGB);
            [self create_new_focal_point:cortImg mat:mat];
            
            [self gauss_norm_img:oldselfx y:oldselfy shape0:originalStillMat.rows shape1:originalStillMat.cols ];
            cv::Mat inverse = [self inverse:V x:oldselfx y:oldselfy shape0:shape0 shape1:shape1];
            inverse.convertTo(inverse,CV_8U);
            
            // can convert to rgb inverse here
            cv::cvtColor(inverse, inverse, cv::COLOR_GRAY2RGB);

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
//            mat = cortImg;
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
        
        cv::Mat I1_roi = I1(cv::Rect(x1,y1,x2-x1,y2-y1));
        I1_roi.convertTo(I1_roi,CV_32F);
        cv::Mat multied;
        
        int shape0_coeff =coeff[i].rows;
        int shape1_coeff =coeff[i].cols;
        
        int begX = roundf(shape1_coeff/2.0-(x2-x1)/2.0);
        int endX = roundf(shape1_coeff/2.0+(x2-x1)/2.0);
        
        int begY = shape0_coeff/2.0-(y2-y1)/2.0;
        int endY = shape0_coeff/2.0+(y2-y1)/2.0;
        
        cv::Mat coeff_roi =coeff[i](cv::Rect(begX,begY,endX-begX,endY-begY));
       
        
        if(V.at<float>(i)!=0.0){
            cv::Mat tempMat = V.at<float>(i) * coeff_roi;
            cv::Mat tempMat2;
            tempMat.convertTo(tempMat2, CV_32F);
            cv::add(I1_roi,tempMat2,I1_roi);
        }
        
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
        
        cv::Mat extract = copyMat(cv::Rect(x1,y1,x2-x1,y2-y1));
        extract.convertTo(extract, CV_32F);
        
        cv::Mat multied;
        
        int shape0_coeff =coeff[i].rows;
        int shape1_coeff =coeff[i].cols;
        
        int begX = roundf(shape1_coeff/2.0-(x2-x1)/2.0);
        int endX = roundf(shape1_coeff/2.0+(x2-x1)/2.0);
        
        int begY = shape0_coeff/2.0-(y2-y1)/2.0;
        int endY = shape0_coeff/2.0+(y2-y1)/2.0;
        
        
        cv::Mat coeff_roi =coeff[i](cv::Rect(begX,begY,endX-begX,endY-begY));
        cv::multiply(extract,coeff_roi,multied);
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
        
        
        cv::Mat L_img_roi = L_img(cv::Rect(x1,y1,x2-x1,y2-y1));
        cv::Mat L_gimg_roi = L_gimg(cv::Rect(x1,y1,x2-x1,y2-y1));
        
        cv::add(L_img_roi,g*V.at<float>((int)(p2)),L_img_roi);
        cv::add(L_gimg_roi,g,L_gimg_roi);

    }
    
    cv::Mat left;
    cv::divide(L_img, L_gimg, left);
 
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
        
        cv::Mat R_img_roi = R_img(cv::Rect(x1,y1,x2-x1,y2-y1));
        cv::Mat R_gimg_roi = R_gimg(cv::Rect(x1,y1,x2-x1,y2-y1));
        
        cv::add(R_img_roi,g*V.at<float>((int)(p2)),R_img_roi);
        cv::add(R_gimg_roi,g,R_gimg_roi);
    }
    cv::Mat right;
    cv::divide(R_img, R_gimg, right);
    
    cv::rotate(left, left, 2);
    cv::rotate(right, right, 0);
    cv::hconcat(left,right, cortImg);
    return cortImg;
}

-(IBAction)onSaveButtonPressed{
    self.record = false;
    NSArray *paths = NSSearchPathForDirectoriesInDomains
    (NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    NSLog(@"HERE IS THE DIRECTORY: %@", documentsDirectory);

    NSString *fileName = [NSString stringWithFormat:@"%@/data.txt", documentsDirectory];
//    NSLog(@"Contents %@",self.fileContents);
    NSLog(@"Contents SIZE: %lu",(unsigned long)self.fileContents.length);
    NSString *content = self.fileContents;
    [content writeToFile:fileName atomically:NO encoding:NSStringEncodingConversionAllowLossy error:nil];
    self.fileContents = @"";
    
}

-(void)saveImage:(UIImage *)image{
    
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}





@end
