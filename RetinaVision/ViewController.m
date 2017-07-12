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
    cv::Mat G[10][10];
    cv::Mat GI;
}

@property IBOutlet UIImageView *imageView;
@property IBOutlet UIActivityIndicatorView *activityIndicatorView;
@property IBOutlet UIToolbar *toolbar;

@property VideoCamera *videoCamera;
@property BOOL saveNextFrame;
@property int viewMode;
@property NSMutableArray* loc;
@property NSMutableArray* coeff;
@property NSMutableArray* cort_size;
@property BOOL rotated;
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
- (void)prep;
- (void)cort_prepare:(double)shrink k_width:(int)k_width sigma:(double)sigma;
- (void)gauss100:(int)width sigma:(double)sigma;
- (void)gauss_norm_img:(int)x y:(int)y shape0:(int)shape0 shape1:(int)shape1;
- (cv::Mat)retina_sample:(int)x y:(int)y mat:(cv::Mat &)mat;
- (cv::Mat)cort_img:(cv::Mat &)V k_width:(int)k_width sigma:(float)sigma;

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
    
    
    [self prep];
    
}

-(void)prep{
    [self LRsplit];
    [self cort_map:15];
    [self cort_prepare:0.5 k_width:7 sigma:0.8];
    self.rotated=true;
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
    L = cv::Mat((int)[left count],3, CV_32F);
    for(int i=0;i<[left count];i++){
        for(int j=0;j<3;j++){
            L.at<float>(i,j)=[left[i][j] floatValue];
        }
    }
//    print(L);
    
    R = cv::Mat((int)[right count],3, CV_32F);
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
    
    NSLog(@"%@",@"Here comes the wait 1");
    
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
    
    NSLog(@"%@",@"Here comes the wait 2");
    
    //  Calculating L_theta
    cv::Mat L_theta;

    cv::phase(L.col(0)-alpha, L.col(1), L_theta);
    L_theta = L_theta - M_PI;
//    print(L_theta);

    //  Calculating R_theta
    cv::Mat R_theta;
    
    cv::phase(R.col(0)+alpha, R.col(1), R_theta);
    
    NSLog(@"%@",@"Here comes the wait 3");
    
    for(int i =0;i<R_theta.rows;i++){
        while(R_theta.at<float>(i) > M_PI/2){
            R_theta.at<float>(i)=R_theta.at<float>(i)-M_PI;
//        float value = R_theta.at<float>(i);
//        float modvalue = fmod(value,M_PI/2);
//        R_theta.at<float>(i)=modvalue-M_PI;
        }
    }
    
    NSLog(@"%@",@"Here comes the wait 4");
    
    cv::hconcat(L_theta, L_r, L_loc);
    cv::hconcat(R_theta, R_r, R_loc);
    
    NSLog(@"%@",@"Here comes the wait 5");
//    print(L_loc);
    
    // Calculating x (theta)
    cv::Mat L_theta2;
    L_loc.copyTo(L_theta2);
    cv::Mat R_theta2;
    R_loc.copyTo(R_theta2);
    
    L_theta2.col(1)=0;
    R_theta2.col(1)=0;
    
    NSLog(@"%@",@"Here comes the wait 6");
    
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
    double result = 0;
    for(int i=0;i<mat.rows;i++){
        for(int j=i+1;j<mat.rows;j++){
                result = cv::norm(mat.row(i), mat.row(j));
                sum+=result;
        }
    }
    double mean = (sum*2) / (mat.rows*mat.rows);
    return mean;
}


-(void)cort_prepare:(double)shrink k_width:(int)k_width sigma:(double)sigma{
    [self gauss100:k_width sigma:sigma];
    
    // bring min(x) to 0
    double L_locMin_0;
    cv::minMaxLoc(L_loc.col(0),  &L_locMin_0);
    

    L_loc.col(0) -=L_locMin_0;
    
    double R_locMin_0;
    cv::minMaxLoc(R_loc.col(0),  &R_locMin_0);
    R_loc.col(0) -= R_locMin_0;
    
    
    // flip y and bring min(y) to 0
    L_loc.col(1) = -L_loc.col(1);
    R_loc.col(1) = -R_loc.col(1);
    
    double L_locMin_1;
    cv::minMaxLoc(L_loc.col(1),  &L_locMin_1);
    L_loc.col(1) -= L_locMin_1;
    double R_locMin_1;
    cv::minMaxLoc(R_loc.col(1),  &R_locMin_1);
    R_loc.col(1) -= R_locMin_1;

    
    // k_width more pixels of space from all sides for kernels to fit
    L_loc += k_width;
    R_loc += k_width;
    
    double L_locMax_0;
    cv::minMaxLoc(L_loc.col(0),  NULL, &L_locMax_0);
    double L_locMax_1;
    cv::minMaxLoc(L_loc.col(1),  NULL, &L_locMax_1);
    double cort_y = L_locMax_1 + k_width;
    double cort_x = L_locMax_0 + k_width;
    
    // shrinking
    self.cort_size = [[NSMutableArray alloc]init];
    int y_shrinked = (int) (cort_y*shrink);
    [self.cort_size addObject:[NSNumber numberWithInt:y_shrinked]];
    int x_shrinked = (int) (cort_x*shrink);
    [self.cort_size addObject:[NSNumber numberWithInt:x_shrinked]];
    
    L_loc*=shrink;
    R_loc*=shrink;
    
    NSLog(@"Cort size: %@",self.cort_size);
    
}

-(void)gauss100:(int)width sigma:(double)sigma{
 
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            G[i][j]= cv::Mat(width,width, CV_32F,0.0);

            double dx = width/2 + i*0.1;
            double dy = width/2 + j*0.1;
            
            for(int x=0;x<width;x++){
                for(int y=0;y<width;y++){
                    cv::Mat xy = cv::Mat(1,2, CV_32F);
                    xy.at<float>(0,0)=dx-x;
                    xy.at<float>(0,1)=dy-y;
                    
                    double d = cv::norm(xy);
                    G[i][j].at<float>(y,x)= exp(-pow(d,2)/(2*pow(sigma,2)))/sqrt(2*M_PI*pow(sigma,2));
//                    NSLog(@"%f",G[i][j].at<float>(y,x));
                }
            }
        }
    }
    
}

-(void)viewDidLayoutSubviews{
    [super viewDidLayoutSubviews];
    
    switch ([UIDevice currentDevice].orientation){
        case UIDeviceOrientationPortraitUpsideDown:
            self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortraitUpsideDown;
            self.rotated=true;
            break;
        case UIDeviceOrientationLandscapeLeft:
            self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeLeft;
            self.rotated=true;
            break;
        case UIDeviceOrientationLandscapeRight:
            self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeRight;
            self.rotated=true;
            break;
        default:
            self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
            self.rotated=true;
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
        int shape0 = updatedStillMatRetina.rows;
        int shape1 = updatedStillMatRetina.cols;

//        NSLog(@"0:%d 1:%d",shape0,shape1);
        
        
        // processimage to update it.
        [self processImage:updatedStillMatRetina];
        image = MatToUIImage(updatedStillMatRetina);
        
        self.imageView.image = image;
    }
}


-(void)gauss_norm_img:(int)x y:(int)y shape0:(int)shape0 shape1:(int)shape1{
//    shape0=720;
//    shape1=1280;
//    x =1280/2;
//    y =720/2;
    
    GI = cv::Mat(shape0,shape1, CV_32F,0.0);
    //    print(GI);
    
    int s = (int)[self.loc count];
    
    for(int i=s-1;i>=0;i--){
        float valuei6 = [self.loc[i][6] floatValue];
        float valueXi0 = [self.loc[i][0] floatValue]+x;
        float valueYi1 = [self.loc[i][1] floatValue]+y;
        
        int y1 = valueYi1-valuei6/2+0.5;
        int y2 = valueYi1+valuei6/2+0.5;
        int x1 = valueXi0-valuei6/2+0.5;
        int x2 = valueXi0+valuei6/2+0.5;
        //        NSLog(@"%d %d %d %d",y1,x1,y2-y1,x2-x1);
        //        NSLog(@"%d %d",GI.rows,GI.cols);
        
        cv::Mat imageRegion;
        imageRegion = GI(cv::Rect(x1,y1,x2-x1,y2-y1));
//        print(imageRegion);
        
        int coeffX=0;
        for(int x=x1;x<x2;x++){
            int coeffY=0;
            for(int y=y1;y<y2;y++){
                //                NSLog(@"%@",self.coeff[i][coeffY][coeffX]);
                GI.at<float>(y,x)+=[self.coeff[i][coeffY][coeffX] floatValue];
                coeffY++;
            }
            coeffX++;
        }
    }
    NSLog(@"%@",@"Completed Gaussian Normal");
    //    print(GI);
    
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

    // Implementing inverse gaussian normal of image
    if(self.rotated){
        int shape0 = mat.rows;
        int shape1 = mat.cols;
        
        int x = (int) shape1/2;
        int y = (int) shape0/2;
        
        [self gauss_norm_img:x y:y shape0:shape0 shape1:shape1];
        
        self.rotated=false;
        
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
        int shape0 = mat.rows;
        int shape1 = mat.cols;
        
        int x = (int) shape1/2;
        int y = (int) shape0/2;
        
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2GRAY);
        
        cv::Mat V = [self retina_sample:x y:y mat:mat];
        cv::Mat cortImg =[self cort_img:V k_width:7 sigma:0.8];
//        print(V);
        NSLog(@"%@",@"Completed retina sampling");
    } else {
        
    }
}

-(cv::Mat)retina_sample:(int)x y:(int)y mat:(cv::Mat &)mat{
    cv::Mat copyMat;
    mat.copyTo(copyMat);
    
    int s = (int)[self.loc count];
    
    cv::Mat V = cv::Mat(1,s, CV_32F,0.0);
    
    for(int i=0;i<s;i++){
        float valuei6 = [self.loc[i][6] floatValue];
        float valueXi0 = [self.loc[i][0] floatValue]+x;
        float valueYi1 = [self.loc[i][1] floatValue]+y;
        
        int y1 = valueYi1-valuei6/2+0.5;
        int y2 = valueYi1+valuei6/2+0.5;
        int x1 = valueXi0-valuei6/2+0.5;
        int x2 = valueXi0+valuei6/2+0.5;

        int coeffX=0;
        float sum=0;
        for(int x=x1;x<x2;x++){
            int coeffY=0;
            for(int y=y1;y<y2;y++){
                sum+=( copyMat.at<unsigned char>(y,x) *  [self.coeff[i][coeffY][coeffX] floatValue] );
                coeffY++;
            }
            coeffX++;
        }
        V.at<float>(0,i)=sum;
        
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
//    
//    for(int i=0;i<10;i++){
//        for(int j=0;j<10;j++){
//            print(G[i][j]);
//        }
//    }
//    print(G);
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
        y2 = y + k_width/2 + 1;
        
        if ((x - k_width/2) > 0){
            x1 = x - k_width/2;
        }
        x2 = x + k_width/2 + 1;
        
        // coords into the 10x10 gaussian filters array (used floor instead)
//        NSLog(@"%f %f %f %f", (roundf(p0*10)/10), roundf(p0), 10*((roundf(p0*10)/10) - floor(p0)), roundf(10*((roundf(p0*10)/10) - floor(p0)) ) );
//        NSLog(@"%d", (int)(10*( (roundf(p0*10)/10) - roundf(p0) )) );
        
        int dx = (int)(roundf(10*((roundf(p0*10)/10) - floor(p0)) ));
        if(dx==10){
            dx=0;
        }
        int dy = (int)(roundf(10*((roundf(p1*10)/10) - floor(p1)) ));
        if(dy==10){
            dy=0;
        }
//        dx = int(10*(np.round(L_loc[p,0], decimals=1) - round(L_loc[p,0])))
//        dy = int(10*(np.round(L_loc[p,1], decimals=1) - round(L_loc[p,1])))
        
        
        // in case of big kernels, clipping kernels at img edges
        int gy1=0;
        int gy2=k_width;
        int gx1=0;
        int gx2=k_width;
        
        if ((y - k_width/2) < 0){
            gy1 = -(y - k_width/2);
        }
        if (y2 > shape0){
            gy2 = k_width-(y2-shape0);
        }
        if ((x - k_width/2) < 0){
            gx1=-(x - k_width/2);
        }
        if (x2 > shape1){
            gx2=k_width-(x2-shape1);
        }
//        NSLog(@"%d %d %d %d",gy1,gy2,gx1,gx2);
        NSLog(@"%d %d %d %d",gy1,gy2,gx1,gx2);
        
        
//        int coeffX=0;
        for(int x=gx1;x<gx2;x++){
//            int coeffY=0;
            for(int y=gy1;y<gy2;y++){
//                NSLog(@"%f %f", G[dx][dy].at<float>(y,x),V.at<float>((int)p2));
                L_img.at<float>(y,x)+= G[dx][dy].at<float>(y,x)* V.at<float>((int)p2);
                L_gimg.at<float>(y,x)+=G[dx][dy].at<float>(y,x);
//                coeffY++;
            }
//            coeffX++;
        }

    }
//    print(L_img);
    
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
