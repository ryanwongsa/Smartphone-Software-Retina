//
//  VideoCamera.h
//  RetinaVision
//
//  Created by Ryan Wong on 2017/07/05.
//  Copyright © 2017 Ryan Wong. All rights reserved.
//

#import <opencv2/videoio/cap_ios.h>

@interface VideoCamera : CvVideoCamera

@property BOOL letterboxPreview;

-(void)setPointOfInterestInParentViewSpace:(CGPoint)point;

@end
