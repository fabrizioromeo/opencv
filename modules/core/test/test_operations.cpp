/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>

using namespace cv;
using namespace std;
using namespace cvtest::ocl;

#if defined _MSC_VER && _MSC_VER < 1400
#define MSVC_OLD 1
#else
#define MSVC_OLD 0
#endif

TEST(Core_Array, expressions_Mat)
{
    Mat one_3x1(3, 1, CV_32F, Scalar(1.0));
    Mat shi_3x1(3, 1, CV_32F, Scalar(1.2));
    Mat shi_2x1(2, 1, CV_32F, Scalar(-1));
    Scalar shift = Scalar::all(15);

    float data[] = { sqrt(2.f)/2, -sqrt(2.f)/2, 1.f, sqrt(2.f)/2, sqrt(2.f)/2, 10.f };
    Mat rot_2x3(2, 3, CV_32F, data);

    Mat res = one_3x1 + shi_3x1 + shi_3x1 + shi_3x1;
    res = Mat(Mat(2 * rot_2x3) * res - shi_2x1) + shift;

    Mat tmp, res2;
    add(one_3x1, shi_3x1, tmp);
    add(tmp, shi_3x1, tmp);
    add(tmp, shi_3x1, tmp);
    gemm(rot_2x3, tmp, 2, shi_2x1, -1, res2, 0);
    add(res2, Mat(2, 1, CV_32F, shift), res2);

    EXPECT_MAT_NEAR_RELATIVE(res, res2, 0);

    Mat mat4x4(4, 4, CV_32F);
    randu(mat4x4, Scalar(0), Scalar(10));

    Mat roi1 = mat4x4(Rect(Point(1, 1), Size(2, 2)));
    Mat roi2 = mat4x4(cv::Range(1, 3), cv::Range(1, 3));

    EXPECT_MAT_NEAR_RELATIVE(roi1, roi2, 0);
    EXPECT_MAT_NEAR_RELATIVE(mat4x4, mat4x4(Rect(Point(0,0), mat4x4.size())), 0);

    Mat intMat10(3, 3, CV_32S, Scalar(10));
    Mat intMat11(3, 3, CV_32S, Scalar(11));
    Mat resMat(3, 3, CV_8U, Scalar(255));

    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 == intMat10, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 <  intMat11, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat11 >  intMat10, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 <= intMat11, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat11 >= intMat10, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat11 != intMat10, 0);

    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 == 10.0, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, 10.0 == intMat10, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 <  11.0, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, 11.0 > intMat10, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, 10.0 < intMat11, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, 11.0 >= intMat10, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, 10.0 <= intMat11, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, 10.0 != intMat11, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat11 != 10.0, 0);

    Mat eye =  Mat::eye(3, 3, CV_16S);
    Mat maskMat4(3, 3, CV_16S, Scalar(4));
    Mat maskMat1(3, 3, CV_16S, Scalar(1));
    Mat maskMat5(3, 3, CV_16S, Scalar(5));
    Mat maskMat0(3, 3, CV_16S, Scalar(0));

    EXPECT_MAT_NEAR_RELATIVE(maskMat0, maskMat4 & maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, Scalar(1) & maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, maskMat4 & Scalar(1), 0);

    Mat m;
    m = maskMat4.clone(); m &= maskMat1; EXPECT_MAT_NEAR_RELATIVE(maskMat0, m, 0);
    m = maskMat4.clone(); m &= maskMat1 | maskMat1; EXPECT_MAT_NEAR_RELATIVE(maskMat0, m, 0);
    m = maskMat4.clone(); m &= (2* maskMat1 - maskMat1); EXPECT_MAT_NEAR_RELATIVE(maskMat0, m, 0);

    m = maskMat4.clone(); m &= Scalar(1); EXPECT_MAT_NEAR_RELATIVE(maskMat0, m, 0);
    m = maskMat4.clone(); m |= maskMat1; EXPECT_MAT_NEAR_RELATIVE(maskMat5, m, 0);
    m = maskMat5.clone(); m ^= maskMat1; EXPECT_MAT_NEAR_RELATIVE(maskMat4, m, 0);
    m = maskMat4.clone(); m |= (2* maskMat1 - maskMat1); EXPECT_MAT_NEAR_RELATIVE(maskMat5, m, 0);
    m = maskMat5.clone(); m ^= (2* maskMat1 - maskMat1); EXPECT_MAT_NEAR_RELATIVE(maskMat4, m, 0);

    m = maskMat4.clone(); m |= Scalar(1); EXPECT_MAT_NEAR_RELATIVE(maskMat5, m, 0);
    m = maskMat5.clone(); m ^= Scalar(1); EXPECT_MAT_NEAR_RELATIVE(maskMat4, m, 0);



    EXPECT_MAT_NEAR_RELATIVE(maskMat0, (maskMat4 | maskMat4) & (maskMat1 | maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, (maskMat4 | maskMat4) & maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, maskMat4 & (maskMat1 | maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, (maskMat1 | maskMat1) & Scalar(4), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, Scalar(4) & (maskMat1 | maskMat1), 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat0, maskMat5 ^ (maskMat4 | maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, (maskMat4 | maskMat1) ^ maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, (maskMat4 + maskMat1) ^ (maskMat4 + maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, Scalar(5) ^ (maskMat4 | Scalar(1)), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat1, Scalar(5) ^ maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, Scalar(5) ^ (maskMat4 + maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, Scalar(5) | (maskMat4 + maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, (maskMat4 + maskMat1) ^ Scalar(5), 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat5, maskMat5 | (maskMat4 ^ maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, (maskMat4 ^ maskMat1) | maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, maskMat5 | (maskMat4 ^ Scalar(1)), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, (maskMat4 | maskMat4) | Scalar(1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, Scalar(1) | (maskMat4 | maskMat4), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, Scalar(1) | maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, (maskMat5 | maskMat5) | (maskMat4 ^ maskMat1), 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat1, min(maskMat1, maskMat5), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat1, min(Mat(maskMat1 | maskMat1), maskMat5 | maskMat5), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, max(maskMat1, maskMat5), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, max(Mat(maskMat1 | maskMat1), maskMat5 | maskMat5), 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat1, min(maskMat1, maskMat5 | maskMat5), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat1, min(maskMat1 | maskMat1, maskMat5), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, max(maskMat1 | maskMat1, maskMat5), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, max(maskMat1, maskMat5 | maskMat5), 0);

    EXPECT_MAT_NEAR_RELATIVE(~maskMat1, maskMat1 ^ -1, 0);
    EXPECT_MAT_NEAR_RELATIVE(~(maskMat1 | maskMat1), maskMat1 ^ -1, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat1, maskMat4/4.0, 0);

    /////////////////////////////

    EXPECT_MAT_NEAR_RELATIVE(1.0 - (maskMat5 | maskMat5), -maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 | maskMat4) * 1.0 + 1.0, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(1.0 + (maskMat4 | maskMat4) * 1.0, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat5 | maskMat5) * 1.0 - 1.0, maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(5.0 - (maskMat4 | maskMat4) * 1.0, maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 | maskMat4) * 1.0 + 0.5 + 0.5, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(0.5 + ((maskMat4 | maskMat4) * 1.0 + 0.5), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(((maskMat4 | maskMat4) * 1.0 + 2.0) - 1.0, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(5.0 - ((maskMat1 | maskMat1) * 1.0 + 3.0), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE( ( (maskMat1 | maskMat1) * 2.0 + 2.0) * 1.25, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE( 1.25 * ( (maskMat1 | maskMat1) * 2.0 + 2.0), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE( -( (maskMat1 | maskMat1) * (-2.0) + 1.0), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE( maskMat1 * 1.0 + maskMat4 * 0.5 + 2.0, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE( 1.0 + (maskMat1 * 1.0 + maskMat4 * 0.5 + 1.0), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE( (maskMat1 * 1.0 + maskMat4 * 0.5 + 2.0) - 1.0, maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(5.0 -  (maskMat1 * 1.0 + maskMat4 * 0.5 + 1.0), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1 * 1.0 + maskMat4 * 0.5 + 1.0)*1.25, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(1.25 * (maskMat1 * 1.0 + maskMat4 * 0.5 + 1.0), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(-(maskMat1 * 2.0 + maskMat4 * (-1) + 1.0), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1 * 1.0 + maskMat4), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 + maskMat1 * 1.0), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1 * 3.0 + 1.0) + maskMat1, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat1 + (maskMat1 * 3.0 + 1.0), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat1*4.0 + (maskMat1 | maskMat1), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1 | maskMat1) + maskMat1*4.0, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1*3.0 + 1.0) + (maskMat1 | maskMat1), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1 | maskMat1) + (maskMat1*3.0 + 1.0), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat1*4.0 + maskMat4*2.0, maskMat1 * 12, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1*3.0 + 1.0) + maskMat4*2.0, maskMat1 * 12, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat4*2.0 + (maskMat1*3.0 + 1.0), maskMat1 * 12, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1*3.0 + 1.0) + (maskMat1*2.0 + 2.0), maskMat1 * 8, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat5*1.0 - maskMat4, maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5 - maskMat1 * 4.0, maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 * 1.0 + 4.0)- maskMat4, maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5 - (maskMat1 * 2.0 + 2.0), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5*1.0 - (maskMat4 | maskMat4), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat5 | maskMat5) - maskMat1 * 4.0, maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 * 1.0 + 4.0)- (maskMat4 | maskMat4), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat5 | maskMat5) - (maskMat1 * 2.0 + 2.0), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat1*5.0 - maskMat4 * 1.0, maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1*5.0 + 3.0)- maskMat4 * 1.0, maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat4 * 2.0 - (maskMat1*4.0 + 3.0), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1 * 2.0 + 3.0) - (maskMat1*3.0 + 1.0), maskMat1, 0);

    EXPECT_MAT_NEAR_RELATIVE((maskMat5 - maskMat4)* 4.0, maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(4.0 * (maskMat5 - maskMat4), maskMat4, 0);

    EXPECT_MAT_NEAR_RELATIVE(-((maskMat4 | maskMat4) - (maskMat5 | maskMat5)), maskMat1, 0);

    EXPECT_MAT_NEAR_RELATIVE(4.0 * (maskMat1 | maskMat1), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 | maskMat4)/4.0, maskMat1, 0);

#if !MSVC_OLD
    EXPECT_MAT_NEAR_RELATIVE(2.0 * (maskMat1 * 2.0) , maskMat4, 0);
#endif
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 / 2.0) / 2.0 , maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(-(maskMat4 - maskMat5) , maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(-((maskMat4 - maskMat5) * 1.0), maskMat1, 0);


    /////////////////////////////
    EXPECT_MAT_NEAR_RELATIVE(maskMat4 /  maskMat4, maskMat1, 0);

    ///// Element-wise multiplication

    EXPECT_MAT_NEAR_RELATIVE(maskMat4.mul(maskMat4, 0.25), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat4.mul(maskMat1 * 4, 0.25), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat4.mul(maskMat4 / 4), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat4.mul(maskMat4 / 4), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat4.mul(maskMat4) * 0.25, maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(0.25 * maskMat4.mul(maskMat4), maskMat4, 0);

    ////// Element-wise division

    EXPECT_MAT_NEAR_RELATIVE(maskMat4 / maskMat4, maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 & maskMat4) / (maskMat1 * 4), maskMat1, 0);

    EXPECT_MAT_NEAR_RELATIVE((maskMat4 & maskMat4) / maskMat4, maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat4 / (maskMat4 & maskMat4), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1 * 4) / maskMat4, maskMat1, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat4 / (maskMat1 * 4), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 * 0.5 )/ (maskMat1 * 2), maskMat1, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat4 / maskMat4.mul(maskMat1), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat4 & maskMat4) / maskMat4.mul(maskMat1), maskMat1, 0);

    EXPECT_MAT_NEAR_RELATIVE(4.0 / maskMat4, maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(4.0 / (maskMat4 | maskMat4), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(4.0 / (maskMat1 * 4.0), maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(4.0 / (maskMat4 / maskMat1), maskMat1, 0);

    m = maskMat4.clone(); m/=4.0; EXPECT_MAT_NEAR_RELATIVE(m, maskMat1, 0);
    m = maskMat4.clone(); m/=maskMat4; EXPECT_MAT_NEAR_RELATIVE(m, maskMat1, 0);
    m = maskMat4.clone(); m/=(maskMat1 * 4.0); EXPECT_MAT_NEAR_RELATIVE(m, maskMat1, 0);
    m = maskMat4.clone(); m/=(maskMat4 / maskMat1); EXPECT_MAT_NEAR_RELATIVE(m, maskMat1, 0);

    /////////////////////////////
    float matrix_data[] = { 3, 1, -4, -5, 1, 0, 0, 1.1f, 1.5f};
    Mat mt(3, 3, CV_32F, matrix_data);
    Mat mi = mt.inv();
    Mat d1 = Mat::eye(3, 3, CV_32F);
    Mat d2 = d1 * 2;
    MatExpr mt_tr = mt.t();
    MatExpr mi_tr = mi.t();
    Mat mi2 = mi * 2;

    EXPECT_MAT_NEAR_RELATIVE(mi2 * mt, d2 , 1e-5);
    EXPECT_MAT_NEAR_RELATIVE(mi * mt, d1 , 1e-5);
    EXPECT_MAT_NEAR_RELATIVE(mt_tr * mi_tr, d1 , 1e-5);

    m = mi.clone();
    m*=mt;
    EXPECT_MAT_NEAR_RELATIVE(m, d1, 1e-5);

    m = mi.clone();
    m*= (2 * mt - mt) ;
    EXPECT_MAT_NEAR_RELATIVE(m, d1, 1e-5);

    m = maskMat4.clone();
    m+=(maskMat1 * 1.0);
    EXPECT_MAT_NEAR_RELATIVE(m, maskMat5, 0);
    m = maskMat5.clone();
    m-=(maskMat1 * 4.0);
    EXPECT_MAT_NEAR_RELATIVE(m, maskMat1, 0);

    m = maskMat1.clone();
    m+=(maskMat1 * 3.0 + 1.0);
    EXPECT_MAT_NEAR_RELATIVE(m, maskMat5, 0);

    m = maskMat5.clone();
    m-=(maskMat1 * 3.0 + 1.0);
    EXPECT_MAT_NEAR_RELATIVE(m, maskMat1, 0);

#if !MSVC_OLD
    m = mi.clone();
    m+=(3.0 * mi * mt + d1);
    EXPECT_MAT_NEAR_RELATIVE(m, mi + d1 * 4, 1e-5);

    m = mi.clone();
    m-=(3.0 * mi * mt + d1);
    EXPECT_MAT_NEAR_RELATIVE(m, mi - d1 * 4, 1e-5);

    m = mi.clone();
    m*=(mt * 1.0);
    EXPECT_MAT_NEAR_RELATIVE(m, d1, 1e-5);

    m = mi.clone();
    m*=(mt * 1.0 + Mat::eye(m.size(), m.type()));
    EXPECT_MAT_NEAR_RELATIVE(m, d1 + mi, 1e-5);

    m = mi.clone();
    m*=mt_tr.t();
    EXPECT_MAT_NEAR_RELATIVE(m, d1, 1e-5);

    EXPECT_MAT_NEAR_RELATIVE( (mi * 2) * mt, d2, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( mi * (2 * mt), d2, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( mt.t() * mi_tr, d1 , 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( mt_tr * mi.t(), d1 , 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( (mi * 0.4) * (mt * 5), d2, 1e-5);

    EXPECT_MAT_NEAR_RELATIVE( mt.t() * (mi_tr * 2), d2 , 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( (mt_tr * 2) * mi.t(), d2 , 1e-5);

    EXPECT_MAT_NEAR_RELATIVE(mt.t() * mi.t(), d1, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( (mi * mt) * 2.0, d2, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( 2.0 * (mi * mt), d2, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( -(mi * mt), -d1, 1e-5);

    EXPECT_MAT_NEAR_RELATIVE( (mi * mt) / 2.0, d1 / 2, 1e-5);

    Mat mt_mul_2_plus_1;
    gemm(mt, d1, 2, Mat::ones(3, 3, CV_32F), 1, mt_mul_2_plus_1);

    EXPECT_MAT_NEAR_RELATIVE( (mt * 2.0 + 1.0) * mi, mt_mul_2_plus_1 * mi, 0);        // (A*alpha + beta)*B
    EXPECT_MAT_NEAR_RELATIVE( mi * (mt * 2.0 + 1.0), mi * mt_mul_2_plus_1, 0);        // A*(B*alpha + beta)
    EXPECT_MAT_NEAR_RELATIVE( (mt * 2.0 + 1.0) * (mi * 2), mt_mul_2_plus_1 * mi2, 0); // (A*alpha + beta)*(B*gamma)
    EXPECT_MAT_NEAR_RELATIVE( (mi *2)* (mt * 2.0 + 1.0), mi2 * mt_mul_2_plus_1, 0);   // (A*gamma)*(B*alpha + beta)
    EXPECT_MAT_NEAR_RELATIVE( (mt * 2.0 + 1.0) * mi.t(), mt_mul_2_plus_1 * mi_tr, 1e-5); // (A*alpha + beta)*B^t
    EXPECT_MAT_NEAR_RELATIVE( mi.t() * (mt * 2.0 + 1.0), mi_tr * mt_mul_2_plus_1, 1e-5); // A^t*(B*alpha + beta)

    EXPECT_MAT_NEAR_RELATIVE( (mi * mt + d2)*5, d1 * 3 * 5, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( mi * mt + d2, d1 * 3, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( -(mi * mt) + d2, d1, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( (mi * mt) + d1, d2, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( d1 + (mi * mt), d2, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( (mi * mt) - d2, -d1, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( d2 - (mi * mt), d1, 1e-5);

    EXPECT_MAT_NEAR_RELATIVE( (mi * mt) + d2 * 0.5, d2, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( d2 * 0.5 + (mi * mt), d2, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( (mi * mt) - d1 * 2, -d1, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( d1 * 2 - (mi * mt), d1, 1e-5);

    EXPECT_MAT_NEAR_RELATIVE( (mi * mt) + mi.t(), mi_tr + d1, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( mi.t() + (mi * mt), mi_tr + d1, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( (mi * mt) - mi.t(), d1 - mi_tr, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( mi.t() - (mi * mt), mi_tr - d1, 1e-5);

    EXPECT_MAT_NEAR_RELATIVE( 2.0 *(mi * mt + d2), d1 * 6, 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( -(mi * mt + d2), d1 * -3, 1e-5);

    EXPECT_MAT_NEAR_RELATIVE(mt.inv() * mt, d1, 1e-5);

    EXPECT_MAT_NEAR_RELATIVE(mt.inv() * (2*mt - mt), d1, 1e-5);
#endif
}

TEST(Core_Array, expressions_MatFunctions)
{
    Mat rgba( 10, 10, CV_8UC4, Scalar(1,2,3,4) );
    Mat bgr( rgba.rows, rgba.cols, CV_8UC3 );
    Mat alpha( rgba.rows, rgba.cols, CV_8UC1 );
    Mat out[] = { bgr, alpha };
    // rgba[0] -> bgr[2], rgba[1] -> bgr[1],
    // rgba[2] -> bgr[0], rgba[3] -> alpha[0]
    int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
    mixChannels( &rgba, 1, out, 2, from_to, 4 );

    Mat bgr_exp( rgba.size(), CV_8UC3, Scalar(3,2,1));
    Mat alpha_exp( rgba.size(), CV_8UC1, Scalar(4));

    EXPECT_MAT_NEAR_RELATIVE(bgr_exp, bgr, 0);
    EXPECT_MAT_NEAR_RELATIVE(alpha_exp, alpha, 0);
}

TEST(Core_Array, expressions_SubMatAccess)
{
    Mat_<float> T_bs(4,4);
    Vec3f cdir(1.f, 1.f, 0.f);
    Vec3f ydir(1.f, 0.f, 1.f);
    Vec3f fpt(0.1f, 0.7f, 0.2f);
    T_bs.setTo(0);
    T_bs(cv::Range(0,3),cv::Range(2,3)) = 1.0*Mat(cdir); // wierd OpenCV stuff, need to do multiply
    T_bs(cv::Range(0,3),cv::Range(1,2)) = 1.0*Mat(ydir);
    T_bs(cv::Range(0,3),cv::Range(0,1)) = 1.0*Mat(cdir.cross(ydir));
    T_bs(cv::Range(0,3),cv::Range(3,4)) = 1.0*Mat(fpt);
    T_bs(3,3) = 1.0;

    // set up display coords, really just the S frame
    std::vector<float>coords;

    for (int i=0; i<16; i++)
    {
        coords.push_back(T_bs(i));
    }
    EXPECT_FLOAT_EQ(0, cvtest::norm(coords, T_bs.reshape(1,1), NORM_INF));
}


template<typename _Tp> void TestType(Size sz, _Tp value)
{
    cv::Mat_<_Tp> m(sz);
    ASSERT_EQ(m.cols, sz.width);
    ASSERT_EQ(m.rows, sz.height);
    ASSERT_EQ(m.depth(), DataType<_Tp>::depth);
    ASSERT_EQ(m.channels(), DataType<_Tp>::channels);
    ASSERT_EQ(m.elemSize(), sizeof(_Tp));
    ASSERT_EQ(m.step, m.elemSize()*m.cols);

    for( int y = 0; y < sz.height; y++ )
        for( int x = 0; x < sz.width; x++ )
            m(y,x) = value;

    double s = sum(Mat(m).reshape(1))[0];
    ASSERT_EQ(s, (double)sz.width*sz.height);
}


TEST(Core_Array, expressions_TemplateMat)
{
    Mat_<float> one_3x1(3, 1, 1.0f);
    Mat_<float> shi_3x1(3, 1, 1.2f);
    Mat_<float> shi_2x1(2, 1, -2);
    Scalar shift = Scalar::all(15);

    float data[] = { sqrt(2.f)/2, -sqrt(2.f)/2, 1.f, sqrt(2.f)/2, sqrt(2.f)/2, 10.f };
    Mat_<float> rot_2x3(2, 3, data);

    Mat_<float> res = Mat(Mat(2 * rot_2x3) * Mat(one_3x1 + shi_3x1 + shi_3x1 + shi_3x1) - shi_2x1) + shift;
    Mat_<float> resS = rot_2x3 * one_3x1;

    Mat_<float> tmp, res2, resS2;
    add(one_3x1, shi_3x1, tmp);
    add(tmp, shi_3x1, tmp);
    add(tmp, shi_3x1, tmp);
    gemm(rot_2x3, tmp, 2, shi_2x1, -1, res2, 0);
    add(res2, Mat(2, 1, CV_32F, shift), res2);

    gemm(rot_2x3, one_3x1, 1, shi_2x1, 0, resS2, 0);
    EXPECT_MAT_NEAR_RELATIVE(res, res2, 0);
    EXPECT_MAT_NEAR_RELATIVE(resS, resS2, 0);


    Mat_<float> mat4x4(4, 4);
    randu(mat4x4, Scalar(0), Scalar(10));

    Mat_<float> roi1 = mat4x4(Rect(Point(1, 1), Size(2, 2)));
    Mat_<float> roi2 = mat4x4(cv::Range(1, 3), cv::Range(1, 3));

    EXPECT_MAT_NEAR_RELATIVE(roi1, roi2, 0);
    EXPECT_MAT_NEAR_RELATIVE(mat4x4, mat4x4(Rect(Point(0,0), mat4x4.size())), 0);

    Mat_<int> intMat10(3, 3, 10);
    Mat_<int> intMat11(3, 3, 11);
    Mat_<uchar> resMat(3, 3, 255);

    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 == intMat10, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 <  intMat11, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat11 >  intMat10, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 <= intMat11, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat11 >= intMat10, 0);

    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 == 10.0, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 <  11.0, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat11 >  10.0, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat10 <= 11.0, 0);
    EXPECT_MAT_NEAR_RELATIVE(resMat, intMat11 >= 10.0, 0);

    Mat_<uchar> maskMat4(3, 3, 4);
    Mat_<uchar> maskMat1(3, 3, 1);
    Mat_<uchar> maskMat5(3, 3, 5);
    Mat_<uchar> maskMat0(3, 3, (uchar)0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat0, maskMat4 & maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, Scalar(1) & maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, maskMat4 & Scalar(1), 0);

    Mat_<uchar> m;
    m = maskMat4.clone(); m&=maskMat1; EXPECT_MAT_NEAR_RELATIVE(maskMat0, m, 0);
    m = maskMat4.clone(); m&=Scalar(1); EXPECT_MAT_NEAR_RELATIVE(maskMat0, m, 0);

    m = maskMat4.clone(); m|=maskMat1; EXPECT_MAT_NEAR_RELATIVE(maskMat5, m, 0);
    m = maskMat4.clone(); m^=maskMat1; EXPECT_MAT_NEAR_RELATIVE(maskMat5, m, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat0, (maskMat4 | maskMat4) & (maskMat1 | maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, (maskMat4 | maskMat4) & maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, maskMat4 & (maskMat1 | maskMat1), 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat0, maskMat5 ^ (maskMat4 | maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat0, Scalar(5) ^ (maskMat4 | Scalar(1)), 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat5, maskMat5 | (maskMat4 ^ maskMat1), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, maskMat5 | (maskMat4 ^ Scalar(1)), 0);

    EXPECT_MAT_NEAR_RELATIVE(~maskMat1, maskMat1 ^ 0xFF, 0);
    EXPECT_MAT_NEAR_RELATIVE(~(maskMat1 | maskMat1), maskMat1 ^ 0xFF, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat1 + maskMat4, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat1 + Scalar(4), maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(Scalar(4) + maskMat1, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(Scalar(4) + (maskMat1 & maskMat1), maskMat5, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat1 + 4.0, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat1 & 0xFF) + 4.0, maskMat5, 0);
    EXPECT_MAT_NEAR_RELATIVE(4.0 + maskMat1, maskMat5, 0);

    m = maskMat4.clone(); m+=Scalar(1); EXPECT_MAT_NEAR_RELATIVE(m, maskMat5, 0);
    m = maskMat4.clone(); m+=maskMat1; EXPECT_MAT_NEAR_RELATIVE(m, maskMat5, 0);
    m = maskMat4.clone(); m+=(maskMat1 | maskMat1); EXPECT_MAT_NEAR_RELATIVE(m, maskMat5, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat5 - maskMat1, maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5 - Scalar(1), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat5 | maskMat5) - Scalar(1), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5 - 1, maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat5 | maskMat5) - 1, maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE((maskMat5 | maskMat5) - (maskMat1 | maskMat1), maskMat4, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat1, min(maskMat1, maskMat5), 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat5, max(maskMat1, maskMat5), 0);

    m = maskMat5.clone(); m-=Scalar(1); EXPECT_MAT_NEAR_RELATIVE(m, maskMat4, 0);
    m = maskMat5.clone(); m-=maskMat1; EXPECT_MAT_NEAR_RELATIVE(m, maskMat4, 0);
    m = maskMat5.clone(); m-=(maskMat1 | maskMat1); EXPECT_MAT_NEAR_RELATIVE(m, maskMat4, 0);

    m = maskMat4.clone(); m |= Scalar(1); EXPECT_MAT_NEAR_RELATIVE(maskMat5, m, 0);
    m = maskMat5.clone(); m ^= Scalar(1); EXPECT_MAT_NEAR_RELATIVE(maskMat4, m, 0);

    EXPECT_MAT_NEAR_RELATIVE(maskMat1, maskMat4/4.0, 0);

    Mat_<float> negf(3, 3, -3.0);
    Mat_<float> posf = -negf;
    Mat_<float> posf2 = posf * 2;
    Mat_<int> negi(3, 3, -3);

    EXPECT_MAT_NEAR_RELATIVE(abs(negf), -negf, 0);
    EXPECT_MAT_NEAR_RELATIVE(abs(posf - posf2), -negf, 0);
    EXPECT_MAT_NEAR_RELATIVE(abs(negi), -(negi & negi), 0);

    EXPECT_MAT_NEAR_RELATIVE(5.0 - maskMat4, maskMat1, 0);


    EXPECT_MAT_NEAR_RELATIVE(maskMat4.mul(maskMat4, 0.25), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat4.mul(maskMat1 * 4, 0.25), maskMat4, 0);
    EXPECT_MAT_NEAR_RELATIVE(maskMat4.mul(maskMat4 / 4), maskMat4, 0);


    ////// Element-wise division

    EXPECT_MAT_NEAR_RELATIVE(maskMat4 / maskMat4, maskMat1, 0);
    EXPECT_MAT_NEAR_RELATIVE(4.0 / maskMat4, maskMat1, 0);
    m = maskMat4.clone(); m/=4.0; EXPECT_MAT_NEAR_RELATIVE(m, maskMat1, 0);

    ////////////////////////////////

    typedef Mat_<int> TestMat_t;

    const TestMat_t cnegi = negi.clone();

    TestMat_t::iterator beg = negi.begin();
    TestMat_t::iterator end = negi.end();

    TestMat_t::const_iterator cbeg = cnegi.begin();
    TestMat_t::const_iterator cend = cnegi.end();

    int sum = 0;
    for(; beg!=end; ++beg)
        sum+=*beg;

    for(; cbeg!=cend; ++cbeg)
        sum-=*cbeg;

    ASSERT_EQ(0, sum);

    EXPECT_MAT_NEAR_RELATIVE(negi.col(1), negi.col(2), 0);
    EXPECT_MAT_NEAR_RELATIVE(negi.row(1), negi.row(2), 0);
    EXPECT_MAT_NEAR_RELATIVE(negi.col(1), negi.diag(), 0);

    EXPECT_EQ(sizeof(float), Mat_<Point2f>(1, 1).elemSize1());
    EXPECT_EQ(2 * sizeof(float), Mat_<Point2f>(1, 1).elemSize());
    EXPECT_EQ(CV_32F, Mat_<Point2f>(1, 1).depth());
    EXPECT_EQ(CV_32F, Mat_<float>(1, 1).depth());
    EXPECT_EQ(CV_32S, Mat_<int>(1, 1).depth());
    EXPECT_EQ(CV_64F, Mat_<double>(1, 1).depth());
    EXPECT_EQ(CV_64F, Mat_<Point3d>(1, 1).depth());
    EXPECT_EQ(CV_8S, Mat_<signed char>(1, 1).depth());
    EXPECT_EQ(CV_16U, Mat_<unsigned short>(1, 1).depth());
    EXPECT_EQ(1, Mat_<unsigned short>(1, 1).channels());
    EXPECT_EQ(2, Mat_<Point2f>(1, 1).channels());
    EXPECT_EQ(3, Mat_<Point3f>(1, 1).channels());
    EXPECT_EQ(3, Mat_<Point3d>(1, 1).channels());

    Mat_<uchar> eye = Mat_<uchar>::zeros(2, 2);
    EXPECT_MAT_NEAR_RELATIVE(Mat_<uchar>::zeros(Size(2, 2)), eye, 0);

    eye.at<uchar>(Point(0,0)) = 1;
    eye.at<uchar>(1, 1) = 1;
    EXPECT_MAT_NEAR_RELATIVE(Mat_<uchar>::eye(2, 2), eye, 0);
    EXPECT_MAT_NEAR_RELATIVE(eye, Mat_<uchar>::eye(Size(2,2)), 0);

    Mat_<uchar> ones(2, 2, (uchar)1);
    EXPECT_MAT_NEAR_RELATIVE(ones, Mat_<uchar>::ones(Size(2,2)), 0);
    EXPECT_MAT_NEAR_RELATIVE(Mat_<uchar>::ones(2, 2), ones, 0);

    Mat_<Point2f> pntMat(2, 2, Point2f(1, 0));
    EXPECT_EQ((size_t)2, pntMat.stepT());

    uchar uchar_data[] = {1, 0, 0, 1};

    Mat_<uchar> matFromData(1, 4, uchar_data);
    const Mat_<uchar> mat2 = matFromData.clone();
    EXPECT_MAT_NEAR_RELATIVE(matFromData, eye.reshape(1, 1), 0);
    EXPECT_EQ(uchar_data[0], matFromData(Point(0,0)));
    EXPECT_EQ(uchar_data[0], mat2(Point(0,0)));

    EXPECT_EQ(uchar_data[0], matFromData(0,0));
    EXPECT_EQ(uchar_data[0], mat2(0,0));

    Mat_<uchar> rect(eye, Rect(0, 0, 1, 1));
    EXPECT_EQ(1, rect.cols );
    EXPECT_EQ(1, rect.rows );
    EXPECT_EQ(uchar_data[0], rect(0,0) );

    ///////////////////////////////

    float matrix_data[] = { 3, 1, -4, -5, 1, 0, 0, 1.1f, 1.5f};
    Mat_<float> mt(3, 3, matrix_data);
    Mat_<float> mi = mt.inv();
    Mat_<float> d1 = Mat_<float>::eye(3, 3);
    Mat_<float> d2 = d1 * 2;
    Mat_<float> mt_tr = mt.t();
    Mat_<float> mi_tr = mi.t();
    Mat_<float> mi2 = mi * 2;

    EXPECT_MAT_NEAR_RELATIVE( mi2 * mt, d2 , 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( mi * mt, d1 , 1e-5);
    EXPECT_MAT_NEAR_RELATIVE( mt_tr * mi_tr, d1 , 1e-5);

    Mat_<float> mf;
    mf = mi.clone();
    mf*=mt;
    EXPECT_MAT_NEAR_RELATIVE(mf, d1, 1e-5);

    ////// typedefs //////

    EXPECT_EQ(sizeof(uchar), Mat1b(1, 1).elemSize() );
    EXPECT_EQ(2 * sizeof(uchar), Mat2b(1, 1).elemSize() );
    EXPECT_EQ(3 * sizeof(uchar), Mat3b(1, 1).elemSize() );
    EXPECT_EQ(sizeof(float), Mat1f(1, 1).elemSize() );
    EXPECT_EQ(2 * sizeof(float), Mat2f(1, 1).elemSize() );
    EXPECT_EQ(3 * sizeof(float), Mat3f(1, 1).elemSize() );
    EXPECT_EQ(CV_32F, Mat1f(1, 1).depth() );
    EXPECT_EQ(CV_32F, Mat3f(1, 1).depth() );
    EXPECT_EQ(CV_32FC3, Mat3f(1, 1).type() );
    EXPECT_EQ(CV_32S, Mat1i(1, 1).depth() );
    EXPECT_EQ(CV_64F, Mat1d(1, 1).depth() );
    EXPECT_EQ(CV_8U, Mat1b(1, 1).depth() );
    EXPECT_EQ(CV_8UC3, Mat3b(1, 1).type() );
    EXPECT_EQ(CV_16U, Mat1w(1, 1).depth() );
    EXPECT_EQ(CV_16S, Mat1s(1, 1).depth() );
    EXPECT_EQ(1, Mat1f(1, 1).channels() );
    EXPECT_EQ(1, Mat1b(1, 1).channels() );
    EXPECT_EQ(1, Mat1i(1, 1).channels() );
    EXPECT_EQ(1, Mat1w(1, 1).channels() );
    EXPECT_EQ(1, Mat1s(1, 1).channels() );
    EXPECT_EQ(2, Mat2f(1, 1).channels() );
    EXPECT_EQ(2, Mat2b(1, 1).channels() );
    EXPECT_EQ(2, Mat2i(1, 1).channels() );
    EXPECT_EQ(2, Mat2w(1, 1).channels() );
    EXPECT_EQ(2, Mat2s(1, 1).channels() );
    EXPECT_EQ(3, Mat3f(1, 1).channels() );
    EXPECT_EQ(3, Mat3b(1, 1).channels() );
    EXPECT_EQ(3, Mat3i(1, 1).channels() );
    EXPECT_EQ(3, Mat3w(1, 1).channels() );
    EXPECT_EQ(3, Mat3s(1, 1).channels() );

    vector<Mat_<float> > mvf, mvf2;
    Mat_<Vec2f> mf2;
    mvf.push_back(Mat_<float>::ones(4, 3));
    mvf.push_back(Mat_<float>::zeros(4, 3));
    merge(mvf, mf2);
    split(mf2, mvf2);
    EXPECT_FLOAT_EQ(0, cvtest::norm(mvf2[0], mvf[0], CV_C));
    EXPECT_FLOAT_EQ(0, cvtest::norm(mvf2[1], mvf[1], CV_C));

    {
        Mat a(2,2,CV_32F,1.f);
        Mat b(1,2,CV_32F,1.f);
        Mat c = (a*b.t()).t();
        EXPECT_FLOAT_EQ(4., cvtest::norm(c, CV_L1));
    }

    {
        Mat m1 = Mat::zeros(1, 10, CV_8UC1);
        Mat m2 = Mat::zeros(10, 10, CV_8UC3);
        EXPECT_THROW(m1.copyTo(m2.row(1)), cv::Exception);
    }

    Size size(2, 5);
    TestType<float>(size, 1.f);
    cv::Vec3f val1 = 1.f;
    TestType<cv::Vec3f>(size, val1);
    cv::Matx31f val2 = 1.f;
    TestType<cv::Matx31f>(size, val2);
    cv::Matx41f val3 = 1.f;
    TestType<cv::Matx41f>(size, val3);
    cv::Matx32f val4 = 1.f;
    TestType<cv::Matx32f>(size, val4);
}

TEST(Core_Array, expressions_MatND)
{
    int sizes[] = { 3, 3, 3};
    cv::MatND nd(3, sizes, CV_32F);
}

TEST(Core_Array, expressions_SparseMat)
{
    int sizes[] = { 10, 10, 10};
    int dims = sizeof(sizes)/sizeof(sizes[0]);
    SparseMat mat(dims, sizes, CV_32FC2);

    EXPECT_EQ(mat.dims(), dims);
    EXPECT_EQ(2, mat.channels() );
    EXPECT_EQ(CV_32F, mat.depth() );

    SparseMat mat2 = mat.clone();
}


TEST(Core_Array, expressions_MatxMultiplication)
{
    Matx33f mat(1, 1, 1, 0, 1, 1, 0, 0, 1); // Identity matrix
    Point2f pt(3, 4);
    Point3f res = mat * pt; // Correctly assumes homogeneous coordinates

    Vec3f res2 = mat*Vec3f(res.x, res.y, res.z);

    EXPECT_EQ(8.0, res.x );
    EXPECT_EQ(5.0, res.y );
    EXPECT_EQ(1.0, res.z );

    EXPECT_EQ(14.0, res2[0] );
    EXPECT_EQ(6.0, res2[1] );
    EXPECT_EQ(1.0, res2[2] );

    Matx44f mat44f(1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1);
    Matx44d mat44d(1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1);
    Scalar s(4, 3, 2, 1);
    Scalar sf = mat44f*s;
    Scalar sd = mat44d*s;

    EXPECT_EQ(10.0, sf[0] );
    EXPECT_EQ(6.0, sf[1] );
    EXPECT_EQ(3.0, sf[2] );
    EXPECT_EQ(1.0, sf[3] );

    EXPECT_EQ(10.0, sd[0] );
    EXPECT_EQ(6.0, sd[1] );
    EXPECT_EQ(3.0, sd[2] );
    EXPECT_EQ(1.0, sd[3] );
}

TEST(Core_Array, expressions_MatxElementwiseDivison)
{
    Matx22f mat(2, 4, 6, 8);
    Matx22f mat2(2, 2, 2, 2);

    Matx22f res = mat.div(mat2);

    EXPECT_EQ(1.0, res(0, 0) );
    EXPECT_EQ(2.0, res(0, 1) );
    EXPECT_EQ(3.0, res(1, 0) );
    EXPECT_EQ(4.0, res(1, 1) );
}


TEST(Core_Array, expressions_Vec)
{
    cv::Mat hsvImage_f(5, 5, CV_32FC3), hsvImage_b(5, 5, CV_8UC3);
    int i = 0,j = 0;
    cv::Vec3f a;

    //these compile
    cv::Vec3b b = a;
    hsvImage_f.at<cv::Vec3f>(i,j) = cv::Vec3f((float)i,0,1);
    hsvImage_b.at<cv::Vec3b>(i,j) = cv::Vec3b(cv::Vec3f((float)i,0,1));

    //these don't
    b = cv::Vec3f(1,0,0);
    cv::Vec3b c;
    c = cv::Vec3f(0,0,1);
    hsvImage_b.at<cv::Vec3b>(i,j) = cv::Vec3f((float)i,0,1);
    hsvImage_b.at<cv::Vec3b>(i,j) = a;
    hsvImage_b.at<cv::Vec3b>(i,j) = cv::Vec3f(1,2,3);
}

TEST(Core_Array, expressions_1)
{
    Point3d p1(1, 1, 1), p2(2, 2, 2), p4(4, 4, 4);
    p1*=2;
    EXPECT_EQ(p2, p1);
    EXPECT_EQ(p4, p2 * 2);
    EXPECT_EQ(p4, p2 * 2.f);
    EXPECT_EQ(p4, p2 * 2.f);

    Point2d pi1(1, 1), pi2(2, 2), pi4(4, 4);
    pi1*=2;
    EXPECT_EQ(pi2, pi1);
    EXPECT_EQ(pi4, pi2 * 2);
    EXPECT_EQ(pi4, pi2 * 2.f);
    EXPECT_EQ(pi4, pi2 * 2.f);

    Vec2d v12(1, 1), v22(2, 2);
    v12*=2.0;
    EXPECT_EQ(v12, v22);

    Vec3d v13(1, 1, 1), v23(2, 2, 2);
    v13*=2.0;
    EXPECT_EQ(v13, v23);

    Vec4d v14(1, 1, 1, 1), v24(2, 2, 2, 2);
    v14*=2.0;
    EXPECT_EQ(v14, v24);

    Size sz(10, 20);
    EXPECT_EQ(200, sz.area() );
    EXPECT_EQ(10, sz.width );
    EXPECT_EQ(20, sz.height );
    EXPECT_EQ(10, ((CvSize)sz).width );
    EXPECT_EQ(20, ((CvSize)sz).height );

    Vec<double, 5> v5d(1, 1, 1, 1, 1);
    Vec<double, 6> v6d(1, 1, 1, 1, 1, 1);
    Vec<double, 7> v7d(1, 1, 1, 1, 1, 1, 1);
    Vec<double, 8> v8d(1, 1, 1, 1, 1, 1, 1, 1);
    Vec<double, 9> v9d(1, 1, 1, 1, 1, 1, 1, 1, 1);
    Vec<double,10> v10d(1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

    Vec<double,10> v10dzero;
    for (int ii = 0; ii < 10; ++ii)
    {
        EXPECT_EQ(0.0, v10dzero[ii] );
    }

    Mat A(1, 32, CV_32F), B;
    for( int i = 0; i < A.cols; i++ )
        A.at<float>(i) = (float)(i <= 12 ? i : 24 - i);
    transpose(A, B);

    int minidx[2] = {0, 0}, maxidx[2] = {0, 0};
    double minval = 0, maxval = 0;
    minMaxIdx(A, &minval, &maxval, minidx, maxidx);

    EXPECT_EQ(0, minidx[0] );
    EXPECT_EQ(31, minidx[1] );
    EXPECT_EQ(0, maxidx[0] );
    EXPECT_EQ(12, maxidx[1] );
    EXPECT_EQ(-7, minval );
    EXPECT_EQ(12, maxval );

    minMaxIdx(B, &minval, &maxval, minidx, maxidx);

    EXPECT_EQ(31 , minidx[0] );
    EXPECT_EQ(0 , minidx[1] );
    EXPECT_EQ(12 , maxidx[0] );
    EXPECT_EQ(0 , maxidx[1] );
    EXPECT_EQ(-7 , minval );
    EXPECT_EQ(12, maxval );

    Matx33f b(1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f);
    Mat c;
    add(Mat::zeros(3, 3, CV_32F), b, c);
    EXPECT_FLOAT_EQ(0, cvtest::norm(b, c, CV_C));

    add(Mat::zeros(3, 3, CV_64F), b, c, noArray(), c.type());
    EXPECT_FLOAT_EQ(0, cvtest::norm(b, c, CV_C));

    add(Mat::zeros(6, 1, CV_64F), 1, c, noArray(), c.type());
    EXPECT_FLOAT_EQ(0, cvtest::norm(Matx61f(1.f, 1.f, 1.f, 1.f, 1.f, 1.f), c, CV_C));

    vector<Point2f> pt2d(3);
    vector<Point3d> pt3d(2);

    EXPECT_EQ(3, Mat(pt2d).checkVector(2));
    EXPECT_GT(0, Mat(pt2d).checkVector(3));
    EXPECT_GT(0, Mat(pt3d).checkVector(2));
    EXPECT_EQ(2, Mat(pt3d).checkVector(3));

    Matx44f m44(0.8147f, 0.6324f, 0.9575f, 0.9572f,
            0.9058f, 0.0975f, 0.9649f, 0.4854f,
            0.1270f, 0.2785f, 0.1576f, 0.8003f,
            0.9134f, 0.5469f, 0.9706f, 0.1419f);
    EXPECT_NEAR(-0.0262, determinant(m44), 0.001);

    Cv32suf z;
    z.i = 0x80000000;
    EXPECT_FLOAT_EQ(0, cvFloor(z.f));
    EXPECT_FLOAT_EQ(0, cvCeil(z.f));
    EXPECT_FLOAT_EQ(0, cvRound(z.f));
}

TEST(Core_Array, expressions_Exp)
{
    Mat1f tt = Mat1f::ones(4,2);
    Mat1f outs;
    exp(-tt, outs);
    Mat1f tt2 = Mat1f::ones(4,1), outs2;
    exp(-tt2, outs2);
}

TEST(Core_Array, expressions_SVD)
{
    Mat A = (Mat_<double>(3,4) << 1, 2, -1, 4, 2, 4, 3, 5, -1, -2, 6, 7);
    Mat x;
    SVD::solveZ(A,x);
    EXPECT_GE(FLT_EPSILON, cvtest::norm(A*x, CV_C));

    SVD svd(A, SVD::FULL_UV);
    EXPECT_GE(FLT_EPSILON, cvtest::norm(A*svd.vt.row(3).t(), CV_C));

    Mat Dp(3,3,CV_32FC1);
    Mat Dc(3,3,CV_32FC1);
    Mat Q(3,3,CV_32FC1);
    Mat U,Vt,R,T,W;

    Dp.at<float>(0,0)=0.86483884f; Dp.at<float>(0,1)= -0.3077251f; Dp.at<float>(0,2)=-0.55711365f;
    Dp.at<float>(1,0)=0.49294353f; Dp.at<float>(1,1)=-0.24209651f; Dp.at<float>(1,2)=-0.25084701f;
    Dp.at<float>(2,0)=0;           Dp.at<float>(2,1)=0;            Dp.at<float>(2,2)=0;

    Dc.at<float>(0,0)=0.75632739f; Dc.at<float>(0,1)= -0.38859656f; Dc.at<float>(0,2)=-0.36773083f;
    Dc.at<float>(1,0)=0.9699229f;  Dc.at<float>(1,1)=-0.49858192f;  Dc.at<float>(1,2)=-0.47134098f;
    Dc.at<float>(2,0)=0.10566688f; Dc.at<float>(2,1)=-0.060333252f; Dc.at<float>(2,2)=-0.045333147f;

    Q=Dp*Dc.t();
    SVD decomp;
    decomp=SVD(Q);
    U=decomp.u;
    Vt=decomp.vt;
    W=decomp.w;
    Mat I = Mat::eye(3, 3, CV_32F);

    EXPECT_LE(0, W.at<float>(2));
    EXPECT_LE(W.at<float>(2), W.at<float>(1));
    EXPECT_LE(W.at<float>(1), W.at<float>(0));
    EXPECT_GE(FLT_EPSILON, cvtest::norm(U*U.t(), I, CV_C));
    EXPECT_GE(FLT_EPSILON, cvtest::norm(Vt*Vt.t(), I, CV_C));
    EXPECT_GE(FLT_EPSILON, cvtest::norm(U*Mat::diag(W)*Vt, Q, CV_C));
}


class CV_SparseMatTest : public cvtest::BaseTest
{
public:
    CV_SparseMatTest() {}
    ~CV_SparseMatTest() {}
protected:
    void run(int)
    {
        try
        {
            RNG& rng = theRNG();
            const int MAX_DIM=3;
            int sizes[MAX_DIM], idx[MAX_DIM];
            for( int iter = 0; iter < 100; iter++ )
            {
                ts->printf(cvtest::TS::LOG, ".");
                ts->update_context(this, iter, true);
                int k, dims = rng.uniform(1, MAX_DIM+1), p = 1;
                for( k = 0; k < dims; k++ )
                {
                    sizes[k] = rng.uniform(1, 30);
                    p *= sizes[k];
                }
                int j, nz = rng.uniform(0, (p+2)/2), nz0 = 0;
                SparseMat_<int> v(dims,sizes);

                CV_Assert( (int)v.nzcount() == 0 );

                SparseMatIterator_<int> it = v.begin();
                SparseMatIterator_<int> it_end = v.end();

                for( k = 0; it != it_end; ++it, ++k )
                    ;
                CV_Assert( k == 0 );

                int sum0 = 0, sum = 0;
                for( j = 0; j < nz; j++ )
                {
                    int val = rng.uniform(1, 100);
                    for( k = 0; k < dims; k++ )
                        idx[k] = rng.uniform(0, sizes[k]);
                    if( dims == 1 )
                    {
                        CV_Assert( v.ref(idx[0]) == v(idx[0]) );
                    }
                    else if( dims == 2 )
                    {
                        CV_Assert( v.ref(idx[0], idx[1]) == v(idx[0], idx[1]) );
                    }
                    else if( dims == 3 )
                    {
                        CV_Assert( v.ref(idx[0], idx[1], idx[2]) == v(idx[0], idx[1], idx[2]) );
                    }
                    CV_Assert( v.ref(idx) == v(idx) );
                    v.ref(idx) += val;
                    if( v(idx) == val )
                        nz0++;
                    sum0 += val;
                }

                CV_Assert( (int)v.nzcount() == nz0 );

                it = v.begin();
                it_end = v.end();

                for( k = 0; it != it_end; ++it, ++k )
                    sum += *it;
                CV_Assert( k == nz0 && sum == sum0 );

                v.clear();
                CV_Assert( (int)v.nzcount() == 0 );

                it = v.begin();
                it_end = v.end();

                for( k = 0; it != it_end; ++it, ++k )
                    ;
                CV_Assert( k == 0 );
            }
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        }
    }
};

TEST(Core_SparseMat, iterations) { CV_SparseMatTest test; test.safe_run(); }
