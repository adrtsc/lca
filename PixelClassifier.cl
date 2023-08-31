/*
OpenCL RandomForestClassifier
classifier_class_name = PixelClassifier
feature_specification = original gaussian_blur=1 sobel_of_gaussian_blur=1
num_ground_truth_dimensions = 3
num_classes = 2
num_features = 3
max_depth = 2
num_trees = 10
apoc_version = 0.6.5
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
if(i2<387.5731201171875){
 if(i0<32.5){
  s0+=41485.0;
  s1+=239.0;
 } else {
  s0+=100.0;
  s1+=229.0;
 }
} else {
 if(i0<23.5){
  s0+=607.0;
  s1+=340.0;
 } else {
  s0+=119.0;
  s1+=5604.0;
 }
}
if(i0<29.5){
 if(i2<356.3924560546875){
  s0+=41272.0;
  s1+=140.0;
 } else {
  s0+=923.0;
  s1+=686.0;
 }
} else {
 if(i1<21.955791473388672){
  s0+=231.0;
  s1+=13.0;
 } else {
  s0+=19.0;
  s1+=5439.0;
 }
}
if(i0<28.5){
 if(i2<355.92401123046875){
  s0+=41023.0;
  s1+=127.0;
 } else {
  s0+=924.0;
  s1+=655.0;
 }
} else {
 if(i1<22.442916870117188){
  s0+=288.0;
  s1+=17.0;
 } else {
  s0+=27.0;
  s1+=5662.0;
 }
}
if(i1<22.442916870117188){
 if(i2<262.69891357421875){
  s0+=40134.0;
  s1+=11.0;
 } else {
  s0+=2157.0;
  s1+=106.0;
 }
} else {
 if(i2<540.1542358398438){
  s0+=66.0;
  s1+=1805.0;
 } else {
  s0+=22.0;
  s1+=4422.0;
 }
}
if(i0<28.5){
 if(i2<408.89129638671875){
  s0+=41386.0;
  s1+=208.0;
 } else {
  s0+=580.0;
  s1+=499.0;
 }
} else {
 if(i0<32.5){
  s0+=190.0;
  s1+=240.0;
 } else {
  s0+=136.0;
  s1+=5484.0;
 }
}
if(i1<22.75994873046875){
 if(i2<271.047607421875){
  s0+=40288.0;
  s1+=21.0;
 } else {
  s0+=2053.0;
  s1+=119.0;
 }
} else {
 if(i1<25.453109741210938){
  s0+=54.0;
  s1+=113.0;
 } else {
  s0+=19.0;
  s1+=6056.0;
 }
}
if(i2<372.24688720703125){
 if(i1<21.218868255615234){
  s0+=41362.0;
  s1+=66.0;
 } else {
  s0+=22.0;
  s1+=303.0;
 }
} else {
 if(i1<24.93709945678711){
  s0+=829.0;
  s1+=74.0;
 } else {
  s0+=36.0;
  s1+=6031.0;
 }
}
if(i1<22.730897903442383){
 if(i1<19.137611389160156){
  s0+=41944.0;
  s1+=42.0;
 } else {
  s0+=262.0;
  s1+=105.0;
 }
} else {
 if(i1<25.520191192626953){
  s0+=55.0;
  s1+=102.0;
 } else {
  s0+=24.0;
  s1+=6189.0;
 }
}
if(i1<22.70690155029297){
 if(i2<267.01593017578125){
  s0+=40328.0;
  s1+=19.0;
 } else {
  s0+=2064.0;
  s1+=114.0;
 }
} else {
 if(i2<538.1828002929688){
  s0+=56.0;
  s1+=1780.0;
 } else {
  s0+=17.0;
  s1+=4345.0;
 }
}
if(i0<28.5){
 if(i0<20.5){
  s0+=40653.0;
  s1+=338.0;
 } else {
  s0+=1329.0;
  s1+=401.0;
 }
} else {
 if(i0<32.5){
  s0+=185.0;
  s1+=264.0;
 } else {
  s0+=125.0;
  s1+=5428.0;
 }
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
