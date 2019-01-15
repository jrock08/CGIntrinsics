curl -O http://www.cs.cornell.edu/projects/megadepth/dataset/cgintrinsics/intrinsics_final.zip
curl -O http://www.cs.cornell.edu/projects/megadepth/dataset/cgintrinsics/IIW.zip
curl -O http://labelmaterial.s3.amazonaws.com/release/iiw-dataset-json-only-release-0.zip
curl -O http://labelmaterial.s3.amazonaws.com/release/iiw-dataset-release-0.zip
curl -O http://www.cs.cornell.edu/projects/megadepth/dataset/cgintrinsics/SAW.zip

unzip intrinsics_final.zip 
unzip IIW.zip
unzip SAW.zip

git clone git clone https://github.com/kovibalu/saw_release.git
./saw_release/download_saw.sh
mv saw/saw_images_512 SAW/
mv saw/saw_pixel_labels SAW/
rm -r saw
rm -rf saw_release

unzip iiw-dataset-release-0.zip
mv iiw-dataset/data/* IIW/data/
mv iiw-dataset IIW/
rm -r iiw-dataset

mkdir CGIntrinsics
mv IIW CGIntrinsics/
mv intrinsics_final CGIntrinsics/intrinsics_final
mv SAW CGIntrinsics/
