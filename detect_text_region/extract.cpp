#include "opencv2/opencv.hpp"
#include <iostream>
#include <cstdlib>
#include <dirent.h>

std::vector<cv::Rect> detectLetters(cv::Mat img)
{
  std::vector<cv::Rect> boundRect;
  cv::Mat img_gray, img_sobel, img_threshold, element;
  // グレースケールに変換
  cv::cvtColor(img, img_gray, CV_BGR2GRAY);

  // エッジ検出
  // x 方向に1次微分
  // Sobel(const Mat& src, Mat& dst, int ddepth, int xorder, int yorder, int ksize=3, double scale=1, double delta=0, int borderType=BORDER_DEFAULT)
  cv::Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  //  cv::imshow("imgSub", img_sobel);
  //  cv::waitKey();

  // 閾値処理, 2値化
  // threshold(const Mat& src, Mat& dst, double thresh, double maxVal, int thresholdType)
  // 0 より大きい部分は 255, そうでない部分は 0 => ではない。CV_THRESH_OTSU が優先され、threshold は無効らしい
  // 大津？
  cv::threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
  //  cv::imshow("imgSub", img_threshold);
  //  cv::waitKey();

  // getStructuringElement(int shape, Size esize, Point anchor=Point(-1, -1))
  // 領域を白く塗りつぶす
  element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(18, 18) );
  cv::morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element); //Does the trick
  //  cv::imshow("imgSub", img_threshold);
  //  cv::waitKey();

  // 輪郭抽出
  std::vector< std::vector< cv::Point> > contours;
  cv::findContours(img_threshold, contours, 0, 1);
  
  // Approximate contours to polygons + get bounding rects and circles
  std::vector< std::vector<cv::Point> > contours_poly( contours.size() );
  for( int i = 0; i < contours.size(); i++ )
    if (contours[i].size()>100)
      {
        cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
        cv::Rect appRect( boundingRect( cv::Mat(contours_poly[i]) ));
        if (appRect.width>appRect.height)
          boundRect.push_back(appRect);
      }
  return boundRect;
}

std::vector<std::string> readdir(char* path) {
  //  const char* path = &dirs;
  DIR *dp;       // ディレクトリへのポインタ
  dirent* entry; // readdir() で返されるエントリーポイント
  std::vector<std::string> dir;
  int i = 0;
  dp = opendir(path);
  if (dp==NULL) {
    std::cout << "no";
    exit(1);
  }
  do {
    entry = readdir(dp);
    if (entry != NULL)
      dir.push_back(entry->d_name);
    i++;
  } while (entry != NULL);
  dir.erase(dir.begin(), dir.begin() + 2);
  return dir;
}


int main(int argc, char** argv)
{
  /*
    ディレクトリ中の画像ファイルそれぞれについて文字領域抽出し、
    抽出された領域でトリミングした画像を連番をつけて保存する。
   */
  int i = 0;
  char input_dir[] = "./data/";
  char header[] = "susphi_";
  char output[30];
  std::vector<std::string> files = readdir(input_dir);

  for (std::string filename: files) {
    std::cout << filename << std::endl;
    //Read
    cv::Mat img=cv::imread(input_dir + filename);
    //Detect
    std::vector<cv::Rect> letterBBoxes=detectLetters(img);
    for(int j=0; j< letterBBoxes.size(); j++){
        sprintf(output, "output/%s%03d_%04d.png", header, i, j);
        std::cout << output << std::endl;
        cv::Mat imgSub(img, letterBBoxes[j]);
        cv::imwrite(output, imgSub);
    }
    i++;
  }
  
  return 0;
}
