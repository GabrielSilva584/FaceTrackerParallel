///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010, Jason Mora Saragih, all rights reserved.
//
// This file is part of FaceTracker.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions (licence) and the following disclaimer
//       in the documentation and/or other materials provided with the
//       distribution.
//     * The name of the author may not be used to endorse or promote products
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite the following work:
//
//       J. M. Saragih, S. Lucey, and J. F. Cohn. Face Alignment through
//       Subspace Constrained Mean-Shifts. International Conference of Computer
//       Vision (ICCV), September, 2009.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////
#include <FaceTracker/Tracker.h>
#include <opencv/highgui.h>
#include <iostream>

//OpenMP
#include <omp.h>

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if defined(_WIN32)
#include <Windows.h>
#include <time.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>
#include <sys/times.h>
#include <time.h>

#else
#error "Unable to define getCPUTime( ) for an unknown OS."
#endif

/**
 * Returns the amount of CPU time used by the current process,
 * in seconds, or -1.0 if an error occurred.
 */
double getCPUTime( )
{
#if defined(_WIN32)
	/* Windows -------------------------------------------------- */
	FILETIME createTime;
	FILETIME exitTime;
	FILETIME kernelTime;
	FILETIME userTime;
	if ( GetProcessTimes( GetCurrentProcess( ),
		&createTime, &exitTime, &kernelTime, &userTime ) != -1 )
	{
		SYSTEMTIME userSystemTime;
		if ( FileTimeToSystemTime( &userTime, &userSystemTime ) != -1 )
			return (double)userSystemTime.wHour * 3600.0 +
				(double)userSystemTime.wMinute * 60.0 +
				(double)userSystemTime.wSecond +
				(double)userSystemTime.wMilliseconds / 1000.0;
	}

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
	/* AIX, BSD, Cygwin, HP-UX, Linux, OSX, and Solaris --------- */

#if defined(_POSIX_TIMERS) && (_POSIX_TIMERS > 0)
	/* Prefer high-res POSIX timers, when available. */
	{
		clockid_t id;
		struct timespec ts;
#if _POSIX_CPUTIME > 0
		/* Clock ids vary by OS.  Query the id, if possible. */
		if ( clock_getcpuclockid( 0, &id ) == -1 )
#endif
#if defined(CLOCK_PROCESS_CPUTIME_ID)
			/* Use known clock id for AIX, Linux, or Solaris. */
			id = CLOCK_PROCESS_CPUTIME_ID;
#elif defined(CLOCK_VIRTUAL)
			/* Use known clock id for BSD or HP-UX. */
			id = CLOCK_VIRTUAL;
#else
			id = (clockid_t)-1;
#endif
		if ( id != (clockid_t)-1 && clock_gettime( id, &ts ) != -1 )
			return (double)ts.tv_sec +
				(double)ts.tv_nsec / 1000000000.0;
	}
#endif

#if defined(RUSAGE_SELF)
	{
		struct rusage rusage;
		if ( getrusage( RUSAGE_SELF, &rusage ) != -1 )
			return (double)rusage.ru_utime.tv_sec +
				(double)rusage.ru_utime.tv_usec / 1000000.0;
	}
#endif

#if defined(_SC_CLK_TCK)
	{
		const double ticks = (double)sysconf( _SC_CLK_TCK );
		struct tms tms;
		if ( times( &tms ) != (clock_t)-1 )
			return (double)tms.tms_utime / ticks;
	}
#endif

#if defined(CLOCKS_PER_SEC)
	{
		clock_t cl = clock( );
		if ( cl != (clock_t)-1 )
			return (double)cl / (double)CLOCKS_PER_SEC;
	}
#endif

#endif

	return -1;		/* Failed. */
}

void printTime(struct timespec start){
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t delta_ns = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000);
    printf("Execution time = %lu ns\n", delta_ns);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//=============================================================================
void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi){
    int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        #ifdef _OPENMP
        #pragma omp single
        #endif
        {
            //draw lines (point 0 to 1)
            for(i = 0; i < tri.rows; i++){
                if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
                    visi.at<int>(tri.at<int>(i,1),0) == 0 ||
                    visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
                p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
                    shape.at<double>(tri.at<int>(i,0)+n,0));
                p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
                    shape.at<double>(tri.at<int>(i,1)+n,0));
                cv::line(image,p1,p2,c);
            }
        }
        #ifdef _OPENMP
        #pragma omp single
        #endif
        {
            //draw lines (point 0 to 2)
            for(i = 0; i < tri.rows; i++){
                if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
                    visi.at<int>(tri.at<int>(i,1),0) == 0 ||
                    visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
                p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
                    shape.at<double>(tri.at<int>(i,0)+n,0));
                p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
                    shape.at<double>(tri.at<int>(i,2)+n,0));
                cv::line(image,p1,p2,c);
            }
        }
        #ifdef _OPENMP
        #pragma omp single
        #endif
        {
            //draw lines (point 1 to 2)
            for(i = 0; i < tri.rows; i++){
                if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
                    visi.at<int>(tri.at<int>(i,1),0) == 0 ||
                    visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
                p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
                    shape.at<double>(tri.at<int>(i,2)+n,0));
                p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
                    shape.at<double>(tri.at<int>(i,1)+n,0));
                cv::line(image,p1,p2,c);
            }
        }
        #ifdef _OPENMP
        #pragma omp single
        #endif
        {
            //draw connections
            c = CV_RGB(0,0,255);
            for(i = 0; i < con.cols; i++){
                if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
                    visi.at<int>(con.at<int>(1,i),0) == 0)continue;
                p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
                    shape.at<double>(con.at<int>(0,i)+n,0));
                p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
                    shape.at<double>(con.at<int>(1,i)+n,0));
                cv::line(image,p1,p2,c,1);
            }
        }
        #ifdef _OPENMP
        #pragma omp single
        #endif
        {
            //draw points
            for(i = 0; i < n; i++){
                if(visi.at<int>(i,0) == 0)continue;
                p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
                c = CV_RGB(255,0,0); cv::circle(image,p1,2,c);
            }
        }
    }return;
}
//=============================================================================
int parse_cmd(int argc, const char** argv,
	      char* ftFile,char* conFile,char* triFile,
	      bool &fcheck,double &scale,int &fpd){
    int i; fcheck = false; scale = 1; fpd = -1;
    for(i = 1; i < argc; i++){
        if((std::strcmp(argv[i],"-?") == 0) ||
        (std::strcmp(argv[i],"--help") == 0)){
        std::cout << "track_face:- Written by Jason Saragih 2010" << std::endl
        << "Performs automatic face tracking" << std::endl << std::endl
        << "#" << std::endl
        << "# usage: ./face_tracker [options]" << std::endl
        << "#" << std::endl << std::endl
        << "Arguments:" << std::endl
        << "-m <string> -> Tracker model (default: ../model/face2.tracker)"
        << std::endl
        << "-c <string> -> Connectivity (default: ../model/face.con)"
        << std::endl
        << "-t <string> -> Triangulation (default: ../model/face.tri)"
        << std::endl
        << "-s <double> -> Image scaling (default: 1)" << std::endl
        << "-d <int>    -> Frames/detections (default: -1)" << std::endl
        << "--check     -> Check for failure" << std::endl;
        return -1;
        }
    }
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"--check") == 0){fcheck = true; break;}
    }
    if(i >= argc)fcheck = false;
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-s") == 0){
        if(argc > i+1)scale = std::atof(argv[i+1]); else scale = 1;
        break;
        }
    }
    if(i >= argc)scale = 1;
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-d") == 0){
        if(argc > i+1)fpd = std::atoi(argv[i+1]); else fpd = -1;
        break;
        }
    }
    if(i >= argc)fpd = -1;
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-m") == 0){
        if(argc > i+1)std::strcpy(ftFile,argv[i+1]);
        else strcpy(ftFile,"../model/face2.tracker");
        break;
        }
    }
    if(i >= argc)std::strcpy(ftFile,"../model/face2.tracker");
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-c") == 0){
        if(argc > i+1)std::strcpy(conFile,argv[i+1]);
        else strcpy(conFile,"../model/face.con");
        break;
        }
    }
    if(i >= argc)std::strcpy(conFile,"../model/face.con");
    for(i = 1; i < argc; i++){
        if(std::strcmp(argv[i],"-t") == 0){
        if(argc > i+1)std::strcpy(triFile,argv[i+1]);
        else strcpy(triFile,"../model/face.tri");
        break;
        }
    }
    if(i >= argc)std::strcpy(triFile,"../model/face.tri");
    return 0;
}
//=============================================================================
int main(int argc, const char** argv)
{
  //parse command line arguments
  char ftFile[256],conFile[256],triFile[256];
  bool fcheck = false;
  double scale = 1;
  int fpd = -1;
  bool show = true;
  if(parse_cmd(argc,argv,ftFile,conFile,triFile,fcheck,scale,fpd)<0){
    return 0;
  }

  double start = getCPUTime();
  struct timespec startT;
	clock_gettime(CLOCK_MONOTONIC_RAW, &startT);

  //set other tracking parameters
  std::vector<int> wSize1(1);
  wSize1[0] = 7;
  std::vector<int> wSize2(3);
  wSize2[0] = 11;
  wSize2[1] = 9;
  wSize2[2] = 7;
  int nIter = 5;
  double clamp=3,fTol=0.01;
  FACETRACKER::Tracker model(ftFile);
  cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
  cv::Mat con=FACETRACKER::IO::LoadCon(conFile);

  //initialize camera and display window
  cv::Mat frame,gray,im;
  double fps=0;
  char sss[256];
  std::string text;
  cv::VideoCapture camera(CV_CAP_ANY);
  if(!camera.isOpened()){
    return -1;
  }
  int64 t1,t0 = cvGetTickCount();
  int fnum=0;
  cvNamedWindow("Face Tracker",1);
  std::cout << "Hot keys: "        << std::endl
	    << "\t ESC - quit"     << std::endl
	    << "\t d   - Redetect" << std::endl;

  //loop until quit (i.e user presses ESC)
  bool failed = true;
  int frameskp = 0;
  int redetectskp = 5;
  int cycles = 170;

  while(1){
    if(frameskp==0){
      //grab image, resize and flip
      camera.read(frame);
      if(scale == 1){
        im = frame;
      }else{
        cv::resize(frame,im,cv::Size(scale*frame.cols,scale*frame.rows));
      }
      cv::flip(im,im,1);
      cv::cvtColor(im,gray,CV_BGR2GRAY);

      //track this image
      std::vector<int> wSize;
      if(failed){
        wSize = wSize2;
        for(int skp = 0; skp < redetectskp; skp++){
          camera.read(frame);
        }
      }else{
        wSize = wSize1;
      }
      if(model.Track(gray,wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
        int idx = model._clm.GetViewIdx();
        failed = false;
        Draw(im,model._shape,con,tri,model._clm._visi[idx]);
      }else{
        if(show){
          cv::Mat R(im,cvRect(0,0,150,50));
          R = cv::Scalar(0,0,255);
        }
        model.FrameReset();
        failed = true;
      }

      //draw framerate on display image
      if(fnum >= 9){
        t1 = cvGetTickCount();
        fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6);
        t0 = t1;
        fnum = 0;
      }else{
        fnum += 1;
      }
      if(show){
        sprintf(sss,"%d frames/sec",(int)round(fps));
        text = sss;
        cv::putText(im,text,cv::Point(10,20),
            CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
      }

      //show image and check for user input
      imshow("Face Tracker",im);
      int c = cv::waitKey(10);
      if(c == 27){
        break;
      }else if(char(c) == 'd'){
        model.FrameReset();
      }
      frameskp = 0;

      //Get and show CPU usage time
      if(cycles>0){
        cycles--;
        if(cycles==0){
          double end = getCPUTime();
          printf("CPU time used = %lf secs\n", (end - start) );
          printTime(startT);
          return 0;
        }
      }

    }else{
        camera.read(frame);
        frameskp--;
    }
  }return 0;
}
//=============================================================================
