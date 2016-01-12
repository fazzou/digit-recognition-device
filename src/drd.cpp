
#include "bitmap.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <GL/glut.h>
#include <math.h>
#include "LogisticRegression.h"
//#include "DBN.h"

#define RESIZED_WIDTH 28
#define RESIZED_HEIGHT 28
#define NR_OF_LABELS 10


class ResizedDigit {
  private:
    int digit[RESIZED_WIDTH][RESIZED_HEIGHT];

  public: 
    void setDigitPixel(int x, int y, int pixelValue){
        this->digit[x][y]=pixelValue;
    }
    float getValueOfPixel(int x, int y){
        return (float) this->digit[x][y];
    }
    int * getPixelArray(){
        return (int *) this->digit;
    }

};


int        Width = 640;       /* Width of window */
int        Height = 480;      /* Height of window */
// BITMAPINFO *BitmapInfo; /* Bitmap information */
GLubyte    *BitmapBits; /* Bitmap data */
float      *workingBmp;
float      *workingBg;
char* buffer;
int camfd;
//DBN dbn;
LogisticRegression logisticRegression;
double results[NR_OF_LABELS];

void Redraw(void);
void Resize(int width, int height);
ResizedDigit resizeDigit(int massCenterX, int massCenterY, int width, int height, int newWidth, int newHeight);
void drawResizedDigit(ResizedDigit digit, int digitNumber, int width, int height);
int capture();
int findAllInSet(float radius, int x, int y, int* data);


static int xioctl(int fd, int request, void *arg)
{
        int r;

        do r = ioctl (fd, request, arg);
        while (-1 == r && EINTR == errno);

        return r;
}

int print_caps(int fd)
{
        struct v4l2_capability caps = {};
        if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &caps))
        {
                perror("Querying Capabilities");
                return 1;
        }

        struct v4l2_cropcap cropcap = {0};
        cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl (fd, VIDIOC_CROPCAP, &cropcap))
        {
                perror("Querying Cropping Capabilities");
                return 1;
        }

        int support_grbg10 = 0;

        struct v4l2_fmtdesc fmtdesc = {0};
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        char fourcc[5] = {0};
        while (0 == xioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc))
        {
                strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
                if (fmtdesc.pixelformat == V4L2_PIX_FMT_YUYV)
                    support_grbg10 = 1;
                
                fmtdesc.index++;
        }

        if (!support_grbg10)
        {
            printf("Doesn't support GRBG10.\n");
            return 1;
        }

        struct v4l2_format fmt = {0};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = Width;
        fmt.fmt.pix.height = Height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
        {
            perror("Setting Pixel Format");
            return 1;
        }

        strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
        return 0;
}

int init_mmap(int fd)
{
    struct v4l2_requestbuffers req;
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req))
    {
        perror("Requesting Buffer");
        return 1;
    }

    struct v4l2_buffer buf;
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if(-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
    {
        perror("Querying Buffer");
        return 1;
    }
    buffer = mmap (NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

    if (buffer == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    return 0;
}

int capture_image(int fd)
{
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if(-1 == xioctl(fd, VIDIOC_QBUF, &buf))
    {
        perror("Query Buffer");
        return 1;
    }

    if(-1 == xioctl(fd, VIDIOC_STREAMON, &buf.type))
    {
        perror("Start Capture");
        return 1;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    struct timeval tv = {0};
    tv.tv_sec = 2;
    int r = select(fd+1, &fds, NULL, NULL, &tv);
    if(-1 == r)
    {
        perror("Waiting for Frame");
        return 1;
    }

    if(-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
    {
        perror("Retrieving Frame");
        return 1;
    }


    return 0;
}

int getImg()
{
        camfd = open("/dev/video1", O_RDWR);
        if (camfd == -1)
        {
                perror("Opening video device");
                return 1;
        }

        if(print_caps(camfd))
            return 1;

        if(init_mmap(camfd))
            return 1;

        capture();

    return 0;
}

//setter and getter for bits of shown bitmap (operates on r, g and b)
void setBit(int x, int y, GLubyte val, GLubyte* bmp) {
    if (x >= Width || x < 0) return;
    if (y >= Height || y < 0) return;
    bmp[(Width*Height - y*Width - Width + x)*3 + 1] = val;
    bmp[(Width*Height - y*Width - Width + x)*3 + 2] = val;
    bmp[(Width*Height - y*Width - Width + x)*3 + 3] = val;
}

GLubyte getBit(int x, int y, GLubyte* bmp) {
    if (x >= Width || x < 0) return 0;
    if (y >= Height || y < 0) return 0;
    return bmp[(Width*Height - y*Width - Width + x)*3 + 1];
}

void setBitGreen(int x, int y, GLubyte* bmp) {
    if (x >= Width || x < 0) return;
    if (y >= Height || y < 0) return;
    bmp[(Width*Height - y*Width - Width + x)*3 + 1] = 255;
    bmp[(Width*Height - y*Width - Width + x)*3 + 2] = 0;
    bmp[(Width*Height - y*Width - Width + x)*3 + 3] = 0;
}

//setter and getter for float representation of grayscale image 
//(just one value from 0 to 255)
void setfBit(int x, int y, float val, float* bmp) {
    if (x >= Width || x < 0) return;
    if (y >= Height || y < 0) return;
    bmp[Width*Height - y*Width - Width + x] = val;
}

float getfBit(int x, int y, float* bmp) {
    if (x >= Width || x < 0) return 0;
    if (y >= Height || y < 0) return 0;
    return bmp[Width*Height - y*Width - Width + x];
}

//same for int
void setiBit(int x, int y, int val, int* bmp) {
    if (x >= Width || x < 0) return;
    if (y >= Height || y < 0) return;
    bmp[Width*Height - y*Width - Width + x] = val;
}

int getiBit(int x, int y, int* bmp) {
    if (x >= Width || x < 0) return 0;
    if (y >= Height || y < 0) return 0;
    return bmp[Width*Height - y*Width - Width + x];
}

//get array of float 0-255 pixels 
//and write them to array of pixels which are going to be shown
void setAsCurrentlyVisible(float* working) {
    int x,y;
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            setBit(x, y, getfBit(x, y, working), BitmapBits);
        }
    }
}

//Fills circle with center in (x,y) with radius r with color, 
//which is avg of all pixel colors in that circle
void blurCircle(int x, int y, int r, float* bmp) {
    float fr = (float)r;
    int a = r*2 + 1;
    float weight = 0;
    float sum = 0;
    int xp, yp;
    //calculate avg
    for (xp = -r; xp < r; xp++) {
        for (yp = -r; yp < r; yp++) {
            float fxp = (float)xp;
            float fyp = (float)yp;
            if (sqrt(fxp*fxp + fyp*fyp) < r) {
                if (getfBit(x + xp, y + yp, bmp) != 0) weight++;
                sum += getfBit(x + xp, y + yp, bmp);
            }
        }
    }
    //fill
    for (xp = -r; xp < r; xp++) {
        for (yp = -r; yp < r; yp++) {
            float fxp = (float)xp;
            float fyp = (float)yp;
            if (sqrt(fxp*fxp + fyp*fyp) < r) {
                setfBit(x + xp, y + yp, sum/weight, bmp);
            }
        }
    }
}

//divide -1's in data into sets 1-n
//belongingness of pixel to specified set depends on
//whether there is any other set within a given radius
//returns n - number of found sets
int markSets(float radius, int* data) {
    int x,y;
    float r = radius;
    int a = r*2 + 1;
    int xp, yp;
    int sets = 0;
    for (x = 0; x<Height; x++) {
        for (y = 0; y<Width; y++) {
            int val = getiBit(x, y, data);
            if (val == -1) {
                for (xp = -r; xp < r; xp++) {
                    for (yp = -r; yp < r; yp++) {
                        int pv = getiBit(x + xp, y + yp, data);
                        if (pv > 0) setiBit(x, y, pv, data);
                    }
                }
                if (getiBit(x, y, data) == -1) { 
                    sets++;
                    setiBit(x, y, sets, data);
                    findAllInSet(radius, x, y, data);
                }
            }
        }
    }
    return sets;
}

int findAllInSet(float radius, int x, int y, int* data) {
	int xp, yp;
	int setNo = getiBit(x, y, data);
	for (xp = -radius; xp < radius; xp++) {
		for (yp = -radius; yp < radius; yp++) {
			if (getiBit(x + xp, y + yp, data) == -1) {
				setiBit(x + xp, y + yp, setNo, data);
				findAllInSet(radius, x + xp, y + yp, data);
				
			}
		}
	}
}

//returns
//[massCenterX, massCenterY, width, height]
int* setParams(int setNo, int* data) {
    int x,y;
    int minx = Width, miny = Height, maxx = 0, maxy = 0;
    int* res = malloc(4*sizeof(int));
    float xval = 0, yval = 0, wg = 0;
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            if (getiBit(x, y, data) == setNo) {
                wg++;
                xval += (float)x;
                yval += (float)y;
                if (x > maxx) maxx = x;
                if (x < minx) minx = x;
                if (y > maxy) maxy = y;
                if (y < miny) miny = y;
            }
        }
    }
    res[0] = (int)(xval/wg);
    res[1] = (int)(yval/wg);
    res[2] = maxx - minx;
    res[3] = maxy - miny;
    return res;
}

void drawGreenRectangle(int xStart, int yStart, int width, int height) {
    int x,y;
    for (x = xStart; x<=xStart+width; x+=width) {
        for (y = yStart; y<=yStart+height; y++) {
            setBitGreen(x, y, BitmapBits);
        }
    }
    for (y = yStart; y<=yStart+height; y+=height) {
        for (x = xStart; x<=xStart+width; x++) {
            setBitGreen(x, y, BitmapBits);
        }
    }
}

void processFrame() {
    int i=0;
    int x, y;
    //load image to float array from webcam
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            char greyval = buffer[i*2] & 0xff;
            int igreyval = ((int)greyval & 0xff);
            i++;
            float fgreyval = (float)igreyval;
            setfBit(x, y, fgreyval, workingBmp);
        }
    }

    //invert image
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            float val = getfBit(x, y, workingBmp);
            val = 255 - val;
            setfBit(x, y, val, workingBmp);
        }
    }
    //write loaded image to bg working image also
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            setfBit(x, y, getfBit(x, y, workingBmp), workingBg);
        }
    }
    //blur background (to get basic light distribution on image)
    int rad = 30;
    for (y = 0; y<Height; y+=15) {
        for (x = 0; x<Width; x+=15) {
            blurCircle(x, y, rad, workingBg);
        }
    }

    float max = 0;
    float min = 255;
    //overwrite image with difference between image and background
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            float val = getfBit(x, y, workingBmp) - getfBit(x, y, workingBg);
            if (val < 0) val = 0;
            if (val > max) max = val;
            if (val < min) min = val;
            setfBit(x, y, val, workingBmp);
        }
    }
    //normalize difference(to be distributed between 0-255)
    float factor = 255/(max - min);
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            float val = getfBit(x, y, workingBmp);
            setfBit(x, y, floor((val-min)*factor), workingBmp);
        }
    }
    //invert again
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            float val = getfBit(x, y, workingBmp);
            setfBit(x, y, 255 - val, workingBmp);
        }
    }

    //apply image as currently shown
    // setAsCurrentlyVisible(workingBmp);
    
    //find min and max color
    min = 255;
    max = 0;
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            float val = 255 - getfBit(x, y, workingBmp);
            if (val < min) min = val;
            if (val > max) max = val;
        }
    }


    //create array of ints representing whether digit can be there or not
    //(-1) represents digit
    int* digitsArr;
    digitsArr = (int *) malloc(Width * Height * sizeof(int));
    for (y = 0; y<Height; y++) {
        for (x = 0; x<Width; x++) {
            float bound = (max - min)*0.3;
            float val = 255 - getfBit(x, y, workingBmp);
            if (val > bound) val = 255;
            else val = 0;
            setiBit(x, y, -(int)val / 255, digitsArr);
            setfBit(x, y, val, workingBmp);
        }
    }
    setAsCurrentlyVisible(workingBmp);

    //mark sets
    int setsNo = markSets(20.0, digitsArr);
    //find mass centers and draw rectangles arount them

    //create table for resized digits
    ResizedDigit resizedDigitsArray[setsNo];

    for (i = 1; i<=setsNo; i++) {
        int* center = setParams(i, digitsArr);
        drawGreenRectangle(center[0] - center[2]/2 - 25, center[1] - center[3]/2 - 25, center[2] + 50, center[3] + 50);

        //resize digit and display it on the screen
        resizedDigitsArray[i-1] = resizeDigit(center[0], center[1], (int)(center[2]*1.5), (int)(center[3]*1.5), RESIZED_WIDTH, RESIZED_HEIGHT);
        drawResizedDigit(resizedDigitsArray[i-1], i-1, RESIZED_WIDTH, RESIZED_HEIGHT);

        predict(&logisticRegression,  resizedDigitsArray[i-1].getPixelArray(), results);
        for (int j = 0; j < NR_OF_LABELS; ++j) {
            printf("%f\n", results[j]);
        }
        int result = chooseBest(results);
        printf("%d", result);
        printf("\n");
        
        free(center);

    }
    printf("\n\n");



}

void drawResizedDigit(ResizedDigit digit, int digitNumber, int width, int height) {
    for(int x=0; x<width; x++){
        for (int y = 0; y < height; ++y) {
            setBit(x+digitNumber*width,y, digit.getValueOfPixel(x,y), BitmapBits);
        }
    }
}

ResizedDigit resizeDigit(int massCenterX, int massCenterY,
                         int width, int height, int newWidth, int newHeight) {
    ResizedDigit resizedDigit;
    double x_ratio = width/(double)newWidth;
    double y_ratio = height/(double)newHeight;

    //coordinates of pixel which value should be used as value of pixel in resized image
    double pixelX;
    double pixelY;

    //coordinates of left upper corner of set rectangle
    int leftUpperX = massCenterX - width/2;
    int leftUpperY = massCenterY - height/2;

    //iterate over all the pixels of new image and set proper values
    for(int y=0;y<newHeight;y++){
        for(int x=0;x<newWidth;x++){
            pixelX = floor(x*x_ratio + leftUpperX);
            pixelY = floor(y*y_ratio + leftUpperY);

            //set value of new pixel as value of "closest neighbour" in original image
            resizedDigit.setDigitPixel(x, y, (int)getfBit(pixelX, pixelY, workingBmp));
        }
    }

    return resizedDigit;
}

int capture() {
    if(capture_image(camfd))
        return 1;
    processFrame();
}

//function run when everything is ready
void updateImg(void) {
    int refreshTime = 5000;
    usleep(refreshTime);
    Redraw();
}

int initBitmapStructs() {
    if ((BitmapBits = malloc(Width*Height*3)) == NULL) {
        return 1;
    }
    if ((workingBmp = malloc(Width*Height*sizeof(float))) == NULL) {
        return 1;
    }
    if ((workingBg = malloc(Width*Height*sizeof(float))) == NULL) {
        return 1;
    }
    return 0;
}

void freeBitmapStructs() {
    free(BitmapBits);
    free(workingBmp);
    free(workingBg);
}

/* O - Exit status */
/* I - Number of command-line arguments */
/* I - Command-line arguments */
 int main(int  argc, char *argv[])
 {

     if (initBitmapStructs()) return 1;

     glutInit(&argc, argv);
     glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
     glutInitWindowSize(Width, Height);
     glutCreateWindow("DRD - Digit Recognition Device");
     glutReshapeFunc(Resize);
     glutDisplayFunc(Redraw);
     glutIdleFunc(updateImg);

 //    int train_N = 20000;
 //    int n_ins = 28*28;
 //    int n_outs = 10;
 //    int hidden_layer_sizes[] = {500, 500, 2000};
 //    int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);

 //    loadDBNModel(&dbn, train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);

     int trainingSetSize = 50000;
     int inputDimension = 28 * 28;
     int nrOfClasses = 10;

     buildLogisticRegressionModel(&logisticRegression, trainingSetSize, inputDimension, nrOfClasses);
     loadWeightsLR(&logisticRegression);

     getImg();

     glutMainLoop();

     freeBitmapStructs();

     return 0;
 }

void
Redraw(void) {

    capture();
    GLfloat xsize, ysize;     /* Size of image */
    GLfloat xoffset, yoffset; /* Offset of image */
    GLfloat xscale, yscale;   /* Scaling of image */

    /* Clear the window to black */
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);


    xsize = Width;
    ysize = Height * xsize /
            Width;
    if (ysize > Height)
        {
        ysize = Height;
        xsize = Width * ysize /
                Height;
        }

    xscale  = xsize / Width;
    yscale  = ysize / Height;

    xoffset = (Width - xsize) * 0.5;
    yoffset = (Height - ysize) * 0.5;

    glRasterPos2f(xoffset, yoffset);
    glPixelZoom(xscale, yscale);

    glDrawPixels(Width,
                 Height,
                 GL_BGR_EXT, GL_UNSIGNED_BYTE, BitmapBits);

    glFinish();
    }


void
Resize(int width,  /* I - Width of window */
       int height) /* I - Height of window */
    {
    Width  = width;
    Height = height;

    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, (GLfloat)width, 0.0, (GLfloat)height, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    }

