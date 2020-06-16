#include <GL/freeglut.h>
// #include <freeglut.h>
#include "GL/gl.h"
//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
//#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/filter.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/registration/transforms.h>
//#include <pcl/registration/ndt.h>
#include <pcl/console/parse.h>
//#include <pcl/registration/icp.h>
//#include <pcl/common/time.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/PCLPointCloud2.h>

#include "cudaWrapper.h"
//设定窗口
const unsigned int window_width  = 512;
const unsigned int window_height = 512;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -20.0;
float translate_x, translate_y = 0.0;


pcl::PointCloud<pcl::PointXYZ> point_cloud;
CCudaWrapper cudaWrapper;
// cuda 计算是以wrapper为单位， 每个wrapper有32个threads

bool initGL(int *argc, char **argv);
void display(); //声明void 变量
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void motion(int x, int y);
void printHelp();


int main(int argc, char **argv)
//        argc和argv参数在用命令行编译程序时有用。main( int argc, char* argv[], char **env ) 中
//第一个参数，int型的argc，为整型，用来统计程序运行时发送给main函数的命令行参数的个数，在VS中默认值为1
// 第二个参数，char*型的argv[]，为字符串数组，用来存放指向的字符串参数的指针数组，每一个元素指向一个参数。各成员含义如下：
//        argv[0]指向程序运行的全路径名
//        argv[1]指向在DOS命令行中执行程序名后的第一个字符串
//        argv[2]指向执行程序名后的第二个字符串
//        argv[3]指向执行程序名后的第三个字符串
//        argv[argc]为NULL
//        第三个参数，char**型的env，为字符串数组。env[]的每一个元素都包含ENVVAR=value形式的字符串，其中ENVVAR为环境变量，value为其对应的值。平时使用到的比较少
{
	if(argc < 2)
	{
		std::cout << "Usage:\n";
		// 设定文件路径
		std::cout << argv[0] <<" point_cloud_file.pcd\n";
		std::cout << "Default:  ../../data/scan_Velodyne_VLP16.pcd\n";
        // 如果加载失败/成功
		if(pcl::io::loadPCDFile("../../data/scan_Velodyne_VLP16.pcd", point_cloud) == -1)
		{
			return -1;
		}
	}else
	{
		std::vector<int> ind_pcd; // 创建初始化向量
		ind_pcd = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
		// pcl程序中经常用到程序后面带选项，选项解析使用pcl::console::parse_argument()来完成。 如果文件不是pcd文件，不继续进行

        if(ind_pcd.size()!=1)
		{
			std::cout << "did you forget pcd file location? return" << std::endl;
			return -1;
		}

		if(pcl::io::loadPCDFile(argv[1], point_cloud) == -1)
		{
			return -1;
		}
	}


	if (false == initGL(&argc, argv)) // 如果初始化opengl失败，结束
	{
		return -1;
	}

	printHelp();

	cudaWrapper.warmUpGPU(); //分配gpu 位置？？

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
    glutReshapeFunc(reshape);
	glutMainLoop();
}

bool initGL(int *argc, char **argv) //初始化opengl
{
    glutInit(argc, argv);
    glutDisplayFunc(display);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Lesson 0 - basic transformations"); //定义窗口名称
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, window_width, window_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.01, 10000.0);

    return true;
}

void reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)width / (GLfloat) height, 0.01, 10000.0);
}

void display() // 没有输入
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 0.0, 1.0);

    glBegin(GL_LINES); //开始绘制线段 每个顶点定义为一个独立的线段 GL_LINES：   把每个顶点作为一个独立的线段，顶点2n-1和2n之间定义了n条线段，绘制N/2条线段

    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f); // 绘制一个三维点
    glVertex3f(1.0f, 0.0f, 0.0f);

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glEnd();

    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_POINTS); // GL_POINTS：把每个顶点作为一个点进行处理，顶点n定义了点n，绘制N个点。

    for(size_t i = 0; i < point_cloud.size(); i++)
    	{
    		glVertex3f(point_cloud[i].x, point_cloud[i].y, point_cloud[i].z); // 绘制点云文件的点集
    	}
    glEnd();

    glutSwapBuffers();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            glutDestroyWindow(glutGetWindow());
            return;
        case 'a' :
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

			cudaWrapper.rotateLeft(point_cloud); // 旋转点云文件

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.rotateLeft computation_time: " << computation_time << std::endl;
        	break;
        }
        case 'd' :
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

        	cudaWrapper.rotateRight(point_cloud);

        	computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
        	std::cout << "cudaWrapper.rotateRight computation_time: " << computation_time << std::endl;
          	break;
        }
        case 'w':
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

        	cudaWrapper.translateForward(point_cloud); // 向右边移动点云文件

        	computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.translateForward computation_time: " << computation_time << std::endl;
           	break;
        }
        case 's':
		{
			clock_t begin_time;
			double computation_time;
			begin_time = clock();

			cudaWrapper.translateBackward(point_cloud);

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.translateBackward computation_time: " << computation_time << std::endl;
			break;
		}
        case 'r':
		{
			clock_t begin_time;
			double computation_time;
			begin_time = clock();

			cudaWrapper.removePointsInsideSphere(point_cloud);

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.removePointsInsideSphere computation_time: " << computation_time << std::endl;
			break;
		}
    }
    glutPostRedisplay();
    printHelp();
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;

    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.05f;
    }
    else if (mouse_buttons & 3)
    {
            translate_x += dx * 0.05f;
            translate_y -= dy * 0.05f;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    glutPostRedisplay();
}

void printHelp()
{
	std::cout << "----------------------" << std::endl;
	std::cout << "press 'a': rotate Left (10 degree)" << std::endl;
	std::cout << "press 'd': rotate Right (10 degree)" << std::endl;
	std::cout << "press 'w': translate Forward (1 meter)" << std::endl;
	std::cout << "press 's': translate Backward (1 meter)" << std::endl;
	std::cout << "press 'r': remove points inside sphere (radius == 1.0)" << std::endl;
	std::cout << "press 'Esc': EXIT" << std::endl;
}
