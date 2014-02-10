#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef linux
#include <netinet/in.h>
#endif
#ifdef _WIN32
#include <winsock.h>
#endif



void main()
{
	FILE *fp;
	unsigned char *buffer;
	unsigned char widthBuffer[4];
	unsigned char heightBuffer[4];
	unsigned char *widthString;
	unsigned char *heightString;
	unsigned int width;
	unsigned int height;
	size_t result;
	unsigned long ll = 320;
	fp = fopen("./test3.png", "rb");
	
	buffer = (unsigned char*) malloc (sizeof(unsigned char)*24);
	widthString = (unsigned char *) malloc(sizeof(unsigned char) * 4);
	heightString = (unsigned char *)malloc(sizeof(unsigned char) * 4);
	
	if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}
	result = fread (buffer,1,24,fp);
	int i = 0;
	int j = 0;
    for (i = 16; i < 20; i++)
	{
		widthBuffer[j] = *(buffer+i);
		j++;
	}
	//bufferOffset[4] = 0;
	sprintf(widthString, "%02X%02X%02X%02X", widthBuffer[0], widthBuffer[1], widthBuffer[2], widthBuffer[3]);
	printf("String: %s\n", widthString); 
	width = (int)strtol(widthString, NULL, 16);
	printf("Width: %d\n",width);
	
	j= 0;
	for (i = 20; i < 24; i++)
	{
		heightBuffer[j] = *(buffer+i);
		j++;
	}
	sprintf(heightString, "%02X%02X%02X%02X", heightBuffer[0], heightBuffer[1], heightBuffer[2], heightBuffer[3]);
	printf("String: %s\n", heightString);
	height = (int)strtol(heightString, NULL, 16);
	printf("Height: %d\n", height);
	
}