/*
LodePNG Examples

Copyright (c) 2005-2012 Lode Vandevenne

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

	1. The origin of this software must not be misrepresented; you must not
	claim that you wrote the original software. If you use this software
	in a product, an acknowledgment in the product documentation would be
	appreciated but is not required.

	2. Altered source versions must be plainly marked as such, and must not be
	misrepresented as being the original software.

	3. This notice may not be removed or altered from any source
	distribution.
*/

#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
3 ways to decode a PNG from a file to RGBA pixel data (and 2 in-memory ways).
*/

/*
Example 1
Decode from disk to raw pixels with a single function call
*/
void decodeOneStep(const char* filename)
{
  unsigned error;
  unsigned char* image;
  unsigned width, height;

  error = lodepng_decode24_file(&image, &width, &height, filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
  printf("I get here!\n");
 // while((*image) != '\0')
  //{
//	printf("%d \n", *image);
//	image++;
//  }
  /*use image here*/
  int count = 0;
  int rowsize = height;
  int colsize = width;
  int nlines = 0;
  //  printf("I get here %d", width);
  int *R= (int *)malloc(sizeof(int) * height * width);
  int *G= (int *)malloc(sizeof(int) * height * width);
  int *B= (int *)malloc(sizeof(int) * height * width);
  int it = 0;
  /**
  for (it =0; it < height; it++)
  {
    *(R+it) = (int *)malloc(sizeof(int) * width);
    *(G+it) = (int *)malloc(sizeof(int) * width);
    *(B+it) = (int *)malloc(sizeof(int) * width);
  }
  */
 
  /**
  int (*R)[width] = malloc(height * sizeof(int) * width);
  int (*G)[width] = malloc(height * sizeof(int) * width);
  int (*B)[width] = malloc(height * sizeof(int) * width);
  */
  unsigned char *newImage = malloc(sizeof(unsigned char) * width * height * 3);
 
  //printf("I get here 3");
  int  maxlen = width;
  char str[maxlen], lines[height][maxlen];
	unsigned int h1, h2, h3;
  unsigned char *sptr;
  int row = 0; 
  int col = 0;
  int total = width * height;
  int totalReached =0;
  int i =0, j=0;
  //while (totalReached <= total)
 // {
   // sprintf(str,"%1309c",image);
   // printf("%X\n\n\n",str);
   // if (nlines < height) {strcpy((char *)lines[nlines++],(char *)str);}
   // else
    //{
      printf("Width: %d\n", width);
      printf("Height: %d\n", height);
     // char *first;
    //  char *second;
    //  char *third;
     /**
      for (count = 0;count < (width * height * 3);count+=3)
      {
        //count++;
      //  first = (image + count);
      //  second = (image + count + 1);
      //  third = (image + count + 2);
      
        sscanf((image +count), "%2x",&h1);
        sscanf((image +count + 1),"%2x",&h2);
        sscanf((image +count + 2),"%2x",&h3);
        //sprintf(h1, "%2x", *image);
        //sprintf(h1, "%2x", *(image + 2));
        //sprintf(h1, "%2x", *(image + 4));
        if (col==colsize)
        {
          col = 0;
          row++;
        }
        if (row < rowsize) 
        {
          R[row][col] = h1;
          G[row][col] = h2;
          B[row][col] = h3;
          //printf("%d, %d, %d\n", h1, h2, h3);
          //printf("%2x, %2x, %2x \n", R[row][col], G[row][col], B[row][col]);
        }
        col++;
        
      }
      printf("Rows: %d\n", row);
      printf("Col: %d\n", col);
     */ 
     char *first = malloc(sizeof(char));
     char *second = malloc(sizeof(char));
     char *third = malloc(sizeof(char));
     char *firstPointer = first;
     char *secondPointer = second;
     char *thirdPointer = third;
    for(i=0,j=0;i<3*width*height;i+=3,j++){
      
     sprintf(first, "%02x", *(image + i));
     sprintf(second, "%02x", *(image + i+1));
     sprintf(third, "%02x", *(image + i+2));
    *(R+j) = strtol(first, NULL, 16);
    *(G+j) = strtol(second, NULL, 16);
    *(B+j) = strtol(third, NULL, 16);
    first = firstPointer;
    second = secondPointer;
    third = thirdPointer;
     //printf("%2x%2x%2x", *(R+j), *(G+j), *(B+j));
     
     
}
    //}
   // image+=1309;
	  //totalReached = row * colsize;
   // printf("%d reached \n", total - totalReached );
    //count++;
  //}
  
  //char str2[3];
  //int i= 0;
  /**
  while (totalReached < total)
  {
    sprintf(&h1, "%2x", image+(row *colsize) + col);
    sprintf(&h2, "%2x", image+(row *colsize) + col);
    sprintf(&h3, "%2x", image+(row *colsize) + col);
    totalReached++;
    col++;
    if (col==colsize)
    {
      row++;
      col = 0;
    }
  }
  */
  unsigned char *newImage2 = newImage;
  int lineno =0, linelen=200;
  printf("I get here\n");
  char *convertMe = malloc(sizeof(char) * 1);
  char *convertMePointer = convertMe;
	for(row=0;row<rowsize;row++){
    for (col=0;col<colsize;col++){
      sprintf(convertMe  ,"%02x",*(R+ (row * colsize) + col));
      *newImage2 = strtol(convertMe, NULL, 16);
      convertMe = convertMePointer;
      //sprintf(newImage2  ,"%02x%02x%02x",*(R+ (row * colsize) + col),*(G + (row * colsize) + col),*(B + (row * colsize) + col));
      newImage2+=1;
      sprintf(convertMe, "%02x", *(G + (row * colsize) + col));
      *newImage2 = strtol(convertMe, NULL, 16);
      convertMe = convertMePointer;
      //sprintf(newImage2  ,"%02x%02x%02x",*(R+ (row * colsize) + col),*(G + (row * colsize) + col),*(B + (row * colsize) + col));
      newImage2+=1;
      sprintf(convertMe, "%02x", *(B + (row * colsize) + col));
      *newImage2 = strtol(convertMe, NULL, 16);
      newImage2+=1;
      }
    
  }
  printf("I get here 2\n");

  size_t pngsize = width * height * 3;
  printf("I get here 3\n");
 // unsigned char * fileImage = (unsigned char *)malloc(sizeof(unsigned char) * width * height * 3);
  unsigned error2 = lodepng_encode24_file("file.png",newImage, width, height);
  printf("I get here 4\n");
  //if(!error2) lodepng_save_file(fileImage, pngsize, "newFile.png");
  printf("I get here 5\n");
  free(image);
  free(newImage);
  int f = 0;
  free(R);
  free(G);
  free(B);

}


int main(int argc, char *argv[])
{
  const char* filename = argc > 1 ? argv[1] : "Laplace.png";

  decodeOneStep("Laplace.png");
  
  return 0;
}
