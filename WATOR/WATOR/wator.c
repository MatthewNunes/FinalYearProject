/*
 * Copyright 1989 O'Reilly and Associates, Inc.
 * See ../Copyright for complete rights and liability information.
 */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>

#include <stdio.h>

#include "icon_bitmap"
#define BITMAPDEPTH 1
#define MAX_COLORS 3

/* WaTor definitions */
#define MAX_OCEAN_SIZE 400
#define EMPTY 0
#define FISHY 1
#define SHARK 2

/* WaTor data structures */

typedef struct ocean{
        int what_sort;
        struct list *f_list;
} OCEAN;

typedef struct fish{
	int type, age, last_ate, iteration, x, y;
} FISH;

typedef struct list{
        FISH swimmer;
        struct list *prev;
        struct list *next;
} LIST;

OCEAN ocean[MAX_OCEAN_SIZE][MAX_OCEAN_SIZE];
LIST *fish_list;
int   list_size = sizeof(LIST);
int   fish_size = sizeof(FISH);
int update_number;
int rows, columns;
int rectwidth=4, rectheight=4;

/* Display and screen_num are used as arguments to nearly every Xlib routine, 
 * so it simplifies routine calls to declare them global.  If there were 
 * additional source files, these variables would be declared extern in
 * them. */
Display *display;
int screen_num;
Screen *screen_ptr;
Window win;
GC gc;

/* pixel values */
unsigned long foreground_pixel, background_pixel, border_pixel;

/* values for window_size in main, is window big enough to be useful? */
#define SMALL 1
#define OK 0

char *progname;

int main(argc, argv)
int argc;
char **argv;
{
	unsigned int width, height, x, y;     /* window size and position */
	unsigned int borderwidth = 4;	      /* four pixels */
	unsigned int display_width, display_height;
	unsigned int icon_width, icon_height;
	char *window_name = "WaTor Window Program";
	char *icon_name = "basicwin";
	Pixmap icon_pixmap;
	XSizeHints size_hints;
	XEvent report;
	XFontStruct *font_info;
	char *display_name = NULL;
	int window_size = 0;    /* OK, or too SMALL to display contents */
        int i;
        int num_fish, num_shark, breed_fish, breed_shark, starve_shark, seed;
	int oldage_shark, oldage_fish, updates_per_click;
        int ctimes;

        get_input(&num_fish, &num_shark, &breed_fish, &breed_shark, &starve_shark, 
                  &oldage_shark, &oldage_fish, &seed, &updates_per_click);
      
        setup_problem (num_fish, num_shark, seed);

	progname = argv[0];

	/* connect to X server */

	if ( (display=XOpenDisplay(display_name)) == NULL )
	{
		(void) fprintf( stderr, 
				"basicwin: cannot connect to X server %s\\n",
				XDisplayName(display_name));
		exit( -1 );
	}

	/* get screen_num size from display structure macro */
	screen_num = DefaultScreen(display);
	screen_ptr = DefaultScreenOfDisplay(display);
	display_width = DisplayWidth(display, screen_num);
	display_height = DisplayHeight(display, screen_num);

	/* place window */
	x = 50; y = 50;

	/* size window */
	height = rows*rectheight; width = columns*rectwidth;

	get_colors();

	/* create opaque window */
	win = XCreateSimpleWindow(display, RootWindow(display,screen_num), x, y, 
			width, height, borderwidth, border_pixel,
	    		background_pixel);

	/* Create pixmap of depth 1 (bitmap) for icon */
	icon_pixmap = XCreateBitmapFromData(display, win, icon_bitmap_bits, 
			icon_bitmap_width, icon_bitmap_height);

	/* Set resize hints */
	size_hints.flags = PPosition | PSize | PMinSize;
	size_hints.x = x;
	size_hints.y = y;
	size_hints.width = width;
	size_hints.height = height;
	size_hints.min_width = 350;
	size_hints.min_height = 250;

	/* set Properties for window manager (always before mapping) */
	XSetStandardProperties(display, win, window_name, icon_name, 
	    icon_pixmap, argv, argc, &size_hints);

	/* Select event types wanted */
	XSelectInput(display, win, ExposureMask | KeyPressMask | 
			ButtonPressMask | StructureNotifyMask);

	load_font(&font_info);

	/* create GC for text and drawing */
	get_GC(font_info);

	/* Display window */
	XMapWindow(display, win);

/*        printf("Input integer to continue => "); scanf("%d",&i);*/
	/* get events, use first to display text and graphics */
        ctimes = 0;
	while (1)  {
		XNextEvent(display, &report);
		switch  (report.type) {
		case Expose:
			/* get all other Expose events on the queue */
			while (XCheckTypedEvent(display, Expose, &report));
                        DrawArray ();
			break;
		case ButtonPress:
                        for (i=0;i<updates_per_click;++i){
                          update_ocean (starve_shark, breed_shark, breed_fish, 
                                        oldage_shark, oldage_fish);
                        }
			break;
/*
		case ConfigureNotify:
                     if(ctimes>0){
			width = report.xconfigure.width;
			height = report.xconfigure.height;
                        rows = (height-1)/rectheight + 1;
                        columns = (width-1)/rectwidth + 1;
                        setup_problem (num_fish, num_shark, seed);
                     }
                        ctimes++;
			break;
*/
		case KeyPress:
			XUnloadFont(display, font_info->fid);
			XFreeGC(display, gc);
			XCloseDisplay(display);
			exit(1);
		default:
			/* all events selected by StructureNotifyMask
			 * except ConfigureNotify are thrown away here,
			 * since nothing is done with them */
			break;
		} /* end switch */
	} /* end while */
}

get_GC(font_info)
XFontStruct *font_info;
{
	unsigned long valuemask = 0; /* ignore XGCvalues and use defaults */
	XGCValues values;
	unsigned int line_width = 6;
	int line_style = LineOnOffDash;
	int cap_style = CapRound;
	int join_style = JoinRound;
	int dash_offset = 0;
	static char dash_list[] = {
		12, 24	};
	int list_length = 2;

	/* Create default Graphics Context */
	gc = XCreateGC(display, win, valuemask, &values);

	/* specify font */
	XSetFont(display, gc, font_info->fid);

	/* specify black foreground since default may be white on white */
	XSetForeground(display, gc, foreground_pixel);

	/* set line attributes */
	XSetLineAttributes(display, gc, line_width, line_style, cap_style, 
			join_style);

	/* set dashes to be line_width in length */
	XSetDashes(display, gc, dash_offset, dash_list, list_length);
}

load_font(font_info)
XFontStruct **font_info;
{
	char *fontname = "9x15";

	/* Access font */
	if ((*font_info = XLoadQueryFont(display,fontname)) == NULL)
	{
		(void) fprintf( stderr, "Basic: Cannot open 9x15 font\\n");
		exit( -1 );
	}
}

DrawArray()
{
        int i, j, thiscolor, xupperleft, yupperleft;
        int left, right, up, down;

        for (i=0;i<rows;i++){
           yupperleft = i*rectheight;
           for (j=0;j<columns;j++){
              thiscolor = ocean[i][j].what_sort;
              xupperleft = j*rectwidth;
              if (thiscolor == 0) 
                 XSetForeground (display, gc, border_pixel);
              else if (thiscolor == 1)
                 XSetForeground (display, gc, background_pixel);
              else{
                 XSetForeground (display, gc, foreground_pixel);
              }
              XFillRectangle (display, win, gc, xupperleft, yupperleft, 
                              rectwidth, rectheight);
           }
        }
}

get_input (num_fish, num_shark, breed_fish, breed_shark, starve_shark, oldage_shark,
           oldage_fish, seed, updates_per_click)
int *num_fish, *num_shark, *breed_fish, *breed_shark, *starve_shark, *oldage_shark,
    *oldage_fish, *seed, *updates_per_click;
{
        printf("\n\nPlease give the number of rows and columns ==> ");
        scanf("%d%d",&rows,&columns);

	printf("\n\nPlease give the initial number of fish and sharks ==> ");
	scanf("%d%d",num_fish,num_shark);

	printf("\n\nPlease give the breeding age of fish and sharks ==> ");
	scanf("%d%d",breed_fish,breed_shark);

	printf("\n\nPlease give the age at which fish and sharks die ==> ");
	scanf("%d%d",oldage_fish,oldage_shark);

	printf("\n\nPlease give the shark starvation age ==> ");
	scanf("%d",starve_shark);

	printf("\n\nPlease give a random number seed ==> ");
	scanf("%d",seed);

	printf("\n\nPlease give number of updates per button click ==> ");
	scanf("%d",updates_per_click);
}

setup_ocean()
{
	int i,j;

	for(i=0;i<rows;++i){
		for(j=0;j<columns;++j){
			ocean[i][j].what_sort = EMPTY;
			ocean[i][j].f_list    = (LIST *)NULL;
		}
	}
}

setup_problem(num_fish, num_shark, seed)
int num_fish, num_shark, seed;
{
        update_number = 0;
	setup_ocean();
	setup_fish_list ();
	setup_fish(num_fish, FISHY);
	setup_fish(num_shark, SHARK);
}

setup_fish_list ()
{
	fish_list = (LIST *)malloc(list_size);
        fish_list->next = (LIST *)NULL;
        fish_list->prev = (LIST *)NULL;
}

setup_fish(number, type)
int number, type;
{
	int x, y, count;
	double myrand();

	count = 0;

	while(count<number){
		x = (int)(myrand()*(double)(columns));
		y = (int)(myrand()*(double)(rows));
                if (ocean[y][x].what_sort == EMPTY) {
                	create_fish (x, y, type);
                        count++;
                }
	}
}

create_fish(x, y, type)
int x, y, type;
{
	FISH ichy;
        LIST *listp;

	ichy.x         = x;
	ichy.y         = y;
	ichy.type      = type;
	ichy.age       = 0;
	ichy.last_ate  = 0;
	ichy.iteration = update_number;

        listp = fish_list;
	insert(&ichy,listp);

	ocean[y][x].what_sort = type;
	ocean[y][x].f_list    = fish_list->next;
}

insert (fishp, listp)
FISH *fishp;
LIST *listp;
{
	LIST *new;

	new = (LIST *)malloc(list_size);

	new->swimmer = *fishp;
	if(listp->next != (LIST *)NULL) listp->next->prev = new;
	new->next = listp->next;
	new->prev = listp;
	listp->next = new;
	listp->prev = (LIST *)NULL;
}

update_ocean(starve_shark, breed_shark, breed_fish, oldage_shark, oldage_fish)
int starve_shark, breed_shark, breed_fish, oldage_shark, oldage_fish;
{
	FISH ichy;
	LIST *listp;
	int age, starve,type,x,y,dir,give_dir(),xnew,ynew;

        listp = fish_list;
   	update_number++;

	while(!is_empty(listp)){
		ichy = listp->next->swimmer;
  		if (ichy.iteration == update_number-1){
			age_fish(listp);
			x      = ichy.x;
			y      = ichy.y;
                	starve = ichy.last_ate;
                	type   = ichy.type;
 			age    = ichy.age;
			if(type==SHARK){
				if(starve>starve_shark || age>oldage_shark){
			   		remove_fish(x,y);
				}
				else{
					dir = give_dir(x,y,FISHY);
					if(dir>=0){
				   		dirtoxy(dir,x,y,&xnew,&ynew);
				   		eat_fish(listp,xnew,ynew,x,y,breed_shark);
					}
					else{
				   		dir = give_dir(x,y,EMPTY);
				   		if(dir>=0){
				     			dirtoxy(dir,x,y,&xnew,&ynew);	
				     			move_location(listp, xnew, ynew, x, y,
                                                                      breed_shark, breed_fish);
							listp->next->swimmer.last_ate++;
                                                        listp = listp->next;
				    		}
					}
				}
			}
			else{
				if(age>oldage_fish){
			   		remove_fish(x,y);
				}
				else{
					dir = give_dir(x,y,EMPTY);
					if(dir>=0){
		           			dirtoxy(dir,x,y,&xnew,&ynew); 
			   			move_location(listp, xnew, ynew, x, y, 
                                                      	breed_shark, breed_fish);
                                        	listp = listp->next;
					}
				}
			}
		}
                else {
                	listp = listp->next;
                }
	}
}

age_fish(listp)
LIST *listp;
{
	listp->next->swimmer.last_ate++;
	listp->next->swimmer.age++;
	listp->next->swimmer.iteration++;
}

eat_fish(listp,xnew,ynew,nx,ny,breed_shark)
LIST *listp;
int xnew,ynew,nx,ny,breed_shark;
{
	LIST *food;

	food = ocean[ynew][xnew].f_list;
	ocean[ynew][xnew].what_sort = SHARK;
        XSetForeground (display, gc, foreground_pixel);
        XFillRectangle (display, win, gc, xnew*rectwidth, ynew*rectheight,
                        rectwidth, rectheight);
        

	food->swimmer.type = SHARK;
	food->swimmer.last_ate = 0;
	food->swimmer.age    = listp->next->swimmer.age;
	food->swimmer.iteration = listp->next->swimmer.iteration;

	if(listp->next->swimmer.age >= breed_shark){
		listp->next->swimmer.last_ate = 0;
		listp->next->swimmer.age    = 0;
        	XSetForeground (display, gc, foreground_pixel);
        	XFillRectangle (display, win, gc, xnew*rectwidth, ynew*rectheight,
                        	rectwidth, rectheight);
	}
	else{
		remove_fish(nx,ny);
	}
}

int give_dir(x,y,type)
int x,y,type;
{
	int numposs,poss[4];
	double myrand();

	numposs = 0;
	if(ocean[y][(x+1)%columns].what_sort == type) poss[numposs++] = 0;
	if(ocean[y][(x-1+columns)%columns].what_sort == type) poss[numposs++] = 1;
	if(ocean[(y+1)%rows][x].what_sort == type) poss[numposs++] = 2;
	if(ocean[(y-1+rows)%rows][x].what_sort == type) poss[numposs++] = 3;

	if(numposs>0) return poss[(int)(numposs*myrand())];
	else return -1;
}

dirtoxy(dir,x,y,pxnew,pynew)
int dir,x,y,*pxnew,*pynew;
{
	switch(dir){
		case 0:
			*pxnew = (x+1)%columns;
			*pynew = y;
			break;
		case 1:
			*pxnew = (x-1+columns)%columns;
			*pynew = y;
			break;
		case 2:
			*pxnew = x;
			*pynew = (y+1)%rows;
			break;
		case 3:
			*pxnew = x;
			*pynew = (y-1+rows)%rows;
			break;
	}
}

int is_empty(listp)
LIST *listp;
{
	return (listp->next == (LIST *)NULL);
}

move_location(listp,xnew,ynew,nx,ny, breed_shark, breed_fish)
LIST *listp;
int xnew,ynew,nx,ny, breed_shark, breed_fish;
{
	int breed,type;
	int check_breed();

	type = listp->next->swimmer.type;

	listp->next->swimmer.x = xnew;
	listp->next->swimmer.y = ynew;

	breed = check_breed(listp,breed_shark, breed_fish);

	ocean[ynew][xnew].what_sort = type;
	ocean[ynew][xnew].f_list    = listp->next;
	if (type == FISHY)
        	XSetForeground (display, gc, background_pixel);
	else if (type == SHARK)
        	XSetForeground (display, gc, foreground_pixel);
        XFillRectangle (display, win, gc, xnew*rectwidth, ynew*rectheight,
                        rectwidth, rectheight);

	if(breed){
		create_fish(nx,ny,type);
	}
	else{
		ocean[ny][nx].what_sort = EMPTY;
		ocean[ny][nx].f_list    = (LIST *)NULL;
        	XSetForeground (display, gc, border_pixel);
        	XFillRectangle (display, win, gc, nx*rectwidth, ny*rectheight,
                        	rectwidth, rectheight);
	}
}

remove_fish(nx,ny)
int nx,ny;
{
	LIST *listp;

	listp = ocean[ny][nx].f_list;
	remove_from_list(listp);
	free((char*)listp);

	ocean[ny][nx].what_sort = EMPTY;
	ocean[ny][nx].f_list    = (LIST *)NULL;
        XSetForeground (display, gc, border_pixel);
        XFillRectangle (display, win, gc, nx*rectwidth, ny*rectheight,
                        rectwidth, rectheight);
}

int check_breed(listp, breed_shark, breed_fish)
int breed_shark, breed_fish;
LIST *listp;
{
	int age,type,breed_age;

	age  = listp->next->swimmer.age;
	type = listp->next->swimmer.type;
	breed_age = (type == SHARK) ? breed_shark : breed_fish;

	return (age>=breed_age);
}

remove_from_list(listp)
LIST *listp;
{
	listp->prev->next = listp->next;
	if(listp->next != (LIST *)NULL) listp->next->prev = listp->prev;
}

list_all_fish()
{
	LIST *listp;
        FISH ichy;
	int  count = 0;

        listp = fish_list;
	while (!is_empty(listp)) {
		count++;
                ichy = listp->next->swimmer;
		printf("%d : %4d %4d %4d %4d %4d %4d\n", count, ichy.type, ichy.age,
                       ichy.last_ate, ichy.iteration, ichy.x, ichy.y);
		listp = listp->next;
	}
}
