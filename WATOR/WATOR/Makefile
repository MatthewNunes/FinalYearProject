OBJS = wator.o get_color.ro.o myrand.o

wator: $(OBJS)
	gcc -o wator $(OBJS) -L/usr/X11R6/lib -lX11

wator.o: wator.c
	gcc -c wator.c

get_color.o: get_color.c
	gcc -c get_color.c

myrand.o: myrand.c
	gcc -c myrand.c
