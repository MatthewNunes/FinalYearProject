#include <time.h>
#include <stdio.h>
#include <stdlib.h>

long unsigned int get_tick()
{
	struct timespec ts;
	if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return (0);
	return ts.tv_sec*(long int)1000 + ts.tv_nsec / (long int) 1000000;
}

