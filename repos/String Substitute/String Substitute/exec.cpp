#include <stdlib.h>
#include<stdio.h>
#include<string.h>
bool debug = 1;
int numofchar(char* prototype, const char* target)
{
	char prototypel[3962];
	char tool[10];
	int length, i, count;
	count = 0;//statistics of final result
	length = strlen(prototype);
	for (i = 0; i < length; i++)
	{
		if (prototype[i] == tool[0])
			count++;
	}
	if (debug)
		printf("count is :%d", count);
	return count;
}
int main()
{

}