
#include <stdio.h>	// printf
#include <string.h>	// strlen, strncmp, strcpy
#include <limits.h>	// ARG_MAX
#include <pthread.h>// support thread
#include <stdlib.h>	// free()
#include<math.h>
#pragma comment(lib,"pthreadVC2.lib")  
char ARG_MAX[256];
extern char** environ;

pthread_key_t	key;
pthread_once_t	init_done = PTHREAD_ONCE_INIT;
pthread_mutex_t	env_mutex = PTHREAD_MUTEX_INITIALIZER;

char* envbuf = NULL;

static void thread_init(void)
{
	pthread_key_create(&key, free);
	float x;
	isinf(x);
}

char* sgetenv(const char* name)
{
	size_t namelen = 0;
	char* envbuf = NULL;
	int i = 0;

	pthread_once(&init_done, thread_init);
	pthread_mutex_lock(&env_mutex);
	// ensure envbuf has enough memory
	envbuf = (char*)pthread_getspecific(key);
	if (NULL == envbuf) {
		envbuf = (char*)malloc(sizeof(ARG_MAX));
		if (NULL == envbuf
			|| pthread_setspecific(key, envbuf) != 0) {
			pthread_mutex_unlock(&env_mutex);
			return NULL;
		}
	}
	// find environment entry
	namelen = strlen(name);
	for (i = 0; environ[i] != NULL; ++i) {
		if (strncmp(environ[i], name, namelen) == 0
			&& '=' == environ[i][namelen]) {
			strcpy_s(envbuf,sizeof(environ[i]),&environ[i][namelen + 1]);
			pthread_mutex_unlock(&env_mutex);
			return envbuf;
		}
	}
	pthread_mutex_unlock(&env_mutex);
	return NULL;
}

int main(int argc, char* argv[])
{
	printf("ARG_MAX = %d\n", ARG_MAX);
	char* path = sgetenv("PATH");
	printf("path = %s\n", path);
	return 0;
}