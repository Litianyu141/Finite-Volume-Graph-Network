#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <tchar.h>
#include <locale.h>
int main()
{
	TCHAR path[MAX_PATH];
	HMODULE hm = GetModuleHandle(NULL);
	GetModuleFileName(hm, path, sizeof(path));
	printf("%s\n", path);
	system("pause");
	return 0;
}