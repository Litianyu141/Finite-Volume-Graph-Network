#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <string>
#include <cstring>
#include <iostream>
using namespace std;
/*最多启动256个程序*/
bool getPATHbyusers(char* PATH_t)
{
    bool flag = false;
    char test[256];
    string exepath = "";
    string Re_exepath = "";
    while (!flag) 
    {
        cout << "请输入要执行的文件路径及其文件名.exe，如C:\\Users\\DOOMDUKE2.exe-nr，后缀加上-nr表示不进行路径检查" << endl;
        cout << "请输入:"; 
        cin.getline(test,256);
        exepath = test;
        while (!flag)
        {
            Re_exepath = exepath;
            int postion;
            postion =  exepath.find("-nr");
            if (exepath.npos == postion)
            {
                cout << endl << "获得的执行路径为:" << exepath << "若路径有错, 请重新输入, 若没错请输入true" << endl;
                cout << "请输入:"; 
                cin >> exepath;
                if (exepath.compare("true") == 0)
                {
                    flag = true;
                    exepath = Re_exepath;
                }
                else
                {
                    cout << "请重新输入路径" << endl;
                }
            }
            else
            {
                flag = true;
                exepath = Re_exepath;
                exepath.erase(postion,3);
            }
          
        }
        strcpy_s(PATH_t, strlen(PATH_t) + 1, (char*)exepath.c_str());
    }
    return true;
}
wstring stringToWstring(const std::string& str)
{
    LPCSTR pszSrc = str.c_str();
    int nLen = MultiByteToWideChar(CP_ACP, 0, pszSrc, -1, NULL, 0);
    if (nLen == 0)
        return std::wstring(L"");

    wchar_t* pwszDst = new wchar_t[nLen];
    if (!pwszDst)
        return std::wstring(L"");

    MultiByteToWideChar(CP_ACP, 0, pszSrc, -1, pwszDst, nLen);
    std::wstring wstr(pwszDst);
    delete[] pwszDst;
    pwszDst = NULL;

    return wstr;
}
static void* pipeopen(char* cmd)
{
    string exepath_t = "";
    TCHAR szPath[MAX_PATH];
    char args[1024];
    GetSystemDirectory(szPath, sizeof(szPath));

    exepath_t = cmd;
    wstring str = szPath;
    str.append(L"\\cmd.exe");

    wstring cmdStr = L"systeminfo";

    STARTUPINFO si;
    ZeroMemory(&si, sizeof(si));

    si.cb = sizeof(si);
    si.wShowWindow = SW_HIDE;

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    TCHAR temp[2048] = L"/c systeminfo";
    str = stringToWstring(exepath_t);
    strcpy_s(args, strlen(args), (char*)str.c_str());

   
    WCHAR wszClassName[1024];
    memset(wszClassName, 0, sizeof(wszClassName));
    MultiByteToWideChar(CP_ACP, 0, args, strlen(args) + 1, wszClassName,
        sizeof(wszClassName) / sizeof(wszClassName[0]));


    BOOL bRet = CreateProcess(NULL, wszClassName, NULL, NULL, FALSE, NORMAL_PRIORITY_CLASS, NULL, NULL, &si, &pi);//
    if (bRet)
    {
        WaitForSingleObject(pi.hProcess, 3000);// 等待程序退出
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }

    return 0;
}

int main(int argc, char *argv[])
{
    char EXE_PATH[256][256];
    int nproc = 0;
    cout << "请输入想要启动的程序个数:"; cin >> nproc; cin.get();
    for (int n = 0; n < nproc; n++)
    {
        cout << "第" << n+1 << "个"<<endl;
        memset(EXE_PATH[n], ' ', 256);
        getPATHbyusers(EXE_PATH[n]);
        cout << endl << "正在启动第"<<n+1<<"个程序"<< endl;
        pipeopen(EXE_PATH[n]);
    }
    system("PAUSE ");
    return 0;
}