#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <shlobj.h>
#include <shlwapi.h>
#include <objbase.h>

#include <vector>
using namespace std;
#define PATHL 300
bool getcurrentpath(char* cpath);
ifstream myfile_s;
ifstream in;
ofstream out;
/*
@ Brief:    修改行数据
@ Author : Litianyu
@ Created : 2022 / 04 / 02
@ Return:
*/
/*最多启动PATHL个程序*/

/************************************************************************/
/* char*tostr  字符串转化str类型
输入：char * 字符串地址
无输出
返回值： str类型的字符串
*/
/************************************************************************/
string charToStr(char* contentChar)
{
    string tempStr;
    for (int i = 0; contentChar[i] != '\0'; i++)
    {
        tempStr += contentChar[i];
    }
    return tempStr;
}

/************************************************************************/
/* 修改文件某行内容
 输入：文件名 fileName   行号   lineNum ,修改的内容 content
 输出：文件名 fileName
 无返回值
 tip：1,lineNum从第一行开始 2.content需要加上换行符
*/
/************************************************************************/

bool modifyContentInFile(char* fileName, int lineNum, char* content)
{
    
    char line[PATHL];
    in.open(fileName,ios::in);
    int i = 0;
    char* outputdata = new char;
    vector<string> readeddata;
    string linedata = "";
    bool readflag = false;
    while (in.getline(line, sizeof(line)))
    {
       
        if (lineNum == i) 
        {
            readeddata.push_back(content);
        }
        else
        {
            linedata = line;
            readeddata.push_back(linedata);
        }
        i++;
    }
    in.close();
    readflag = true;
    if (readflag)
    {
        out.open(fileName, ios::out);
        for (int j = 0; j < i; j++)
        {
            outputdata = (char*)readeddata[j].c_str();
            out << outputdata<<endl;
        }
        out.close();
        return true;
    }
    else
    {
        return false;
    }
    
    
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
    strcpy_s(args, strlen(args) + 1, (char*)exepath_t.c_str());


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

    return NULL;
}
static void* shellopen(char* cmd)
{
    string exepath_t = "";
    char args[1024];
    WCHAR Excutecmd[1024];
    HWND hwnd = NULL;
    exepath_t = cmd;
    wstring str;
    str.append(L"\\cmd.exe");

    str = stringToWstring(exepath_t);
    strcpy_s(args, strlen(args) + 1, (char*)exepath_t.c_str());

    memset(Excutecmd, 0, sizeof(Excutecmd));
    MultiByteToWideChar(CP_ACP, 0, args, strlen(args) + 1, Excutecmd,
        sizeof(Excutecmd) / sizeof(Excutecmd[0]));

    ShellExecute(hwnd, NULL, Excutecmd, NULL, NULL, SW_HIDE);

    return NULL;
}
//将char转为TCHAR 
bool CharToTchar(const char* _char, TCHAR* tchar)
{
    int iLength;

    iLength = MultiByteToWideChar(CP_ACP, 0, _char, strlen(_char) + 1, NULL, 0);
    MultiByteToWideChar(CP_ACP, 0, _char, strlen(_char) + 1, tchar, iLength);

    return true;

}
//将TCHAR转为char   
//*tchar是TCHAR类型指针，*_char是char类型指针   
bool TcharToChar(TCHAR* tchar, char* _char)
{
    int iLength;
    //获取字节长度   
    iLength = WideCharToMultiByte(CP_ACP, 0, tchar, -1, NULL, 0, NULL, NULL);
    //将tchar值赋给_char    
    WideCharToMultiByte(CP_ACP, 0, tchar, -1, _char, iLength, NULL, NULL);
    return true;
}
static void* shellopenEx(char* cmd)
{
    LPITEMIDLIST pidlWinFiles = NULL;
    LPITEMIDLIST pidlItems = NULL;
    IShellFolder* psfWinFiles = NULL;
    IShellFolder* psfDeskTop = NULL;
    LPENUMIDLIST ppenum = NULL;
    STRRET strDispName;
    TCHAR pszParseName[MAX_PATH];
    ULONG celtFetched;
    SHELLEXECUTEINFO ShExecInfo;
    HRESULT hr;
    BOOL fBitmap = FALSE;

    hr = SHGetFolderLocation(NULL, CSIDL_WINDOWS, NULL, NULL, &pidlWinFiles);

    hr = SHGetDesktopFolder(&psfDeskTop);

    hr = psfDeskTop->BindToObject(pidlWinFiles, NULL, IID_IShellFolder, (LPVOID*)&psfWinFiles);
    hr = psfDeskTop->Release();

    hr = psfWinFiles->EnumObjects(NULL, SHCONTF_FOLDERS | SHCONTF_NONFOLDERS, &ppenum);

    CharToTchar(cmd, pszParseName);
    fBitmap = TRUE;
    ppenum->Release();

    if (fBitmap)
    {
        ShExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
        ShExecInfo.fMask = NULL;
        ShExecInfo.hwnd = NULL;
        ShExecInfo.lpVerb = NULL;
        ShExecInfo.lpFile = pszParseName;
        ShExecInfo.lpParameters = NULL;
        ShExecInfo.lpDirectory = NULL;
        ShExecInfo.nShow = SW_MAXIMIZE;
        ShExecInfo.hInstApp = NULL;

        ShellExecuteEx(&ShExecInfo);
    }

    CoTaskMemFree(pidlWinFiles);
    psfWinFiles->Release();

    return 0;
}
bool getcurrentpath(char* cpath)
{
    TCHAR szPath[_MAX_PATH] = { 0 };
    TCHAR szDrive[_MAX_DRIVE] = { 0 };
    TCHAR szDir[_MAX_DIR] = { 0 };
    TCHAR szFname[_MAX_FNAME] = { 0 };
    TCHAR szExt[_MAX_EXT] = { 0 };
    char dir[260] = { "" };
    char drive[520] = { "" };
    GetModuleFileName(NULL, szPath, sizeof(szPath));
    memset(cpath,' ',PATHL);
    _wsplitpath_s(szPath, szDrive, szDir, szFname, szExt);
    TcharToChar(szDrive, drive);
    TcharToChar(szDir, dir);
    strcat_s(drive, sizeof(drive), dir);
    strcpy_s(cpath, strlen(cpath) + 1, drive);
    return true;
}
bool getPATHbyusers(char* PATH_t)
{
    bool flag = false;
    char test[PATHL];
    string exepath = "";
    string Re_exepath = "";
    string temp = "";
    string filepath_s = "";
    char *filepath = new char;
    int rewrite = 2;
    char tmpLineData[1024] = { 0 };
    int line = 0;
    int edittimes = 0;
    char currentPath[PATHL];
    TCHAR szPath_infile[_MAX_PATH] = { 0 };
    TCHAR szDrive_infile[_MAX_DRIVE] = { 0 };
    TCHAR szDir_infile[_MAX_DIR] = { 0 };
    TCHAR szFname_infile[_MAX_FNAME] = { 0 };
    TCHAR szExt_infile[_MAX_EXT] = { 0 };

    TCHAR szPath[_MAX_PATH] = { 0 };
    TCHAR szDrive[_MAX_DRIVE] = { 0 };
    TCHAR szDir[_MAX_DIR] = { 0 };
    TCHAR szFname[_MAX_FNAME] = { 0 };
    TCHAR szExt[_MAX_EXT] = { 0 };

    getcurrentpath(currentPath);
    filepath_s = currentPath;
    filepath_s += "Path_exe.txt";
    filepath = (char*)filepath_s.c_str();
    myfile_s.open(filepath,ios::in);
    while (!flag) 
    {
        cout << "        " << "请输入要执行的文件路径及其文件名.exe，如C:\\Users\\DOOMDUKE2.exe-nr，后缀加上-nr表示不进行路径检查" << endl;
        cout << "        " << "请输入:";
        cin.getline(test,PATHL);
        exepath = test;
        while (!flag)
        {
            Re_exepath = exepath;
            int postion = 0;
            postion = exepath.find("-nr");
            if (postion == -1)
            {
                cout << endl << endl <<"        "<< "获得的执行路径为:" << exepath << endl << "        " << "若路径有错, 请重新输入, 若没错请输入true" << endl;
                Re_exepath = exepath;
                cout << "        " << "请输入:"; cin.getline(test, PATHL); exepath = test;
                 if (exepath.compare("true") == 0)
                {
                    flag = true;
                    exepath = Re_exepath;
                }
                else
                {
                    cout << endl << endl << "        " << "获得的执行路径为:" << exepath << endl << "        " << "若路径有错, 请重新输入, 若没错请输入true" << endl;
                    cin.getline(test, PATHL);exepath = test;
                    if (exepath.compare("true") == 0)
                    {
                        flag = true;
                        exepath = Re_exepath;
                    }
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
    
    if (!myfile_s.is_open())
    {
      ofstream outfile(filepath, ios::app);
      outfile << PATH_t<<endl;
      cout << endl << endl << "        " << "当前路径已保存到本程序同目录下名为Path_exe.txt的文件中..." << endl;
      outfile.close();
    }
    else
    {
        ofstream outfile(filepath, ios::app);
        if (!getline(myfile_s, temp))//打开为空文件
        { 
            outfile << PATH_t << endl;
            cout << endl << endl << "        " << "当前路径已保存到本程序同目录下名为Path_exe.txt的文件中..." << endl;
            outfile.close();
        }
        else 
        {
            myfile_s.close();
            myfile_s.open(filepath, ios::in);
            while (getline(myfile_s, temp))
            {
                if (temp.npos != temp.find(".exe"))
                {
                    char szFname_infile_c[PATHL];
                    char szFname_c[PATHL];
                    CharToTchar((char*)temp.c_str(), szPath_infile);
                    CharToTchar(PATH_t, szPath);
                    _wsplitpath_s(szPath_infile, szDrive_infile, szDir_infile, szFname_infile, szExt_infile);
                    _wsplitpath_s(szPath, szDrive, szDir, szFname, szExt);
                    TcharToChar(szFname_infile, szFname_infile_c);
                    TcharToChar(szFname, szFname_c);
                    if (strcmp(szFname_c, szFname_infile_c) == 0)
                    {
                        edittimes++;
                        while (rewrite != 0 && rewrite != 1)
                        {
                            cout <<endl << "        " << "发现Path_exe.txt文件中有同名但“不同路径”的程序：";
                            cout << temp << endl;
                            cout << "        " << "是否覆盖，输入0或1:";
                            cin >> rewrite; cin.get();
                            if (rewrite)
                            {
                                myfile_s.close();
                                if(modifyContentInFile(filepath, line, PATH_t))
                                {
                                    cout << endl << endl << "        " << "当前路径已保存到本程序同目录下名为Path_exe.txt的文件中..." << endl;
                                }
                                else
                                {
                                    cout << endl << endl << "        " << "文件覆盖失败..." << endl;
                                }
                               
                            }
                            else
                            {
                                outfile << PATH_t << endl;
                                cout << endl << endl << "        " << "当前路径已保存到本程序同目录下名为Path_exe.txt的文件末尾中..." << endl;
                                outfile.close();
                            }
                        }

                    }
                }
                line++;//记录当前读到第几行了
            }
            if (edittimes == 0)
            {
                outfile << PATH_t << endl;
                cout << endl << endl << "        " << "当前路径已保存到本程序同目录下名为Path_exe.txt的文件中..." << endl;
                outfile.close();
            }

        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    char currentPath[260];
    char EXE_PATH[PATHL][PATHL];
    int nproc = 0;
    int spawnfromfile = 2;
    ifstream myfile;
    string temp = "";
    memset(currentPath,' ', 260);
    getcurrentpath(currentPath);
    strcat_s(currentPath,sizeof(currentPath),"Path_exe.txt");
    cout <<"      " << "**************************************************************************************************" << endl;
    cout << "      " << "**************************************************************************************************" << endl;
    cout << "      " << "**************************************************************************************************" << endl;
    cout << "      " << "**************************************************************************************************" << endl;
    cout << "      " << "***************************************程序启动器——v0.1*****************************************" << endl;
    cout << "      " << "**************************************************************************************************" << endl;
    cout << "      " << "**************************************************************************************************" << endl;
    cout << "      " << "**************************************************************************************************" <<endl<<endl;
    while (spawnfromfile != 0 && spawnfromfile != 1)
    {
        cout << "是否从文件启动,输入0或1:"; cin >> spawnfromfile; cin.get();
    }
    if (spawnfromfile == 0) 
    {
        cout << "请输入想要启动的程序个数:"; cin >> nproc; cin.get();
        for (int n = 0; n < nproc; n++)
        {
            cout << "当前第" << n + 1 << "个" << endl;
            memset(EXE_PATH[n], ' ', PATHL);
            getPATHbyusers(EXE_PATH[n]);
            cout << endl << "        " << "正在启动第" << n + 1 << "个程序" << endl;
            shellopenEx(EXE_PATH[n]);
        }
    }
    else
    {
        myfile.open(currentPath,ios::in);
        if (!myfile.is_open())
        {
            cout << "未成功打开文件,执行失败" << endl;
        }
        else if (!getline(myfile, temp))//打开为空文件
        {
            cout <<"打开为空文件,执行失败" << endl;
            myfile.close();
            return 0;
        }
        else
        {
            myfile.close();
            myfile.open(currentPath, ios::in);
            while (getline(myfile, temp))
            {
                shellopenEx((char*)temp.c_str());
                Sleep(100);
            }
            myfile.close();
        }
    }
    system("PAUSE ");
    return 0;
}

