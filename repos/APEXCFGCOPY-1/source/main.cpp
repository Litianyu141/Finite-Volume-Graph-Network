#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <windows.h>
#include "MyFile.h"
#include <direct.h>
using namespace std;
/*
@ Brief:    APEX的CFG复制器
@ Author : Litianyu
@ Created : 2022 / 04 / 02
@ Return:
*/
base::~base()
{
	//cout << "base Destructor Invoked" << endl;
}
MyFile::~MyFile()
{
	//cout << "MyFile Destructor Invoked" << endl;
}
bool base::setfileAttributes(TCHAR* path,bool isreadoly)
{
	if (isreadoly)
	{
		if (SetFileAttributes(path, FILE_ATTRIBUTE_READONLY))
		{
			cout << endl << "        " << "成功设置文件为只读" << endl << endl;
			return true;
		}
	}
	else
	{
		if (SetFileAttributes(path, -FILE_ATTRIBUTE_READONLY))
		{
			cout << endl << "        " << "成功取消文件只读" << endl << endl;
			return true;
		}
	}
	return false;
	
}
std::vector<std::string>  base::stringSplit(const std::string& strIn, char delim) 
{
	char* str = const_cast<char*>(strIn.c_str());
	std::string s;
	s.append(1, delim);
	std::vector<std::string> elems;
	char* context = NULL;
	char* splitted = strtok_s(str, s.c_str(), &context);
	while (splitted != NULL) {
		elems.push_back(std::string(splitted));
		splitted = strtok_s(NULL, s.c_str(),&context);
	}
	return elems;
}
bool base::CharToTchar(const char* _char, TCHAR* tchar)
{
	int iLength;

	iLength = MultiByteToWideChar(CP_ACP, 0, _char, strlen(_char) + 1, NULL, 0);
	MultiByteToWideChar(CP_ACP, 0, _char, strlen(_char) + 1, tchar, iLength);

	return true;

}
TCHAR* base::CharToTchar(const char* _char)
{
	int iLength;
	TCHAR* tchar = new TCHAR[MAX_PATH];
	iLength = MultiByteToWideChar(CP_ACP, 0, _char, strlen(_char) + 1, NULL, 0);
	MultiByteToWideChar(CP_ACP, 0, _char, strlen(_char) + 1, tchar, iLength);

	return tchar;

}
bool base::TcharToChar(TCHAR* tchar, char* _char)
{
	int iLength;
	//获取字节长度   
	iLength = WideCharToMultiByte(CP_ACP, 0, tchar, -1, NULL, 0, NULL, NULL);
	//将tchar值赋给_char    
	WideCharToMultiByte(CP_ACP, 0, tchar, -1, _char, iLength, NULL, NULL);
	return true;
}
char* base::TcharToChar(TCHAR* tchar)
{
	char* _char = new char[MAX_PATH];
	int iLength;
	//获取字节长度   
	iLength = WideCharToMultiByte(CP_ACP, 0, tchar, -1, NULL, 0, NULL, NULL);
	//将tchar值赋给_char    
	WideCharToMultiByte(CP_ACP, 0, tchar, -1, _char, iLength, NULL, NULL);
	return _char;
}
string base::CharToStr(char* contentChar)
{
	string tempStr;
	for (int i = 0; contentChar[i] != '\0'; i++)
	{
		tempStr += contentChar[i];
	}
	return tempStr;
}
wstring base::stringToWstring(const std::string& str)
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
void base::split_path(TCHAR *szPathwithFilename)
{
	TCHAR OnlyPath[_MAX_PATH];

	wcscpy_s(base::szPathwithFilename, szPathwithFilename);

	_wsplitpath_s(szPathwithFilename, base::szDrive, base::szDir, base::szFname, base::szExt);

	wcscpy_s(OnlyPath, base::szDrive);
	wcscat_s(OnlyPath, base::szDir);
	wcscpy_s(base::szPath, OnlyPath);
}
bool base::buildmulitydir(char* strPath)
{
	char builddir_t[_MAX_PATH] = {};
	char builddir[_MAX_PATH] = {};
	char partition[3] = "\\";
	char* token = new char[10];
	char* context = NULL;

	memset(builddir_t, ' ', _MAX_PATH);
	memset(builddir, ' ', _MAX_PATH);

	//strcpy_s(builddir_t,strlen(builddir_t)+1, strPath);
	for (int i =0; i <= strlen(strPath); i++)
	{
		builddir_t[i] = strPath[i];
	}
	token = strtok_s(builddir_t, partition, &context);

	//strcpy_s(builddir, strlen(builddir) + 1, token);
	for (int i = 0; i <= strlen(token); i++)
	{
		builddir[i] = token[i];
	}
	while (token!=NULL)
	{
		token = strtok_s(NULL, partition, &context);
		if (token != NULL)
		{
			strcat_s(builddir, sizeof(builddir), "\\");
			strcat_s(builddir, sizeof(builddir), token);
			_mkdir(builddir);
		}
	}
	
	cout << endl <<"        " << "成功建立目录" << builddir << endl << endl;
	delete[]token;
	return true;
}
void base::eraseenddir_p(void)
{
	char* path_t = new char[_MAX_PATH];
	char* path_t_r = new char[_MAX_PATH];//返回值
	int partition = 0;
	int times = 0;
	base::TcharToChar(base::szPath, path_t);
	int i = strlen(path_t) - 1;
	for (i; i >=0; i--)
	{
		if (path_t[i]=='\\')
		{
			times++;
			if (times >= 2)
			{
				partition = i;//反向遍历第二次遇到反斜杠就记录位置，然后break
				break;
			}
		}
	}
	for (int j = 0; j < strlen(path_t); j++)
	{
		if(j<=i)
			path_t_r[j] = path_t[j];
		else {
			path_t_r[j] = NULL;
			break;
		}
	}
	base::CharToTchar(path_t_r, base::szPath);
	return;
}
char* base::Path(void) {
	char* szPath_t = new char[MAX_PATH];
	base::TcharToChar(base::szPath, szPath_t);
	return szPath_t;
}
char* base::Drive(void) {
	char* szDrive_t = new char[MAX_PATH];
	base::TcharToChar(base::szFname, szDrive_t);
	return szDrive_t;
}
char* base::Fname(void) {
	char* szFname_t = new char[MAX_PATH];
	base::TcharToChar(base::szFname, szFname_t);
	return szFname_t;
}
char* base::Dir(void) {
	char* szDir_t = new char[MAX_PATH];
	base::TcharToChar(base::szDir, szDir_t);
	return szDir_t;
}
char* base::Ext(void) {
	char* szExt_t = new char[MAX_PATH];
	base::TcharToChar(base::szExt, szExt_t);
	return szExt_t;
}
TCHAR* base::Path_p(void) {
	return base::szPath;
}
TCHAR* base::Drive_p(void) {

	return base::szDrive;
}
TCHAR* base::Fname_p(void) {
	return szFname;
}
TCHAR* base::Dir_p(void) {

	return base::szDir;
}
TCHAR* base::Ext_p(void) {

	return base::szExt;
}

bool MyFile::getsection(string &str_tmp)
{
	if (str_tmp.find("(") != str_tmp.npos)
	{
		str_tmp = str_tmp.substr(str_tmp.find("(") + 1, str_tmp.find(")") - str_tmp.find("(") - 1);
		return true;
	}
	return false;
}
void MyFile::GetWorkList(std::string strWorkPath, std::vector<string>& vctWorkList, bool bSeachChild)
{
	{
		WIN32_FIND_DATAA FileData;
		HANDLE FileHandle = FindFirstFileA((strWorkPath + "*.*").c_str(), &FileData);
		if (FileHandle == INVALID_HANDLE_VALUE)
		{
			return;
		}
		do
		{
			if (strcmp(FileData.cFileName, ".") == 0 || strcmp(FileData.cFileName, "..") == 0)
			{
				continue;
			}
			else if ((FileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
			{
				if (bSeachChild)
				{
					GetWorkList(strWorkPath + FileData.cFileName + "\\", vctWorkList, bSeachChild);
				}
			}
			else
			{
				vctWorkList.push_back(strWorkPath + FileData.cFileName);
			}
		} while (FindNextFileA(FileHandle, &FileData) != 0);
		FindClose(FileHandle);
		return;
	}
}
bool MyFile::backupfile(const char* FiletoPath)
{
	return true;
}
bool MyFile::CopyFileTo(const char* Path,const char* FiletoPath)
{
	TCHAR* tcharPathwithFname = new TCHAR[MAX_PATH];
	TCHAR* tcharFiletoPath = new TCHAR[MAX_PATH];

	TCHAR szPath_t[_MAX_PATH] = { 0 };
	TCHAR szDrive_t[_MAX_DRIVE] = { 0 };
	TCHAR szDir_t[_MAX_DIR] = { 0 };
	TCHAR szFname_t[_MAX_FNAME] = { 0 };
	TCHAR szExt_t[_MAX_EXT] = { 0 };

	base filebackup;
	
	bool is_readonly = true;
	char szPath_t_chr[_MAX_PATH] = { 0 };
	
	CharToTchar(Path, tcharPathwithFname);
	CharToTchar(FiletoPath, tcharFiletoPath);
	APEXFile apexfile;

	
	wstring szPath_t_wstr = tcharPathwithFname;

	_wsplitpath_s(tcharFiletoPath, szDrive_t, szDir_t, szFname_t, szExt_t);
	wcscpy_s(szPath_t, szDrive_t);
	wcscat_s(szPath_t, szDir_t);
	szPath_t_wstr = szPath_t;
	szPath_t_wstr.erase(szPath_t_wstr.end() - 1);
	MyFile::TcharToChar((TCHAR*)szPath_t_wstr.c_str(),szPath_t_chr);

	int* returnvalue ;//0位置表示是否存在，1位置表示是否覆盖

	filebackup.split_path(tcharFiletoPath);
	
	returnvalue = MyFile::IfPathExist(FiletoPath);
	if (returnvalue[0] && returnvalue[1])//存在且覆盖
	{
		{//覆盖文件前备份原文件
			char* backuppath = new char[MAX_PATH];
			filebackup.eraseenddir_p();
			backuppath = filebackup.TcharToChar(filebackup.Path_p());
			strcat_s(backuppath, strlen(backuppath) + strlen("backup") + 1, "backup");
			filebackup.buildmulitydir(backuppath);
			strcat_s(backuppath, strlen(backuppath) + strlen("\\") + 1, "\\");
			strcat_s(backuppath, strlen(backuppath) + strlen(filebackup.Fname()) + 1, filebackup.Fname());
			strcat_s(backuppath, strlen(backuppath) + strlen(filebackup.Ext()) + 1, filebackup.Ext());
			if (CopyFile(tcharFiletoPath, filebackup.CharToTchar(backuppath), FALSE))
			{
				wstring szPath_to_wstr = tcharFiletoPath;
				int len = 2;//GET_ARRAY_LEN(APEXFile::filedontneedtobereadonly) - 1;
				for (int i = 0; i < len; i++)
				{
					if (szPath_to_wstr.find(CharToTchar(apexfile.filedontneedtobereadonly[i])) != szPath_to_wstr.npos)
					{
						is_readonly = false;
					}

				}
				if (is_readonly)
					base::setfileAttributes(tcharFiletoPath,true);//将复制后（to）的文件设置属性为只读
				cout << endl << "        " << "成功备份文件到:" << backuppath << endl;
			}
			else
			{
				cout << endl << "        " << "备份失败" << endl;
			}
		}
		if (CopyFile(tcharPathwithFname, tcharFiletoPath, FALSE))
		{
			wstring szPath_to_wstr = tcharFiletoPath;
			int len = 2;//GET_ARRAY_LEN(APEXFile::filedontneedtobereadonly) - 1;
			for (int i = 0; i < len; i++)
			{
				if (szPath_to_wstr.find(CharToTchar(apexfile.filedontneedtobereadonly[i])) != szPath_to_wstr.npos)
				{
					is_readonly = false;
				}

			}
			if (is_readonly)
				base::setfileAttributes(tcharFiletoPath,true);//将复制后（to）的文件设置属性为只读
			cout << endl << "        " << "成功复制文件到" << FiletoPath << endl << endl;
			return true;
		}
		else
		{
			cout << endl << "        " << "文件复制失败，目的地址:" << FiletoPath << "存在同名的  只读   文件" << endl;
			cout << endl << "        " << "请删除目的地址文件后重试，或修改文件为非只读模式，后重试" << endl;
			return	false;
		}
	}
	else if (returnvalue[0] && (!returnvalue[1]))//存在但不覆盖
	{
		cout << endl << "        " << "文件复制失败,用户取消覆盖" << endl << endl;
		return true;//do nothing
	}
	else if (!returnvalue[0])//不存在
	{
		if (MyFile::buildmulitydir(szPath_t_chr))
		{
			if (CopyFile(tcharPathwithFname, tcharFiletoPath, TRUE))
			{
				cout << endl << "        " << "成功复制文件到" << FiletoPath << endl << endl;
				return true;
			}
			else
			{
				cout << endl << "        " << "文件复制失败，目的地址:" << FiletoPath << "存在同名的  只读   文件" << endl;
				cout << endl << "        " << "请删除目的地址文件,或修改文件为非只读模式，后重试" << endl;
				return	false;
			}
		}
	}
	else
		return true;
	delete[] tcharPathwithFname;
	delete[] tcharFiletoPath;
	
}
void MyFile::readsection_cfg(ifstream& fpo, int sectid)
{
	char tmp[_MAX_PATH] = { 0 };
	string path_t = MyFile::Path();
	string path_tmp = "";
	string path = "";
	MyFile::numofcfg = 0;
	if (sectid == cfg)
	{
		while (fpo.getline(tmp, _MAX_PATH))
		{
			if (strcmp(tmp, "{") == 0)
			{
				continue;
			}
			else if (strcmp(tmp, "}") == 0)
			{
				break;
			}
			else
			{
				vct_target_strFilePathList.push_back(tmp);
				path_tmp = tmp;
				path = path_t + path_tmp;
				vct_strFilePathList.push_back(path);
				numofcfg++;//记录读了几个apex cfg文件
			}
		}
	}
}
void MyFile::readsection_profile(ifstream& fpo, int sectid)
{
	char tmp[_MAX_PATH] = { 0 };
	string path_t = MyFile::Path();
	string path_tmp = "";
	string path = "";
	if (sectid == profile)
	{
		while (fpo.getline(tmp, _MAX_PATH))
		{
			if (strcmp(tmp, "{") == 0)
			{
				continue;
			}
			else if (strcmp(tmp, "}") == 0)
			{
				break;
			}
			else
			{
				vct_target_strFilePathList.push_back(tmp);
				path_tmp = tmp;
				path = path_t + path_tmp;
				vct_strFilePathList.push_back(path);
				numofprofile++;//记录读了几个apex cfg文件
			}
		}
	}
	else
		cout << "sectid:" << sectid << "wrong";
}
void MyFile::readsection_local(ifstream& fpo, int sectid)
{
	char tmp[_MAX_PATH] = { 0 };
	string path_t = MyFile::Path();
	string path_tmp = "";
	string path = "";
	if (sectid == local)
	{
		while (fpo.getline(tmp, _MAX_PATH))
		{
			if (strcmp(tmp, "{") == 0)
			{
				continue;
			}
			else if (strcmp(tmp, "}") == 0)
			{
				break;
			}
			else
			{
				vct_target_strFilePathList.push_back(tmp);
				path_tmp = tmp;
				path = path_t + path_tmp;
				vct_strFilePathList.push_back(path);
				numoflocal++;//记录读了几个apex cfg文件
			}
		}
	}
	else
		cout << "sectid:" << sectid << "wrong";
}
void MyFile::readsection_target_v(ifstream& fpo, int sectid)
{
	char tmp[_MAX_PATH] = { 0 };
	string path_t = MyFile::Path();
	string path_tmp = "";
	string path = "";
	if (sectid == target_v)
	{
		while (fpo.getline(tmp, _MAX_PATH))
		{
			if (strcmp(tmp, "{") == 0)
			{
				continue;
			}
			else if (strcmp(tmp, "}") == 0)
			{
				break;
			}
			else
			{
				int i = MyFile::numofcfg;
				int j = 0;
				for (i; i < vct_target_strFilePathList.size()&& j < MyFile::numofprofile; i++, j++)
				{
						path = tmp;
						path += "\\";
						path += vct_target_strFilePathList[i];
						vct_target_strFilePathList[i] = path;
				}
				j = 0;
				for (i; i < vct_target_strFilePathList.size() && j < MyFile::numoflocal; i++, j++)
				{
						path = tmp;
						path += "\\";
						path += vct_target_strFilePathList[i];
						vct_target_strFilePathList[i] = path;
				}
				
			}
		}
	}
	else
		cout << "sectid:" << sectid << "wrong";
}
void MyFile::readsection_target_c(ifstream& fpo, int sectid)
{
	char tmp[_MAX_PATH] = { 0 };
	string path_t = MyFile::Path();
	string path_tmp = "";
	string path = "";
	if (sectid == target_c)
	{
		while (fpo.getline(tmp, _MAX_PATH))
		{
			if (strcmp(tmp, "{") == 0)
			{
				continue;
			}
			else if (strcmp(tmp, "}") == 0)
			{
				break;
			}
			else
			{
				for (int i = 0; i<vct_target_strFilePathList.size()&& i< MyFile::numofcfg ;i++)
				{
						path = tmp;
						path += "\\";
						path += vct_target_strFilePathList[i];
						vct_target_strFilePathList[i] = path;
				}
			}
		}
	}
	else
		cout << "sectid:" << sectid << "wrong";
}
void MyFile::readsection(ifstream &fpo, int sectid)
{
	char tmp[_MAX_PATH] = { 0 };
	string path_t = MyFile::Path();
	string path_tmp = "";
	string path = "";
	if (sectid > 0)
	{
		switch (sectid)
		{
		case cfg:
			readsection_cfg(fpo, cfg);
			break;
		case profile:
			readsection_profile(fpo, profile);
			break;
		case local:
			readsection_local(fpo, local);
			break;
		case target_v:
			readsection_target_v(fpo, target_v);
			break;
		case target_c:
			readsection_target_c(fpo, target_c);
			break;
		}
	}
	else
		cout << "sectid:" << sectid << "wrong";

}
bool MyFile::ReadFileList(void)
{
	ifstream fpo;
	string filelist_n = "FileCopyList.txt";
	string filelist_n_t = MyFile::Dir();
	filelist_n_t += filelist_n;
	fpo.open(filelist_n_t,ios::in);
	if (!fpo)
	{
		cout << "FileCopyList.txt not found" << endl;
		return false;
	}
	else
	{
		while (!fpo.eof())
		{
			string temp;
			getline(fpo, temp);
			int sectid = 0;
			if (temp.length() == 0)
			{
				continue;
			}
			else
			{
				if (MyFile::getsection(temp))
				{
					sectid = atoi((char*)temp.c_str());
					readsection(fpo, sectid);
				}
			}
		}
		fpo.close();
		return true;
	}
	
}
bool MyFile::GetCurrentPath(void)
{
	TCHAR OnlyPath[_MAX_PATH];
	GetModuleFileName(NULL, MyFile::szPath, sizeof(MyFile::szPath));

	wcscpy_s(MyFile::szPathwithFilename,MyFile::szPath);

	_wsplitpath_s(MyFile::szPath, MyFile::szDrive, MyFile::szDir, MyFile::szFname, MyFile::szExt);
	wcscpy_s(OnlyPath, MyFile::szDrive);
	wcscat_s(OnlyPath, MyFile::szDir);
	wcscpy_s(MyFile::szPath, OnlyPath);

	return true;
}
bool MyFile::IfPathExist(char* path_str)
{
	const string strPath = path_str;
	WIN32_FIND_DATA  wfd;
	bool  rValue = false;
	HANDLE hFind = FindFirstFile((LPCWSTR)strPath.c_str(), &wfd);
	if ((hFind != INVALID_HANDLE_VALUE) && (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
	{
		rValue = true;
	}
	FindClose(hFind);
	return  rValue;
}
int* MyFile::IfPathExist(const string& strPath)
{
	static bool pass = false;

	static int* returnvaluie = new int[2];

	bool isExist = false;
	static bool iscover = false;
	ifstream isfileE;

	char* input = new char[100];

	isfileE.open(strPath.c_str(), ios::in);
	if (isfileE.is_open())
	{
		while (!pass)
		{
			while (!isExist)
			{
				cout << "路径:" << strPath << "下存在同名路径是否覆盖:" << endl;
				cout << "y/n" << endl;
				cin >> input; cin.get();
				if (strcmp(input, "y") == 0)
				{
					iscover = true;
					isExist = true;
				}
				else if (strcmp(input, "n") == 0)
				{
					isExist = true;
					iscover = false;
					break;
				}
				else if (strcmp(input, "yy") == 0)
				{
					isExist = true;
					iscover = true;
					pass = true;
					break;
				}
				else if (strcmp(input, "nn") == 0)
				{
					isExist = false;
					pass = true;
					iscover = false;
					break;
				}
				else
					continue;
			}
			break;
		}
		if (pass)
			isExist = true;
	}
	else
	{
		isExist = false;
	}
	isfileE.close();
	delete[] input;

	returnvaluie[0] = isExist;
	returnvaluie[1] = iscover;
	return returnvaluie;
}
bool MyFile::modifyContentInFile(string& fileName, string &pos_str, string& content)
{
	static ifstream in;
	static ofstream out;
	char line[_MAX_PATHWITHNAME];
	in.open(fileName, ios::in);
	int i = 0;
	char* outputdata = new char;
	vector<string> readeddata;
	string linedata = "";
	bool readflag = false;
	while (in.getline(line, sizeof(line)))
	{
		linedata = line;
		if (linedata.find(pos_str)!= linedata.npos)
		{
			readeddata.push_back(content);
		}
		else
		{
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
			out << outputdata << endl;
		}
		out.close();
		return true;
	}
	else
	{
		return false;
	}


}
LPCWSTR stringToLPCWSTR(std::string orig)
{
	size_t origsize = orig.length() + 1;
	const size_t newsize = 100;
	size_t convertedChars = 0;
	wchar_t* wcstring = (wchar_t*)malloc(sizeof(wchar_t) * (orig.length() - 1));
	mbstowcs_s(&convertedChars, wcstring, origsize, orig.c_str(), _TRUNCATE);

	return wcstring;
}

const char** APEXFile::filedontneedtobereadonly_p(void) 
{
	return APEXFile::filedontneedtobereadonly;
}
const char** APEXFile::resulotion_file_p(void)
{
	return APEXFile::resulotion_file;
}
const char** APEXFile::resulotion_width_p(void)
{
	return APEXFile::resulotion_width;
}
const char** APEXFile::resulotion_height_p(void)
{
	return APEXFile::resulotion_height;
}
void  APEXFile::getres()
{
	string Userres ;
	string resWidth ;
	string resHeight ;
	size_t  size_of_vec;
	cout << endl <<"     " << "请输入游玩分辨率（使用英文的星号*代替乘号）,例如2304*1440" << endl;
	cin >> Userres; cin.get();
	resWidth = APEXFile::resulotion_width_p()[0];
	resHeight = APEXFile::resulotion_height_p()[0];
	APEXFile::videoconfig = base::stringSplit(Userres, '*');
	size_of_vec = videoconfig.size();

	for (size_t i = 0; i < size_of_vec; i++)//把用户输入分辨率替换到vector里面去
	{
		if (i == 0)
		{
			resWidth.replace(resWidth.find("$"), 1, APEXFile::videoconfig[i]);
			videoconfig.erase(videoconfig.begin());
			videoconfig.push_back(resWidth);
		}
		else if (i == 1)
		{
			resHeight.replace(resHeight.find("$"), 1, APEXFile::videoconfig[i-1]);
			videoconfig.erase(videoconfig.begin());
			videoconfig.push_back(resHeight);
		}
		else
			break;
	}
	return;
}
int main()
{
	vector<string> vct_strFilePath;
	APEXFile myFile;

	//获取当前路径
	myFile.GetCurrentPath();
	myFile.getres();
	if(myFile.ReadFileList())
	{
		//使用system接口调用脚本复制文件
		for (size_t i = 0; i < myFile.vct_strFilePathList.size(); i++)
		{
			string resulotion_file_p = myFile.resulotion_file_p()[0];
			if (IFlist(myFile, i).find(resulotion_file_p) != IFlist(myFile, i).npos)
			{
				string width = (char*)"\"setting.defaultres\"";
				string height = (char*)"\"setting.defaultresheight\"";
				myFile.modifyContentInFile(IFlist(myFile, i), width, myFile.videoconfig[0]);
				myFile.modifyContentInFile(IFlist(myFile, i), height, myFile.videoconfig[1]);
			}
			myFile.CopyFileTo(IFlist_str(myFile,i), TOlist_str(myFile, i));
		}
	}
	system("pause");
	return 0;
}
