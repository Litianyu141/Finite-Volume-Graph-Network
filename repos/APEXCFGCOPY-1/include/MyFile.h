#include <iostream>
#include <string>
#include <vector>
#include <windows.h>
#include <macro.h>
using namespace std;


/*
@ Brief:    APEX的CFG复制器
@ Author : Litianyu
@ Created : 2022 / 04 / 02
@ Return:
*/
enum sectname 
{ 
	cfg = 1000, 
	profile,
	local,
	target_c = 2000,
	target_v 
};
class base
{
public:
	TCHAR szPathwithFilename[_MAX_PATHWITHNAME] = { 0 };
public:
	//@主构造函数
	void split_path(TCHAR *szPathwithFilename);

	//@分词
	std::vector<std::string>  stringSplit(const std::string& strIn, char delim);


	bool CharToTchar(const char* _char, TCHAR* tchar);
	bool TcharToChar(TCHAR* tchar, char* _char);
	char* TcharToChar(TCHAR* tchar);
	TCHAR* CharToTchar(const char* _char);
	string CharToStr(char* contentChar);
	wstring stringToWstring(const std::string& str);
	bool buildmulitydir(char* strPath);
	bool setfileAttributes(TCHAR* path,bool isreadoly);
	void eraseenddir_p(void);

	char* Path();
	char* Drive();
	char* Fname();
	char* Dir();
	char* Ext();
	
	TCHAR* Path_p();
	TCHAR* Drive_p();
	TCHAR* Fname_p();
	TCHAR* Dir_p();
	TCHAR* Ext_p();
	
	//@析构函数
	~base();
protected:
	TCHAR szbackupPath[_MAX_PATH] = { 0 };
	TCHAR szPath[_MAX_PATH] = { 0 };
	TCHAR szDrive[_MAX_DRIVE] = { 0 };
	TCHAR szDir[_MAX_DIR] = { 0 };
	TCHAR szFname[_MAX_FNAME] = { 0 };
	TCHAR szExt[_MAX_EXT] = { 0 };
	string CurrentFilepath;
	string CurrentWorkFilepath;
	int numofcfg;
	int numofprofile;
	int numoflocal;
};
class MyFile:public base
{
public:
	//@获取当前路径,构造函数
	bool GetCurrentPath(void);

	//@重载，判断目标路径文件是否存在
	bool IfPathExist(char* path_str);
	bool IfPathExist(TCHAR* path_str);
	int* IfPathExist(const string& strPath);
	bool IfPathExist(const wstring& strPath);


	//@构造函数
	void GetWorkList(std::string strWorkPath, std::vector<string>& vctWorkList, bool bSeachChild);
	
	bool CopyFileTo(const char* Path, const char* FiletoPath);
	bool backupfile(const char* FiletoPath);



	bool ReadFileList(void);//list should be in the same path with the program


	bool getsection(string &str_tmp);
	void readsection(ifstream &fpo,int sectid);
	void readsection_cfg(ifstream& fpo, int sectid);
	void readsection_profile(ifstream& fpo, int sectid);
	void readsection_local(ifstream& fpo, int sectid);
	void readsection_target_v(ifstream& fpo, int sectid);
	void readsection_target_c(ifstream& fpo, int sectid);

	bool modifyContentInFile(string& fileName, string& pos_str, string& content);

	~MyFile();
public:
	
	string FileOutputPath;
	vector<string> vct_strFilePathList;
	vector<string> vct_target_strFilePathList;
};
class APEXFile :public MyFile
{
public:
	char* userresolution;
	vector<string> videoconfig;
public:
	const char** filedontneedtobereadonly_p(void);
	const char** resulotion_file_p(void);
	const char** resulotion_width_p(void);
	const char** resulotion_height_p(void);

	void getres(void);
public:
	const char* filedontneedtobereadonly[3] = { "settings.cfg","previousgamestate.txt",NULL };
	const char* resulotion_file[1] = { "videoconfig.txt" };
	const char* resulotion_width[1] = { "\"setting.defaultres\"        \"$\"" };
	const char* resulotion_height[1] = { "\"setting.defaultresheight\"      \"$\""};
};