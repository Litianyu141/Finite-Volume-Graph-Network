
@echo off
rem 声明采用UTF-8编码
chcp 65001

ECHO 检测当前路径下是否存在build目录

IF NOT EXIST "build\" ( 
	IF EXIST "build" (
		echo.
		ECHO 检测到存在一个build文件，请删除后尝试。
		echo.
		pause
		@echo on
		EXIT /B 1
	)	ELSE (
		echo.
		ECHO build目录不存在，尝试创建。
		echo.
		MKDIR build
	)
)

ECHO 进入build目录，创建工程
echo.
CD source

cmake -A x64
if %ERRORLEVEL% EQU 0(cmake --open .
) else (PAUSE)
@echo on
PAUSE