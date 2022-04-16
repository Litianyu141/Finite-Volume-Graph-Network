#define _MAX_PATHWITHNAME 516

#define IFlist(objname,number) objname.vct_strFilePathList[number]
#define TOlist(objname,number) objname.vct_target_strFilePathList[number]
#define IFlist_str(objname,number) objname.vct_strFilePathList[number].c_str()
#define TOlist_str(objname,number) objname.vct_target_strFilePathList[number].c_str()
#define GET_ARRAY_LEN(array) (sizeof(array) / sizeof(array[0]))	