#include<iostream>
#include<string>
using namespace std;

std::string subreplace(std::string resource_str, std::string sub_str, std::string new_str)
{
    std::string dst_str = resource_str;
    std::string::size_type pos = 0;
    while ((pos = dst_str.find(sub_str)) != std::string::npos)   //替换所有指定子串
    {
        dst_str.replace(pos, sub_str.length(), new_str);
    }
    return dst_str;
}
void main() 
{
    string fnd = "dataD::\\setD::\\setD::\\setD::\\setD::\\setD::\\set";
    string rep = "\\";
    string new_str = "/";
    string res = " ";
    res = subreplace(fnd, rep, new_str);
    exit(0);
    return;
   
}