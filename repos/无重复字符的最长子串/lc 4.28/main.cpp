#include<string>
#include<vector>
#include<map>
#include <unordered_map>
using namespace std;

class Solution {
public:
    int lengthOfLongestSubstring(string s) 
    {
        int length =1;
        int length_max = 0;
        unordered_map<char, int> stringmap;
        int pos =0;
        int j = 0;
        for (int i = 0; i < s.size(); i++)
        {
            i += pos;
            for (string::iterator it = s.begin()+ i; it != s.end(); pos++,it++)
            {
                if (stringmap.size() == 0)
                    stringmap.insert(pair<char, int>(*it, pos));
                else
                {
                    auto iter = stringmap.find(*it);
                    if (iter != stringmap.end())
                    {
                        stringmap.clear();
                        if (length > length_max)
                        {
                           length_max = length;
                        }
                        length = 0;
                        break;
                    }
                    else
                    {
                        length++;
                        stringmap.insert(pair<char, int>(*it, pos));
                    }
                }

            }
            pos = 0;
            stringmap.clear();
            if (length > length_max)
            {
                length_max = length;
            }
            length = 1;
        }
        return length_max;

    }
};
int main()
{
    Solution betest;
    string test = "pwwkew";
    betest.lengthOfLongestSubstring(test);
}