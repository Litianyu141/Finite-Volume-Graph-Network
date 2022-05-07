#include<iostream>
#include<vector>
using namespace std;
class Solution 
{
public:
    vector<int> sortArrayByParity(vector<int>& nums)
    {
        int partition = 0;
        int pos = 0;
        if (nums.empty())
            return nums;
        for (vector<int>::iterator it = nums.begin(); it != nums.end(); it++)
        {
            if ((*it) % 2 == 0)
            {
                partition = *it;
                nums.erase(it);
                nums.insert(nums.begin() + pos, partition);
                pos++;
            }
        }
        return nums;
    }

};

int main() 
{
    Solution betest;
    vector<int> test = { 3,2,1,4 };
    vector<int> res;
    res = betest.sortArrayByParity(test);

}
