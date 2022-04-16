#include <vector>
#include <algorithm>
#include <iostream>

int main()
{
    std::vector<int> v{ 1,2,3 };
    struct Local {
        bool operator()(int n, int m) {
            return n > m;
        }
    };
    std::sort(v.begin(), v.end(), Local()); // since C++11
    for (int n : v) std::cout << n << ' ';
}