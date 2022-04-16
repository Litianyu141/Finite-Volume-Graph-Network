#include<iostream>
using namespace std;
int main() {
	int x = __cplusplus;
	if (x < 199711)
		cout << "C++ version is too old" << endl;
	switch (x) {
		case 199711:
			cout << "C++98" << endl;
			break;
		case 201103:
			cout << "C++11" << endl;
			break;
		case 201402:
			cout << "C++14" << endl;
			break;
		defaut:
			cout << "later than C++14" << endl;
	}

	return 0;
}