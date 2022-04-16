#include <stdio.h>
#include <stdarg.h>
void print1(const char* value, ...)
{
    char* t;
    printf(value);
    va_list val;
    va_start(val, value);
    for (int i = 5; i > 0; i--)
    {
        t = va_arg(val, char*);//获取下一个参数需要赋值的。
        if (t != NULL)
            printf(t);
        else
            printf("NULL");
    }

    va_end(val);
    
}
int main() {

    const char* x1, * x2, * x3, * x4;

    x1 = "1";
    x2 = "2";
    x3 = "3";
    x4 = "4";
    print1(x1, x2, x3);

    return 0;
}