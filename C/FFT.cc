#include <iostream>
#include <cmath>
#include <string>

#define PI 3.141592653589793
#define N 10000
using namespace std;


struct Complex {
    double r, i;
    Complex(): r(0), i(0) {}
    Complex(double real, double imag): r(real), i(imag) {}
    inline Complex operator + (const Complex& rhs) const {
        return Complex(r + rhs.r, i + rhs.i);
    }
    inline Complex operator - (const Complex& rhs) const {
        return Complex(r - rhs.r, i - rhs.i);
    }
    inline Complex operator * (const Complex& rhs) const {
        return Complex(r * rhs.r - i * rhs.i, r * rhs.i + i * rhs.r);
    }
    inline void operator += (const Complex& rhs) {
        r += rhs.r;
        i += rhs.i;
    }
    inline void operator /= (const double &x) {
        r /= x;
        i /= x;
    }
    inline Complex conj() {
        return Complex(r, -i);
    }
    inline string info() {
        return to_string(r) + " + " + to_string(i) + "i";
    }
};


struct FastFourierTransform {
    Complex omega[N], omega_inverse[N];
    FastFourierTransform(const int& n) {
        for (int i = 0; i < n; ++i) {
            omega[i] = Complex(cos(2 * PI / n * i), sin(2 * PI / n * i));
            omega_inverse[i] = omega[i].conj();
        }
    }

    // n = 1 << k
    void transform(Complex *a, const int& n, Complex *omega) {
        // Rearange Complex *a
        for ( int i = 0, j = 0 ; i < n ; ++ i )  {
		    if (i > j) swap(a[i], a[j]) ;
		    for(int l = n >> 1; (j ^= l) < l; l >>= 1);
	    }
        for (int l = 2; l <= n; l <<= 1) {
            int m = l / 2;
            for (Complex *p = a; p != a + n; p += l) {
                for (int i = 0; i < m; ++i) {
                    Complex t = omega[n / l * i] * p[m + i];
                    p[m + i] = p[i] - t;
                    p[i] += t;
                }
            }
        }
    }

    void dft(Complex *a, const int& n) {
        transform(a, n, omega);
    }

    void idft(Complex *a, const int& n) {
        transform(a, n, omega_inverse);
        for (int i = 0; i < n; ++i) a[i] /= (double)n;
    }
};

int main() {
    const int n = 1 << 3;
    FastFourierTransform fft = FastFourierTransform(n);
    Complex *a = new Complex[n];
    for (int i = 0; i < n; ++i) {
        a[i].r = i;
        a[i].i = 10 * i;
    }
    for (int i = 0; i < n; ++i) {
        cout << a[i].info() << endl;
    }
    cout << endl;
    fft.dft(a, n);
    for (int i = 0; i < n; ++i) {
        cout << a[i].info() << endl;
    }
    cout << endl;
    fft.idft(a, n);
    for (int i = 0; i < n; ++i) {
        cout << a[i].info() << endl;
    }
    return 0;
}