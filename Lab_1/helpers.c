

int rectify(int c) {
    return (c >= 127) ? c : 127;
}

unsigned char pickLargest(unsigned char j, unsigned char k, unsigned char l, unsigned char m) {
    unsigned char largest = j;
    printf("%d %d %d %d \n", j, k, l, m);
    if (k > largest) {
        largest = k;
    }

    if (l > largest) {
        largest = l;
    }

    if (m > largest) {
        largest = m;
    }
    printf("%d \n", largest);
    return largest;
}

unsigned char convolve(unsigned char input[][], unsigned char weight[][], int i, int ii, int j, int jj) {
    return input[i + ii - 1][j + jj - 1] * weight[ii][jj];
}