

int rectify(int c) {
    return (c >= 127) ? c : 127;
}

unsigned char pickLargest(unsigned char j, unsigned char k, unsigned char l, unsigned char m) {
    unsigned char largest = j;
    if (k > largest) {
        largest = k;
    }

    if (l > largest) {
        largest = l;
    }

    if (m > largest) {
        largest = m;
    }
    
    return largest;
}

int clamp(int c) {
    if (c < 0)
        return 0;
    if (c > 255)
        return 255;
    return c;
}