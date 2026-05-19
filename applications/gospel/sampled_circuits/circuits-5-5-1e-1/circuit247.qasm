OPENQASM 2.0;
include "qelib1.inc";
qreg q248[5];
rx(3*pi/4) q248[2];
cx q248[3],q248[4];
cx q248[3],q248[2];
cx q248[2],q248[1];
cx q248[1],q248[0];
