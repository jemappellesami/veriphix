OPENQASM 2.0;
include "qelib1.inc";
qreg q955[4];
rz(pi/2) q955[3];
rx(3*pi/2) q955[3];
cx q955[3],q955[2];
cx q955[1],q955[2];
cx q955[0],q955[1];
