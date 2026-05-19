OPENQASM 2.0;
include "qelib1.inc";
qreg q600[5];
rz(3*pi/2) q600[4];
cx q600[3],q600[4];
cx q600[3],q600[2];
cx q600[1],q600[2];
cx q600[1],q600[0];
