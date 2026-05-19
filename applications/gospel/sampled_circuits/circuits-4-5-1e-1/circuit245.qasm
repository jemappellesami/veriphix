OPENQASM 2.0;
include "qelib1.inc";
qreg q246[4];
rz(5*pi/4) q246[3];
cx q246[3],q246[2];
cx q246[1],q246[2];
cx q246[1],q246[0];
