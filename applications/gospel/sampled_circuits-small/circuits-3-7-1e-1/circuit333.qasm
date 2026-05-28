OPENQASM 2.0;
include "qelib1.inc";
qreg q334[3];
rx(pi) q334[2];
rz(pi) q334[2];
rx(3*pi/2) q334[2];
cx q334[1],q334[2];
cx q334[0],q334[1];
